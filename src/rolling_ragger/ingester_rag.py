import logging

logger = logging.getLogger(__name__)
logger.propagate = False
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('(%(funcName)s:%(lineno)d) %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
# --- KEEPING FILE OUTPUT COMMENTED BUT WORKING ---
# file_handler = logging.FileHandler('rag_ingester.log')
# file_handler.setLevel(logging.DEBUG)
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)

from typing import Any, Dict, List, Optional, Tuple
import os
import sys
import asyncio
import gzip
import math
import httpx

# Make the parent 'src' directory importable so that the modules moved there
# (pg_driver, rabbit_driver) can be imported as top-level modules.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pg_driver import PostgresDBClient, ChunkRow
from rabbit_driver import RabbitMQClient
# The jina-v3 embedding servitor endpoint (the shared LLM config module -
# the endpoint is configured in ONE place and the agent imports it).
from llm_config import EMBEDDING_BASE_URL, EMBEDDING_API_KEY, EMBEDDING_MODEL_NAME, EMBEDDING_DIM


# =====================================================================
# RAG ingestion pipeline
# ---------------------------------------------------------------------
# Purpose:
#   Consumes the processed stories published by the scrapper (one
#   message per saved story: the file reference + the source url + the
#   title + the crawl date + the perplexity metrics - no story text)
#   and indexes every story into the RAG Postgres database: the doc
#   row (docs) + the story's chunks (chunks) - the equal-size
#   overlapping windows (a fixed token overlap) in the embedding
#   model's token space, with the precomputed jina-v3 embeddings and
#   the verbatim story text of every chunk (raw_text, for BM25).
#
# Data flow:
#   scrapper_json_queue (RabbitMQ, published by the scrapper - every
#        |               message names a saved .txt.gz story file + the
#        |               source url + the title + the crawl date + the
#        |               perplexity metrics)
#        v
#   the story file (news/<crawl date>/<uuid>.txt.gz, gzip, utf-8)
#        v
#   the jina-v3 embedding servitor (vLLM, ifo4:8001):
#        |   /tokenize - the story in the embedding model's own token
#        |   space (the context window is a property of this
#        |   tokenization) + the equal-size windowing in token space
#        |   (a fixed token overlap between the neighbours)
#        |   + /detokenize - the verbatim window text (the exact text
#        |   embedded) + /v1/embeddings - the precomputed 1024-dim
#        |   chunk vectors
#        v
#   rag_db (Postgres + pgvector + pg_textsearch, the schema is owned
#        |   by the pg_driver's ensure_schema):
#        +--> docs   (id, url UNIQUE, title, date, the ppl_* metrics)
#        +--> chunks (doc_id FK, the prev/next links, the token
#                     offset/len in the story's own tokenization
#                     (reproducible from the story file), the 1024-dim
#                     embedding (HNSW) + raw_text (BM25))
#
# NOTE: in DEBUG (readonly) the run is non-destructive end to end: the
#       queue is read one message at a time and every message is put
#       back (never drained), the Postgres writes are buffered (the
#       url dedup is still checked against the real tables), and the
#       run stops as soon as the queue cycles back to an already-seen
#       file.
#
# NOTE: ALL queue reads use an ENFORCED MANUAL (deferred)
#       acknowledgment (the queue driver's discipline): a message is
#       pulled unacknowledged and stays unacknowledged at the broker
#       until its processing has completed (insert_story committed ->
#       ack_pending; the insert rolled back / the chunking failed ->
#       requeue_pending; unprocessable -> ack_pending) - a worker
#       crash leaves the in-flight message unacknowledged and the
#       broker requeues it for another worker instead of losing it.
#       The ingester is a pure consumer: it has no publish queue.
#
# NOTE: the jina-v3 servitor's wire shapes (probe-confirmed on this
#       deployment):
#       GET  /v1/models      -> data[].id, data[].max_model_len
#       POST /tokenize       {"model", "prompt", "add_special_tokens": false}
#                             -> {"count": int, "tokens": [int]}
#       POST /detokenize     {"model", "tokens": [int]} -> {"prompt": str}
#                             (a range's decode drops the range's LEADING
#                              whitespace - the slice text is stored and
#                              embedded as-is; the chunk's position is
#                              the token span, so this never matters)
#       POST /v1/embeddings  {"model", "input": str | [str]}
#                             -> {"data": [{"embedding": [float x 1024]}],
#                                 "usage": {"prompt_tokens": int}}
#                             (the embeddings endpoint counts 2 extra
#                              special tokens over /tokenize - the
#                              overhead is calibrated at startup, not
#                              hard-coded)
#
# NOTE: the run ends with the BM25 maintenance pass (the memtable merged
#       into the on-disk segment) taken under the RAG index maintenance
#       lock (the pg_driver's RAG_INDEX_LOCK_KEY advisory lock,
#       production mode only). The lock covers the MERGE ONLY (it is
#       held for milliseconds) - the RAG readers (the chat terminal)
#       are never blocked by the data ingestion itself (their reads are
#       snapshot-consistent); a retrieval that lands on the merge waits
#       it out. Concurrent ingester runs are safe: the docs.url unique
#       key + ON CONFLICT DO NOTHING index a story exactly once no
#       matter how many runs see it.
# =====================================================================

# --- The jina-v3 embedding servitor endpoint (EMBEDDING_BASE_URL /
# EMBEDDING_API_KEY / EMBEDDING_MODEL_NAME / EMBEDDING_DIM) is imported
# from llm_config at the top of this file. The chunk embeddings are
# precomputed against THIS model, so the chunking happens in the model's
# own token space (the context window is a property of its tokenizer).
# The per-chunk window is max_model_len - the calibrated special-token
# overhead - this margin: a verbatim slice re-tokenized on its own can
# merge 1-2 boundary tokens differently than in the story's full
# tokenization, and the margin keeps the slice + the special tokens
# under the servitor's context (the max scored total is max_model_len).
EMBED_BUDGET_MARGIN = 64
# The fixed overlap (in tokens) between two consecutive windows: the
# overlap amount is CONSTANT, not a fraction of the window size - so
# it does not depend on the chunk size (every window of a document is
# one size, and every neighbour pair shares the same amount).
CHUNK_OVERLAP_TOKENS = 96
# The number of raw_texts per /v1/embeddings call (the OTHER cap is
# the total input tokens per call - one context window's worth, see
# the batching chapter of build_story_chunks: a batch of several
# near-max-length windows would multiply the per-step time on the
# (slow CPU) servitor and can outlast its per-step RPC deadline).
EMBED_BATCH_SIZE = 8
# --- Liveness tuning (env-overridable) ---
# A call on a busy vLLM queues behind the in-flight batch: the read
# timeout of the servitor client is capped at 1h by default
# (EMBEDDING_READ_TIMEOUT, seconds) - long enough for busy servers,
# short enough to bound a wedged one (the startup health gate is
# covered by it too).
EMBEDDING_READ_TIMEOUT_S = float(os.environ.get("EMBEDDING_READ_TIMEOUT", "3600"))

# --- The RAG database (env-overridable; the defaults match the setup
# scripts: trust auth, the rag_db created by postgre_setup.bash) ---
PG_HOST = os.environ.get("PG_HOST", "localhost")
PG_PORT = int(os.environ.get("PG_PORT", "5432"))
PG_DBNAME = os.environ.get("PG_DBNAME", "rag_db")
PG_USER = os.environ.get("PG_USER", "postgres")

# --- DEBUG FLAG: Use the readonly implementations for testing/debugging ---
# Set the DEBUG environment variable to 'true', '1', or 'yes' (case-insensitive)
# to use the readonly implementations: the queue reads are non-destructive
# (messages are never removed) and the Postgres writes are buffered instead
# of committed (the url dedup is still checked against the real tables).
DEBUG = os.environ.get('DEBUG', '').lower() in ('true', '1', 'yes')

if DEBUG:
    logger.debug("INFO. Running in DEBUG mode.")

# The RabbitMQ queue of the pipeline: the ingester is a pure consumer of
# the processed-stories queue (the scrapper's output - no publish queue).
STORY_QUEUE_NAME = "scrapper_json_queue"

# --- The driver's poison-message bounds ---
# A content-specific chunking/embedding failure (a story the servitor
# cannot index: an empty tokenization, an empty slice text, an
# embedding shape mismatch) is PERSISTENT - the same file fails the
# same way on every
# requeue, so an unbounded requeue would loop the queue forever on the
# same head message. The driver therefore bounds the per-file attempts
# and settles the message as unprocessable (ack) once the bound is
# reached: the story is skipped (loudly logged), the rest of the queue
# keeps draining.
MAX_FILE_RETRIES = 3
# A transient service failure (the embedding servitor or Postgres is
# down / flaky MID-run) is not content-specific: the story must be
# retried, not skipped. It is retried by requeueing, but bounded
# globally - after this many consecutive transient failures the run
# aborts (the in-flight message is left unacknowledged on purpose - the
# broker requeues it on process exit, and a restarted run re-pings the
# services at startup and exits cleanly if they are still down).
MAX_TRANSIENT_RETRIES = 10


async def build_story_chunks(story : str, client : httpx.AsyncClient, token_budget : int) -> Dict[str, Any] :
    """
    Chunk one story and precompute its chunk embeddings. All model
    work goes through the embedding servitor over HTTP (the wire
    contract is documented at the top of this module) - there is no
    client-side tokenizer.

    The chapters:
      1. The story in the model's token space: tokens = POST
         /tokenize (add_special_tokens=false) - the context window is
         a property of this tokenization;
      2. The window spans in token space (pure span math, no I/O):
         N = ceil(n_tokens / (token_budget - overlap)) windows of
         ONE size W = (n_tokens + (N - 1) * overlap) // N (the
         equal-division formula plus each window's share of the
         overlap) - every window of the story is exactly W tokens
         long, the windows tile the story exactly from token 0 to
         the last token, consecutive windows overlap by the fixed
         CHUNK_OVERLAP_TOKENS amount (the integer remainder spreads
         +-1 token over the gaps), and W < token_budget (every
         window fits the embedding context with the margin to
         spare);
      3. The slice texts: per span [s, e), raw_text =
         POST /detokenize(tokens[s:e]) - the EXACT text the embedding
         is computed on (and the raw_text stored for BM25); a chunk's
         position is its span (offset = the first token, len = the
         token count) of the story's own tokenization - exact by
         construction (the slice IS tokens[s:e]), resolvable later by
         re-tokenizing the story file;
      4. The embeddings: POST /v1/embeddings over the slice texts,
         the inputs packed greedily into batches of at most
         EMBED_BATCH_SIZE and at most token_budget total input tokens
         per call (one context window's worth - a batch of several
         near-max-length windows would multiply the per-step time on
         the (slow CPU) servitor and can outlast its per-step RPC
         deadline); the batches and the indices within them are in
         order, so the flattened batch order is the story order; the
         dimension and the count are asserted - the embedding is of
         EXACTLY the stored raw_text.

    Never raises: any failure (dead service, an empty tokenization,
    an empty slice text, an embed shape mismatch) yields
    {"rows": [], "error": <the reason>, "transient": <bool>}.
    """
    try :
        # 1. The story in the embedding model's own token space (the
        #    context window is a property of this tokenization).
        response = await client.post("/tokenize",
                                     json={ "model" : EMBEDDING_MODEL_NAME, "prompt" : story,
                                            "add_special_tokens" : False })
        response.raise_for_status()
        tokens = response.json().get("tokens")
        if not isinstance(tokens, list) or not tokens or not all(isinstance(t, int) for t in tokens) :
            return { "rows" : [], "error" : "no usable /tokenize output", "transient" : False }
        n_tokens = len(tokens)
        # 2. The window spans in token space (N windows of ONE size W
        #    - the equal-division formula plus each window's share of
        #    the fixed overlap; the windows tile the story exactly
        #    from token 0 to the last token, and W < token_budget).
        segment_max = token_budget - CHUNK_OVERLAP_TOKENS
        if n_tokens <= segment_max :
            spans = [ (0, n_tokens) ]
        else :
            n_windows = n_tokens // segment_max + int(n_tokens % segment_max != 0)
            w = (n_tokens + (n_windows - 1) * CHUNK_OVERLAP_TOKENS) // n_windows
            # The integer remainder: (n_windows - 1 - q) of the gaps
            # overlap by the full amount, q of them by one token less.
            q = (n_tokens + (n_windows - 1) * CHUNK_OVERLAP_TOKENS) % n_windows
            full_gaps = n_windows - 1 - q
            spans = []
            s = 0
            for i in range(n_windows) :
                if i == n_windows - 1 :
                    s = n_tokens - w  # the last window ends at the last token
                spans.append((s, s + w))
                if i < n_windows - 1 :
                    s += w - (CHUNK_OVERLAP_TOKENS if i < full_gaps else CHUNK_OVERLAP_TOKENS - 1)
        # 3. The slice texts (the EXACT text each embedding is computed
        #    on - and the raw_text stored for BM25); the chunk's
        #    position is its span (offset = the first token, len = the
        #    token count).
        rows : List[Tuple[int, int, str, int]] = []
        for (s, e) in spans :
            response = await client.post("/detokenize",
                                         json={ "model" : EMBEDDING_MODEL_NAME, "tokens" : tokens[s:e] })
            response.raise_for_status()
            raw = response.json().get("prompt", "")
            if not raw or not raw.strip() :
                return { "rows" : [], "error" : f"empty slice text for the token span [{s}, {e})", "transient" : False }
            rows.append((s, e - s, raw, e - s))
        # 4. The embeddings (the inputs are packed greedily into
        #    batches of at most EMBED_BATCH_SIZE and at most
        #    token_budget total input tokens per call; the dimension
        #    and the count are asserted - the embedding is of
        #    EXACTLY the stored raw_text).
        batches : List[List[int]] = []
        cur : List[int] = []
        cur_tokens = 0
        for i, r in enumerate(rows) :
            if cur and ( len(cur) >= EMBED_BATCH_SIZE or cur_tokens + r[3] > token_budget ) :
                batches.append(cur)
                cur, cur_tokens = [], 0
            cur.append(i)
            cur_tokens += r[3]
        if cur :
            batches.append(cur)
        embeddings : List[List[float]] = []
        for idxs in batches :
            batch = [ rows[i][2] for i in idxs ]
            response = await client.post("/v1/embeddings",
                                         json={ "model" : EMBEDDING_MODEL_NAME, "input" : batch })
            response.raise_for_status()
            data = response.json().get("data", [])
            if len(data) != len(batch) or any(len(d.get("embedding", [])) != EMBEDDING_DIM for d in data) :
                return { "rows" : [], "error" : "unexpected /v1/embeddings shape", "transient" : False }
            for d in data :
                v = d["embedding"]
                if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in v) :
                    return { "rows" : [], "error" : "non-finite embedding values", "transient" : False }
                embeddings.append([ float(x) for x in v ])
        return { "rows" : [ (r[0], r[1], r[2], emb) for r, emb in zip(rows, embeddings) ],
                 "error" : None, "transient" : False }
    except Exception as e :
        # A network / timeout failure is a transient SERVICE failure
        # (the servitor is down or wedged - the story must be retried,
        # not skipped); any other failure is a persistent content
        # failure, bounded per file by the driver (a poison message
        # must not loop the queue forever).
        return { "rows" : [], "error" : str(e), "transient" : isinstance(e, httpx.HTTPError) }


# The RAG ingestion driver
# ------------------------------------------------------------------
# Core batch driver of the pipeline. Workflow:
#   1. External driver loop: feed each story file named in a message
#      of the processed-stories queue through the chunking /
#      embedding / index cycle, one at a time (enforced deferred
#      acknowledgment - the message is settled only after the
#      processing has fully completed).
#   2. For every story: the dedup gate (docs.url - the idempotency
#      key), the story upload from the saved .txt.gz, the equal-size
#      token windows + the precomputed embeddings, and the atomic
#      insert_story settle (the doc + the chunks + the prev/next links
#      in ONE Postgres transaction).
#   3. Log a summary and return the run error code.
# ------------------------------------------------------------------
async def run_rag_ingester(pg_client : PostgresDBClient, rabbit_manager : RabbitMQClient,
                           client : httpx.AsyncClient, token_budget : int,
                           docs_max : int = -1) -> int :
    """
    The driver loop over the processed-stories queue (the analog of
    run_agentic_scraper): the queue's messages name the saved story
    files (no story text - the text is uploaded from file_path); for
    every story the ingester computes the equal-size token windows and
    the chunk embeddings and settles the doc + chunks atomically into
    Postgres (insert_story), with the story's perplexity metrics
    stored alongside (monitoring data).

    The settle policy (the enforced deferred-ack discipline of the
    queue driver):
      * ack_pending  - the story was indexed, or was already indexed
        / unprocessable (the url dedup is the idempotency key, so the
        ack is safe);
      * requeue_pending - the atomic insert failed (rolled back), or
        the dedup check / the chunking failed: the story is
        reprocessed by this or another worker;
      * a worker crash before the settle leaves the message
        unacknowledged - the broker requeues it instead of losing it.
    The poison-message bounds (a persistent failure must not loop the
    queue forever on the same head message):
      * a content-specific chunking/embedding failure is PERSISTENT
        (the same file fails the same way on every requeue): after
        MAX_FILE_RETRIES failed attempts the file's message is
        settled as unprocessable (ack - the story is skipped, loudly
        logged) instead of requeued;
      * a transient service failure (the servitor / Postgres is down
        or flaky mid-run) is not content-specific: the story is
        retried by requeueing, but after MAX_TRANSIENT_RETRIES
        consecutive transient failures the run aborts (the in-flight
        message is left unacknowledged on purpose - the broker
        requeues it on process exit, and a restarted run re-pings the
        services at startup and exits cleanly if they are still down).
    In readonly mode every settle is a non-destructive requeue of the
    head message (the queue is never drained) and the run stops as
    soon as the queue cycles back to an already-seen file (the
    scrapper's seen_file_paths guard).

    Returns:
        0 on a clean run; 4 if the run was aborted on consecutive
        transient failures (a service went down mid-run).
    """
    # The driver owns these counters (the harness never touches them):
    docs_number = 0
    docs_saved  = 0
    # The poison-message bookkeeping (the bounds above): a per-file
    # failure count for the persistent content failures, and a global
    # streak of consecutive transient service failures.
    fail_counts : Dict[str, int] = {}
    transient_streak = 0
    aborted = False

    # External driver loop. The work items are the pending messages of
    # the processed-stories queue (each one names a saved story file
    # via file_path), taken one at a time:
    #   * ALL reads use an ENFORCED deferred (manual) acknowledgment: a
    #     message is pulled unacknowledged and stays unacknowledged at
    #     the broker while it is being processed (each worker holds at
    #     most one such message - prefetch=1); it is settled only after
    #     the processing has fully completed (the settle at the end of
    #     the loop). If a worker crashes mid-processing, the broker
    #     requeues the unacknowledged message for another worker
    #     instead of losing it.
    #   * readonly/debug mode - the settle (ack_pending()) is
    #     non-destructive: the head message is put back onto the queue,
    #     so the queue is never drained and the run can be repeated.
    #     Because the head message is re-read on every iteration, the
    #     same file would otherwise be observed forever: the run
    #     therefore stops as soon as a file seen in this run is
    #     re-observed (the queue has cycled back to it).
    # The loop (the driver's iter_queue_json) stops when the queue is
    # exhausted, the docs_max message budget is spent (checked before the
    # next pull, so nothing is left in flight at the stop), or (readonly)
    # the queue cycles back to an already-seen file.
    async for message in rabbit_manager.iter_queue_json(queue_name=STORY_QUEUE_NAME,
                                                        max_messages=docs_max) :
        file_path = message.get("file_path")
        url = str(message.get("url", "")).strip()
        if not file_path :
            # Unprocessable: settle it NOW (a message left
            # unacknowledged would wedge the run - with prefetch=1 the
            # next read blocks while it is still in flight). The ack
            # consumes it in production; in readonly mode it is a
            # non-destructive requeue and the run stops on the queue
            # cycle.
            logger.error(f"Malformed queue message (no file_path): {message}; "
                         "settling it as unprocessable.")
            await rabbit_manager.ack_pending()
            continue
        docs_number += 1
        logger.debug(f"Read queue message: {os.path.basename(file_path)} (url: {url})")
        requeue = False
        transient = False

        # The dedup gate: the story is indexed exactly once (the unique
        # docs.url) - an idempotent re-run, or another worker got here
        # first.
        exists = await pg_client.doc_exists(file_path)
        if exists == -1 :
            logger.error(f"Postgres dedup check failed for {file_path}; the message will be requeued.")
            requeue = True
            transient = True  # a Postgres failure, not a content failure
        elif exists == 1 :
            logger.info(f"Story already indexed, skipping: {file_path}")
        else :
            # Upload: read the story text from the saved .txt.gz
            story = ""
            try :
                with gzip.open(file_path, "rt", encoding="utf-8") as f :
                    story = f.read()
            except Exception as e :
                logger.error(f"Failed to upload {file_path}, error: {e}")
            if not story or not story.strip() :
                logger.error(f"No content uploaded for {file_path}. Skipping the file.")
            if story :
                # Chunk the story and precompute the chunk embeddings
                # (the model work is the embedding servitor's).
                result = await build_story_chunks(story, client, token_budget)
                if not result["rows"] :
                    logger.error(f"Chunking/embedding failed for {file_path}: {result['error']}; "
                                 "the message will be requeued.")
                    requeue = True
                    transient = result["transient"]
                else :
                    # The crawl date (the queue's timestamp - the
                    # news/<date> folder name, falling back to the story
                    # file's own folder name).
                    date = message.get("timestamp") or os.path.basename(os.path.dirname(file_path))
                    # The story's perplexity metrics (the monitoring data
                    # the scrapper measured pre-publish - stored as-is,
                    # a failed measurement comes out null).
                    ppl = { "ppl_per_word"  : message.get("ppl_per_word"),
                            "ppl_per_token" : message.get("ppl_per_token"),
                            "ppl_tokens"    : message.get("ppl_tokens"),
                            "ppl_words"     : message.get("ppl_words"),
                            "ppls"          : message.get("ppls"),
                            "ppl_frag_p90"  : message.get("ppl_frag_p90") }
                    if message.get("ppl_error") :
                        ppl["ppl_error"] = message["ppl_error"]
                    title = str(message.get("title", "")).strip()
                    chunks = [ ChunkRow(offset=r[0], length=r[1], raw_text=r[2], embedding=r[3])
                               for r in result["rows"] ]
                    # The atomic settle: the doc + the chunks + the
                    # prev/next links commit in ONE Postgres transaction
                    # (a crash can never leave a document without its
                    # chunks - which the url dedup would skip forever).
                    doc_id = await pg_client.insert_story(file_path, title, date, ppl, chunks)
                    if doc_id == 0 :
                        # Already indexed (a race with another worker):
                        # the dedup made the ack safe.
                        logger.info(f"Story already indexed (race), skipping: {file_path}")
                    elif doc_id == -1 :
                        logger.error(f"Failed to index {file_path} (the transaction was rolled back); "
                                     "the message will be requeued.")
                        requeue = True
                        transient = True  # a Postgres failure, not a content failure
                    else :
                        docs_saved += 1
                        logger.info(f"Indexed doc {doc_id}: {len(chunks)} chunks, "
                                    f"offsets {chunks[0].offset}-{chunks[-1].offset} (file: {file_path})")
        # Settle the in-flight message now that its processing has fully
        # completed:
        #  * requeue - the processing did not complete durably (the
        #    dedup check failed, the chunking/embedding failed, the
        #    atomic insert rolled back): the story is reprocessed by
        #    this or another worker;
        #  * ack     - the story was indexed (or was already indexed /
        #    unprocessable - the url dedup is the idempotency key, so
        #    the ack is safe): in production mode it is consumed, in
        #    readonly mode the settle is a non-destructive requeue of
        #    the head message.
        # Settling only now means a worker crash before this point
        # leaves the message unacknowledged and the broker requeues it
        # for another worker instead of losing it.
        if requeue :
            if transient :
                # A transient service failure (the embedding servitor
                # or Postgres is down / flaky mid-run): the story is
                # retried, but the spin is bounded - after
                # MAX_TRANSIENT_RETRIES consecutive transient failures
                # the run aborts (the in-flight message is left
                # unacknowledged on purpose - the broker requeues it on
                # process exit).
                transient_streak += 1
                if transient_streak >= MAX_TRANSIENT_RETRIES :
                    logger.error(f"Aborting the run after {transient_streak} consecutive transient failures "
                                 f"(the last on {file_path}); the in-flight message will be requeued by the broker.")
                    aborted = True
                    break  # the message is intentionally left unacknowledged
                await rabbit_manager.requeue_pending()
            else :
                # A persistent content failure (the story cannot be
                # chunked / embedded - the error will not go away on
                # its own): count it per file, and once the file's
                # failed attempts reach MAX_FILE_RETRIES settle the
                # message as unprocessable (ack) instead of requeueing
                # - a poison file must not loop the queue forever.
                fail_counts[file_path] = fail_counts.get(file_path, 0) + 1
                if fail_counts[file_path] >= MAX_FILE_RETRIES :
                    logger.error(f"Give up on {file_path} after {fail_counts[file_path]} failed attempts; "
                                 "settling the message as unprocessable (the story is skipped).")
                    await rabbit_manager.ack_pending()
                else :
                    await rabbit_manager.requeue_pending()
        else :
            fail_counts.pop(file_path, None)
            transient_streak = 0
            await rabbit_manager.ack_pending()

    logger.info(f"Processed docs: {docs_number}")
    logger.info(f"Saved docs:     {docs_saved}")
    # The post-batch maintenance pass (the BM25 memtable merged into the
    # on-disk segment), taken under the RAG index maintenance lock for
    # the DURATION OF THE MERGE ONLY (milliseconds): the readers are
    # never blocked by the ingestion itself, only a retrieval landing
    # on the merge waits it out. Best-effort: a maintenance hiccup must
    # not fail the run (the data is already committed; the next run
    # repeats the pass). Skipped in readonly (DEBUG) mode (the server
    # is not modified).
    if not pg_client.readonly :
        if await pg_client.try_lock_index() :
            try :
                if not await pg_client.compact_bm25_index() :
                    logger.warning("The BM25 index maintenance pass failed (best-effort - the next run retries).")
            finally :
                await pg_client.unlock_index()
        else :
            logger.warning("The BM25 index maintenance lock is held (another run is merging the "
                           "index); the pass is skipped (the other run consolidates the index).")
    logger.info("======================================================")
    return 4 if aborted else 0


async def run_rag(docs_max : int = -1,
                  rabbit_host : str = "localhost", rabbit_port : int = 5672,
                  pg_host : str = PG_HOST, pg_port : int = PG_PORT,
                  pg_dbname : str = PG_DBNAME, pg_user : str = PG_USER) -> int :
    """
    System bootstrap (the analog of run_stealth_graph):
      1. Plug in the jina-v3 embedding servitor (vLLM) and run the
         startup health gate (the model is exposed with its
         max_model_len, a trivial story tokenizes and embeds to a
         finite EMBEDDING_DIM-dim vector, the special-token overhead
         of /v1/embeddings is calibrated - the per-chunk budget
         follows); any failure aborts the run (code -1).
      2. Connect Postgres (readonly in DEBUG) + ensure_schema()
         (idempotent DDL in production mode, a presence check only in
         readonly) - hard-fail with code 1 (connect) / 2 (schema).
      3. Connect RabbitMQ (reads scrapper_json_queue; the ingester is
         a pure consumer - there is no publish queue; readonly:
         non-destructive reads) - hard-fail with code 3.
      4. Launch the RAG ingester driver; the servitor client is closed
         on every exit path; the driver's error code becomes the
         return value of this function.
    """
    # Plug in the embedding servitor (the scrapper's servitor client,
    # analog): the read timeout is long for a busy vLLM (a request
    # may queue behind the in-flight batch - the startup health gate
    # is covered by it too).
    servitor = httpx.AsyncClient(base_url=EMBEDDING_BASE_URL,
                                  headers={ "Authorization" : f"Bearer {EMBEDDING_API_KEY}",
                                            "Content-Type" : "application/json" },
                                  timeout=httpx.Timeout(connect=10.0, read=EMBEDDING_READ_TIMEOUT_S,
                                                        write=300.0, pool=10.0))
    # The servitor client is closed on every exit path below (the
    # driver's error code becomes the return value of this function).
    try :
        # The startup health gate (analog of the scrapper's LLM /
        # servitor pings): a dead / wedged servitor would fail every
        # story, so the run cannot start (any failure -> code -1).
        try :
            # 1. The model is exposed with its max_model_len:
            response = await servitor.get("/v1/models")
            response.raise_for_status()
            model = next(( d for d in response.json().get("data", []) if d.get("id") == EMBEDDING_MODEL_NAME ), None)
            max_model_len = int(model.get("max_model_len", 0)) if model else 0
            assert model is not None and max_model_len > 0, "the embedding model is not exposed by the servitor"
            # 2. A trivial story tokenizes (the token count is the
            #    overhead-calibration anchor):
            ping_story = "The quick brown fox jumps over the lazy dog."
            response = await servitor.post("/tokenize",
                                           json={ "model" : EMBEDDING_MODEL_NAME, "prompt" : ping_story,
                                                  "add_special_tokens" : False })
            response.raise_for_status()
            token_count = response.json().get("count")
            assert isinstance(token_count, int) and token_count > 0, "no usable /tokenize count"
            # 3. The trivial story embeds to a finite EMBEDDING_DIM-dim
            #    vector; the special-token overhead of /v1/embeddings
            #    is CALIBRATED from usage.prompt_tokens - the /tokenize
            #    count of the same text (probe-confirmed: the endpoint
            #    scores 2 extra special tokens).
            response = await servitor.post("/v1/embeddings",
                                           json={ "model" : EMBEDDING_MODEL_NAME, "input" : ping_story })
            response.raise_for_status()
            body = response.json()
            data = body.get("data", [])
            assert len(data) == 1 and len(data[0].get("embedding", [])) == EMBEDDING_DIM, "unexpected embedding shape"
            assert all(math.isfinite(v) for v in data[0]["embedding"]), "non-finite embedding values"
            overhead = int(body.get("usage", {}).get("prompt_tokens", 0)) - token_count
            assert overhead >= 0, "the embeddings endpoint scored fewer tokens than /tokenize"
            # The per-chunk window: the servitor's context minus the
            # special-token overhead (the overhead is on the TOTAL
            # scored tokens, prompt + special) and minus the boundary
            # margin (a verbatim slice re-tokenized on its own can
            # drift a few boundary tokens - the max scored total is
            # max_model_len).
            token_budget = max_model_len - overhead - EMBED_BUDGET_MARGIN
            assert token_budget > 0, "the calibrated token budget is empty"
            logger.info(f"Embedding servitor OK: model={EMBEDDING_MODEL_NAME}, max_model_len={max_model_len}, "
                        f"special-token overhead={overhead}, per-chunk budget={token_budget} tokens")
        except Exception as e :
            logger.error(f"EMBEDDING SERVITOR Failed with error code {e}")
            return -1
        # Link Postgres (the RAG index) and ensure the schema (the
        # idempotent DDL in production mode, a presence check only in
        # readonly/DEBUG).
        async with PostgresDBClient(host=pg_host, port=pg_port, dbname=pg_dbname, user=pg_user) as pg_client :
            pg_err_code = await pg_client.connect()
            if pg_err_code != 0 :
                logger.error(f"Failed to connect Postgres at host {pg_client.host} port {pg_client.port}")
                return 1
            if (await pg_client.ensure_schema()) != 0 :
                logger.error("Failed to ensure the Postgres RAG schema.")
                return 2
            if pg_client.readonly :
                logger.info(f"DEBUG. Postgres readonly mode: the RAG tables are only checked, "
                            f"the writes are buffered (the server is not modified).")

            # Link RabbitMQ (the ingester is a pure consumer: it reads
            # the processed-stories queue and has no publish queue - the
            # ragger user has read-only access to it).
            async with RabbitMQClient(host=rabbit_host, port=rabbit_port,
                                      receive_queue_name=STORY_QUEUE_NAME,
                                      username="ragger", password="ragger",
                                      heartbeat=int(os.environ.get("RABBIT_HEARTBEAT", "300"))) as rabbit_manager :
                rabbit_err_code = await rabbit_manager.connect()
                if rabbit_err_code != 0 :
                    logger.error(f"Failed to connect RabbitMQ at host {rabbit_manager.host} port {rabbit_manager.port}")
                    return 3
                if rabbit_manager.readonly :
                    logger.info(f"DEBUG. RabbitMQ readonly mode: reads from {STORY_QUEUE_NAME} are non-destructive "
                                f"(messages are never removed); the run stops when the queue cycles back "
                                f"to an already-seen file.")

                # Launch the ingester
                return await run_rag_ingester(pg_client=pg_client, rabbit_manager=rabbit_manager,
                                              client=servitor, token_budget=token_budget,
                                              docs_max=docs_max)
    finally :
        await servitor.aclose()


if __name__ == "__main__":
    """
    Main entry point for command-line execution.

    Parses command-line arguments and launches the RAG ingestion
    pipeline with a user-specified docs limit and connection
    parameters.
    """
    import argparse
    parser = argparse.ArgumentParser(description="The RAG ingester (scrapper_json_queue -> Postgres).")

    parser.add_argument("--docs_max",     "-n",  type=int,  default=-1,           dest="docs_max",        help="Maximal number of docs to index.")
    parser.add_argument("--rabbit_host",  "-bh", type=str,  default="localhost",  dest="rabbit_host",     help="RabbitMQ host.")
    parser.add_argument("--rabbit_port",  "-bp", type=int,  default=5672,         dest="rabbit_port",     help="RabbitMQ port.")
    parser.add_argument("--pg_host",                  type=str,  default=PG_HOST,            dest="pg_host",       help="Postgres host.")
    parser.add_argument("--pg_port",                  type=int,  default=PG_PORT,            dest="pg_port",       help="Postgres port.")
    parser.add_argument("--pg_db",                    type=str,  default=PG_DBNAME,          dest="pg_dbname",     help="Postgres database name.")
    parser.add_argument("--pg_user",                  type=str,  default=PG_USER,            dest="pg_user",       help="Postgres user.")
    args = parser.parse_args()

    return_code = asyncio.run(
                   run_rag(docs_max=args.docs_max,
                           rabbit_host=args.rabbit_host, rabbit_port=args.rabbit_port,
                           pg_host=args.pg_host, pg_port=args.pg_port,
                           pg_dbname=args.pg_dbname, pg_user=args.pg_user))
    logger.info(f"Agent finished with {return_code} error code.")
