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

from typing import Any, Dict, List, Optional, Tuple
import os
import sys
import json
import math
import re
import time
import asyncio
import httpx

# Make the parent 'src' directory importable so that the modules moved there
# (pg_driver, llm_config, harness_framework) can be imported as top-level
# modules.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pg_driver import PostgresDBClient
# The base LLM (the answer generation) and the jina-v3 embedding servitor
# (the query embedding) endpoints - the shared LLM config module (the
# endpoints are configured in ONE place).
from llm_config import (BASE_URL, BASE_API_KEY, MODEL_NAME, LLM_BASE_CONTEXT_LEN,
                        LLM_READ_TIMEOUT_S, EMBEDDING_BASE_URL, EMBEDDING_API_KEY,
                        EMBEDDING_MODEL_NAME, EMBEDDING_DIM)
# The generic two-stages agentic harness (the RAG query stage's topology:
# the PROCESS candidate -> the DETERMINISTIC audit -> the LLM AUDIT
# terminal verdict; on a fail the retry edge replays the compiled
# feedback into the next attempt) and its LLM client (the base vLLM
# endpoint, the OpenAI-compatible API).
from harness_framework import build_graph
from langchain_openai import ChatOpenAI

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text


# =====================================================================
# RAG chat terminal
# ---------------------------------------------------------------------
# Purpose:
#   A two-panel terminal chatbot augmented by the RAG index (rag_db):
#   every question first goes through the RAG query stage (the generic
#   two-stages agentic harness of harness_framework.py, wired for the
#   query decision / reformulation: the PROCESS stage decides whether
#   the question needs the news corpus and, when it does, reformulates
#   the turn into a self-contained 200-400-token retrieval query - the
#   deterministic audit grades the contract / the degenerate gates, the
#   LLM audit is the terminal verdict); when the corpus is needed the
#   query is embedded (the jina-v3 embedding servitor - the corpus's
#   OWN embedding space) and used to retrieve the top chunks (the
#   pg_driver's search_chunks: the BM25 full-text rank + the vector
#   rank, fused with reciprocal rank fusion); the retrieved chunks +
#   the discussion history so far (the multi-turn context, appended
#   per turn, the oldest pairs rolled off under MAX_HISTORY_TOKENS)
#   form the input of a single chat completion on the base LLM (the
#   answer cites the doc ids, the thinking rides the same call).
#
# Layout:
#   a vertical split - the UPPER frame (1/3 of the height) is the info
#   subterminal (the per-turn retrieval summary - how many chunks were
#   retrieved, the paths that produced them, the timings - + the
#   corpus state); the LOWER frame (2/3) is the chat (the "user> ..."
#   and "LLM> ..." lines).
#
# NOTE: the Postgres client runs READONLY (the chat only reads the RAG
#       index - the driver pins the session READ ONLY), and the run
#       never touches the queues.
# =====================================================================

# --- The retrieval / generation budgets
TOP_K = 5                      # the chunks retrieved per question
MAX_ANSWER_TOKENS = 1024       # the answer budget (the base LLM's is far larger)
MAX_THINKING_TOKENS = 8192     # the thinking headroom of the answer call: the
                               # base LLM is a thinking model (Qwen3) that
                               # thinks as long as it decides to, up to this
                               # budget (the thinking tokens count against the
                               # output budget with the answer - the
                               # servitor's qwen3 reasoning parser returns
                               # them in a separate "reasoning" field, the
                               # content stays the clean answer)
EMBED_TIMEOUT_S = 60.0         # one small embedding call on the embedding servitor
# The input room of the answer call: the base LLM's context minus the
# output budget (the thinking headroom + the answer) minus a margin for
# the system prompt and the question (LLM_BASE_CONTEXT_LEN is the
# conservative 163840 - the live server probes larger). The input (the
# system prompt + the retrieved context + the discussion history + the
# question) must fit it, so the context room and the history budget
# below are carved out of the same room. A chunk is added to the
# context only if there is enough room for it (the greedy fill in
# build_context).
INPUT_TOKEN_BUDGET = LLM_BASE_CONTEXT_LEN - MAX_THINKING_TOKENS - MAX_ANSWER_TOKENS - 1024
# The discussion history budget (the multi-turn context of the answer
# calls - the user's questions and the LLM's answers, appended per
# turn): the history's estimated tokens (the char-based house estimate
# in _est_tokens - no client-side tokenizer) stay under this bound, and
# the OLDEST complete (user, answer) pairs drop off first when it is
# exceeded (the rolling trim - the newest turns are always kept).
MAX_HISTORY_TOKENS = 60000
# The guard's char-per-token estimate (no client-side tokenizer - the
# house rule; the server's max-model-len is the hard backstop).
CHARS_PER_TOKEN = 3.0
# --- The RAG query stage (the generic two-stages agentic harness of
# harness_framework.py, wired for the query decision / reformulation):
# the PROCESS stage decides whether the question needs the news corpus
# and, when it does, reformulates the turn into a self-contained
# retrieval query of RAG_QUERY_MIN_TOKENS..RAG_QUERY_MAX_TOKENS tokens
# (the pronouns / ellipses resolved from the discussion - the query
# feeds both the BM25 and the vector paths); the DETERMINISTIC AUDIT
# grades the output contract + the degenerate gates (binary, enforced -
# it floors the LLM verdict), the LLM AUDIT grades the decision / the
# query semantics and is the TERMINAL verdict; on a fail the harness's
# retry edge replays the compiled feedback into the next attempt (up to
# RAG_QUERY_ATTEMPTS_MAX attempts).
RAG_QUERY_MIN_TOKENS = 200     # the query's target length (the PROCESS
RAG_QUERY_MAX_TOKENS = 400     # prompt's instruction - the model is not
                               # trusted to keep it, the det audit
                               # enforces the degenerate gates below)
RAG_QUERY_LENGTH_FLOOR = 100   # det hard gate: under this the query did
RAG_QUERY_LENGTH_CEIL = 600    # not reformulate at all (over: a padding
                               # essay that dilutes both the BM25 and the
                               # vector query)
RAG_QUERY_ATTEMPTS_MAX = 2     # the harness's attempt budget (one correction
                               # cycle with the compiled feedback)
# The output ceiling of the harness's LLM client (shared by the PROCESS
# and the LLM AUDIT stages - it must cover the query's 400 tokens + the
# reason, and the auditor's three bounded observation blocks (each <=
# 2048*4 chars) + the verdict, on top of the thinking headroom).
RAG_QUERY_MAX_OUTPUT_TOKENS = 4096
# The ingester holds the RAG index maintenance lock ONLY around its
# post-batch BM25 merge (milliseconds - the readers are NEVER blocked
# by the data ingestion itself, their reads are snapshot-consistent):
# a retrieval that lands on the merge waits it out (a poll every
# INDEX_LOCK_POLL_S, capped at INDEX_LOCK_WAIT_S - then it proceeds
# anyway). The lock is NOT held during the LLM generation (the answer
# works on the already-retrieved context text).
INDEX_LOCK_POLL_S = 2.0
INDEX_LOCK_WAIT_S = 600.0

SYSTEM_PROMPT = (
    "You are a news analyst assistant with access to a corpus of recent news "
    "stories indexed in a RAG database. Answer the user's question using ONLY "
    "the provided context (the chunks retrieved from the index). If the "
    "context does not contain the answer, say so plainly. Cite the sources "
    "as [doc <id>] using the doc ids given in the context. The conversation "
    "history is provided - use it for the follow-ups (answer them from the "
    "history when no retrieved context is given) and stay consistent with "
    "what you already said. Be concise and factual."
)


def _est_tokens(text : str) -> int:
    """
    The token estimate of a text (the house rule - no client-side
    tokenizer): the char count over CHARS_PER_TOKEN, rounded up, at
    least 1 (an empty text still costs a message slot).
    """
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))


# =====================================================================
# The RAG query stage - the task wiring of the generic two-stages
# agentic harness (harness_framework.py): the PROCESS stage decides
# whether the current question needs the news corpus and, when it
# does, reformulates the turn into a self-contained retrieval query;
# the DETERMINISTIC AUDIT grades the output contract + the degenerate
# gates (binary, enforced - it floors the LLM verdict), the LLM AUDIT
# grades the decision / the query semantics (the terminal verdict); on
# a fail the harness's retry edge replays the compiled feedback into
# the next attempt. The topology / the state / the retry mechanics are
# the harness's - only the prompts and the deterministic audit are
# task-specific (the scrapper's news-extraction task follows the same
# wiring pattern).
# =====================================================================

RAG_QUERY_SYSTEM_PROMPT = (
    "You are the RAG query stage of a news chatbot. The corpus is an index of "
    "recent news stories. You are given the discussion so far between the "
    "user and the assistant, and the user's CURRENT question. Decide and "
    "reformulate:\n"
    "1. rag_needed - whether the question needs facts from the news corpus: "
    "true when the answer requires news facts the discussion so far does not "
    "already contain (a factual question about the news, a follow-up that "
    "needs more than the previous answers gave); false when the question is "
    "answerable from the discussion alone (a follow-up on what was just "
    "answered, a request to rephrase / summarise / expand on the assistant's "
    "own words, chit-chat, a meta question about the conversation). When in "
    "doubt, true - retrieval is cheap.\n"
    f"2. query - when rag_needed is true: a self-contained search query of "
    f"about {RAG_QUERY_MIN_TOKENS}-{RAG_QUERY_MAX_TOKENS} tokens, written so a "
    "search engine that has NEVER seen the discussion can use it: resolve "
    "every pronoun / ellipsis into the explicit entities and topics of the "
    "discussion, name the subjects, the timeframe and what exactly to look "
    "for. The query is fed to both a BM25 full-text index and a vector "
    "index of the news chunks - keep it a focused description of the sought "
    "information, no padding, no repetition.\n"
    "   when rag_needed is false: the empty string.\n"
    "Answer with ONE JSON object and nothing else: "
    '{"rag_needed": true|false, "reason": "<one or two sentences>", '
    '"query": "<the reformulated query, or the empty string>"}'
)

RAG_QUERY_AUDIT_SYSTEM_PROMPT = (
    "You are the auditor of the RAG query stage of a news chatbot. You are "
    "given the discussion so far, the user's CURRENT question, the candidate "
    "decision (rag_needed / reason / query) and the deterministic audit's "
    "verdict as objective evidence. Grade the candidate on four semantic "
    "metrics (one line per metric in the matching block):\n"
    "- decision_correct: rag_needed matches the discussion (a question "
    "answerable from the discussion alone must be false; a factual news "
    "question must be true; when in doubt, true is the safer grade - a "
    "missed retrieval degrades the answer, a spurious one only costs "
    "retrieval);\n"
    "- query_self_contained: a search engine that has never seen the "
    "discussion can act on the query alone (no unresolved pronouns / "
    "ellipses / 'the above' / references to the conversation);\n"
    "- query_fidelity: the query is faithful to the CURRENT question (no "
    "invented topics, no drift to a different subject);\n"
    "- query_focus: the query is a focused description of the sought news "
    "information (entities, topic, timeframe - no padding).\n"
    "The deterministic audit already graded the output contract and the "
    "degenerate gates - do not re-grade those, but treat its issues as "
    "evidence. When rag_needed is false there is no query: grade "
    "decision_correct and report the query metrics as pass.\n"
    "Report the observations in fail_stuff (what is wrong), none_stuff "
    "(what you could not grade), pass_stuff (what is right) - one line per "
    "metric in each - and end with the one-word verdict.\n"
    "Answer with ONE JSON object and nothing else: "
    '{"verdict": "pass"|"fail"|"none", "fail_stuff": "...", "none_stuff": "...", '
    '"pass_stuff": "..."}'
)


def _rag_query_document(history : List[Dict[str, str]], question : str) -> str:
    """
    The harness's `document` input of the RAG query stage: the
    discussion so far rendered as a transcript (the "user> ..." /
    "LLM> ..." lines - the first turn has none) + the current question
    (the stage's decision / reformulation input).
    """
    transcript = "\n".join(f'{"user> " if m["role"] == "user" else "LLM> "}{m["content"]}'
                           for m in history)
    transcript = transcript or "(none - this is the first message of the chat)"
    return transcript + "\n\nCURRENT USER QUESTION:\n" + question


def build_rag_query_process_prompt(document : str, document_name : str,
                                   feedback : str) -> Tuple[str, str]:
    """
    The RAG query stage's PROCESS prompt (the harness's candidate-output
    stage): the discussion + the current question (the `document`) + the
    compiled feedback of the previous attempt (empty on the first).
    """
    user = document
    if feedback :
        user += ("\n\n---\nPREVIOUS ATTEMPT WAS REJECTED - fix the problems "
                 "below and answer again (answer with the same JSON contract):\n"
                 + feedback)
    return (RAG_QUERY_SYSTEM_PROMPT, user)


def build_rag_query_audit_prompt(document : str, document_name : str,
                                 process : Dict[str, Any],
                                 det_audit : Dict[str, Any]) -> Tuple[str, str]:
    """
    The RAG query stage's LLM AUDIT prompt (the harness's terminal-verdict
    stage): the same discussion / question as the PROCESS stage + the
    candidate + the deterministic verdict as objective evidence.
    """
    issues = "\n".join(f"- {i}" for i in det_audit.get("issues") or [])
    user = (document
            + "\n\n---\nCANDIDATE DECISION:\n" + json.dumps(process, ensure_ascii=False)
            + "\n\n---\nDETERMINISTIC AUDIT (objective evidence):\n"
              f"overall: {det_audit.get('overall')}"
            + (("\n" + issues) if issues else ""))
    return (RAG_QUERY_AUDIT_SYSTEM_PROMPT, user)


# The curated marker patterns of the deterministic audit's
# self_contained_markers metric (kept narrow on purpose: a 200-400-token
# reformulated query that still carries a chat-reference is unusable by
# a search engine that has never seen the discussion).
_RAG_QUERY_REFERENCE_PATTERNS = (
    re.compile(r"^(that|this|it|they|these|those|he|she)\b", re.IGNORECASE),
    re.compile(r"\b(as|per)?\s*mentioned (above|before|previously|earlier)", re.IGNORECASE),
    re.compile(r"\b(the above|the aforementioned|the previous (question|answer|turn|message))\b", re.IGNORECASE),
    re.compile(r"\byour (previous|last) (answer|response|message)\b", re.IGNORECASE),
)
# The instruction-leak markers of the contract_leak metric (the model
# echoed its own JSON contract / the task's wording into the query).
_RAG_QUERY_LEAK_PATTERNS = (
    re.compile(r"\brag_?needed\b", re.IGNORECASE),
    re.compile(r"\bjson object\b", re.IGNORECASE),
    re.compile(r"\btoken (band|count|limit|budget)\b", re.IGNORECASE),
    re.compile(r"\bself-contained (search )?query\b", re.IGNORECASE),
)
# The stopwords of the repetition metric (the distinct-word ratio is
# counted over the content words only).
_RAG_QUERY_STOPWORDS = frozenset("""
    a an and are as at be but by for from has have had if in is it its of on
    or that the their then there they this to was we were what when where
    which who will with would you your about against between during before
    after above below into over under out off up down not no nor so than
    too very can could should may might must do does did doing been being
    while within without across per via
""".split())


def _rag_query_deterministic_audit(document : str, document_name : str,
                                   process : Dict[str, Any]) -> Dict[str, Any]:
    """
    The RAG query task's deterministic audit (never an LLM): the PROCESS
    stage's output contract + the query's degenerate gates, graded on
    the BINARY scale (pass/fail - the harness enforces it; a det fail
    floors the LLM verdict, so every check is objective and kept low on
    false positives):
      contract       - rag_needed is a bool, reason a non-empty string,
                       query a string (the harness's JSON recovery already
                       guarantees the top level is a dict);
      decision_shape - rag_needed=true requires a non-empty query,
                       rag_needed=false requires the empty one;
      query_length   - the query's estimated tokens (the char-based house
                       estimate - no client-side tokenizer) sit inside the
                       degenerate gates (RAG_QUERY_LENGTH_FLOOR..
                       RAG_QUERY_LENGTH_CEIL - the 200-400 target is the
                       PROCESS prompt's instruction, the measured estimate
                       rides into the issues as evidence);
      self_contained_markers - no chat-reference marker (a leading
                       unresolved demonstrative, "as mentioned above",
                       "the previous answer", ...);
      repetition     - the distinct content-word ratio is at/above 0.4
                       (a lower ratio is the padding degeneracy - the
                       model repeating words to hit the token target);
      contract_leak  - no echo of the stage's own contract / instructions
                       (rag_needed / json object / token band, ...).
    overall: any fail -> fail; else pass.
    """
    metrics : Dict[str, str] = {}
    issues : List[str] = []

    rag_needed = process.get("rag_needed")
    reason = process.get("reason")
    query = process.get("query")
    if not isinstance(rag_needed, bool) :
        metrics["contract"] = "fail"
        issues.append(f"rag_needed must be a bool (got {rag_needed!r})")
    elif not isinstance(reason, str) or not reason.strip() :
        metrics["contract"] = "fail"
        issues.append("reason must be a non-empty string")
    elif not isinstance(query, str) :
        metrics["contract"] = "fail"
        issues.append(f"query must be a string (got {query!r})")
    else :
        metrics["contract"] = "pass"

    query_text = query.strip() if isinstance(query, str) else ""
    if isinstance(rag_needed, bool) :
        if rag_needed and not query_text :
            metrics["decision_shape"] = "fail"
            issues.append("rag_needed is true but the query is empty")
        elif not rag_needed and query_text :
            metrics["decision_shape"] = "fail"
            issues.append(f"rag_needed is false but the query is not empty "
                          f"({len(query_text)} chars)")
        else :
            metrics["decision_shape"] = "pass"

    if rag_needed is True and query_text :
        tokens = _est_tokens(query_text)
        if RAG_QUERY_LENGTH_FLOOR <= tokens <= RAG_QUERY_LENGTH_CEIL :
            metrics["query_length"] = "pass"
        else :
            metrics["query_length"] = "fail"
            issues.append(f"the query's ~{tokens} tokens sit outside the "
                          f"{RAG_QUERY_LENGTH_FLOOR}-{RAG_QUERY_LENGTH_CEIL} degenerate gates "
                          f"(the target is {RAG_QUERY_MIN_TOKENS}-{RAG_QUERY_MAX_TOKENS})")
        if any(p.search(query_text) for p in _RAG_QUERY_REFERENCE_PATTERNS) :
            metrics["self_contained_markers"] = "fail"
            issues.append("the query carries a chat-reference marker (a leading "
                          "unresolved demonstrative or an 'as mentioned / the previous "
                          "answer' phrase) - it is unusable without the discussion")
        else :
            metrics["self_contained_markers"] = "pass"
        words = [ w for w in re.findall(r"[a-z']+", query_text.lower())
                  if w not in _RAG_QUERY_STOPWORDS ]
        distinct = len(set(words)) / len(words) if words else 0.0
        if distinct < 0.4 :
            metrics["repetition"] = "fail"
            issues.append(f"the query's distinct content-word ratio is {distinct:.2f} "
                          f"(below 0.4 - the padding degeneracy)")
        else :
            metrics["repetition"] = "pass"
        if any(p.search(query_text) for p in _RAG_QUERY_LEAK_PATTERNS) :
            metrics["contract_leak"] = "fail"
            issues.append("the query echoes the stage's own contract / instructions")
        else :
            metrics["contract_leak"] = "pass"

    overall = "fail" if any(v == "fail" for v in metrics.values()) else "pass"
    return { "overall" : overall, "metrics" : metrics, "issues" : issues,
             "feedback" : " - ".join(issues[:3]) }


def _truncate(text : str, limit : int) -> str:
    """A line of the info panel: the text capped at `limit` chars."""
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _rank_pair(rank : Optional[int], score : Optional[float]) -> str:
    """A path's rank/score pair for the info panel ('-' when the path
    had no match for the chunk)."""
    if rank is None :
        return "-"
    return f"#{rank} {score:.4f}" if score is not None else f"#{rank} ?"


def _chunk_log_line(r : Dict[str, Any], width : int) -> str:
    """
    A per-chunk line of the info panel: the doc title (from the RAG
    DB, truncated to the frame width) + the two paths' rank/score
    pairs (bm25 / vec, '-' when the path had no match).
    """
    title_w = max(20, width - 60)
    return (f'   [doc {r["doc_id"]}] "{_truncate(r["title"], title_w)}"  '
            f'bm25: {_rank_pair(r["bm25_rank"], r["bm25_score"])}  '
            f'vec: {_rank_pair(r["vec_rank"], r["vec_score"])}')


def _top_rank_suffix(rows : List[Dict[str, Any]]) -> str:
    """
    The '(top: ...)' suffix of the turn's header line: the best
    (highest = smallest) rank among the top chunk's paths ('' when
    nothing was retrieved or no path matched the top chunk).
    """
    if not rows :
        return ""
    cands = []
    if rows[0]["bm25_rank"] is not None :
        cands.append(("bm25", rows[0]["bm25_rank"]))
    if rows[0]["vec_rank"] is not None :
        cands.append(("vec", rows[0]["vec_rank"]))
    if not cands :
        return ""
    path, rank = min(cands, key=lambda c : c[1])
    return f" (top: {path} #{rank})"


def _chunk_block(r : Dict[str, Any]) -> str:
    """The rendered context block of one retrieved row ([doc <id>] is the citation key)."""
    date = r["date"]
    date = date.isoformat() if hasattr(date, "isoformat") else str(date)
    return f"--- [doc {r['doc_id']} | {r['title']} | {date}] ---\n{r['raw_text']}"


def fit_context(rows : List[Dict[str, Any]], char_budget : int) -> Tuple[List[str], int]:
    """
    The greedy context fill: the chunks are added (in relevance order)
    one at a time, and a new chunk is added only if there is enough
    room in the context for it (the rendered block + the "\n\n"
    separator, counted in chars against the char_budget). The first
    chunk that does not fit ends the fill (the later chunks are less
    relevant).

    Returns:
        (the rendered blocks of the included chunks, the number of
        chunks included).
    """
    blocks : List[str] = []
    used = 0
    for r in rows :
        extra = len(_chunk_block(r)) + (2 if blocks else 0)  # the separator, except for the first
        if used + extra > char_budget :
            break
        blocks.append(_chunk_block(r))
        used += extra
    return blocks, len(blocks)


def build_context(rows : List[Dict[str, Any]], room_tokens : int) -> Tuple[str, int]:
    """
    The context block of the LLM prompt: the chunks of the retrieved
    rows (in relevance order) that fit the ROOM left for the context -
    the tokens of the answer call's input not taken by the system
    prompt, the discussion history and the question ([doc <id>] is the
    citation key the model must use).

    Returns:
        (the context text, the number of chunks that fit the room).
    """
    if not rows :
        return ("(no chunks were retrieved - say the context does not cover the question)", 0)
    blocks, n_included = fit_context(rows, max(0, int(room_tokens * CHARS_PER_TOKEN)))
    if not n_included :
        return ("(no retrieved chunk fit the context room - say the context does not cover the question)", 0)
    context = "\n\n".join(blocks)
    if n_included < len(rows) :
        context += f"\n\n(only {n_included} of {len(rows)} retrieved chunks fit the context room)"
    return context, n_included


async def embed_query(emb_client : httpx.AsyncClient, question : str) -> Optional[List[float]]:
    """
    The question's embedding (the corpus's own embedding space - the
    same jina-v3 model that embedded the chunks). None when the
    servitor fails (the retrieval degrades to the BM25 path only).
    """
    try:
        response = await emb_client.post("/v1/embeddings",
                                         json={ "model" : EMBEDDING_MODEL_NAME, "input" : [question] })
        response.raise_for_status()
        data = response.json().get("data", [])
        if len(data) != 1 or len(data[0].get("embedding", [])) != EMBEDDING_DIM :
            return None
        return [ float(x) for x in data[0]["embedding"] ]
    except Exception as e :
        logger.warning(f"Embedding servitor failed (the BM25 path is used instead): {e}")
        return None


async def ask_llm(llm_client : httpx.AsyncClient, messages : List[Dict[str, str]]) -> str:
    """
    One chat completion on the base LLM (the answer generation): the
    thinking rides the same call (the thinking headroom counts against
    the output budget with the answer - the servitor's qwen3 reasoning
    parser returns the thinking in a separate "reasoning" field, the
    content stays the clean answer). The answer text, or an
    "(LLM error: ...)" line when the servitor fails (the chat loop
    keeps going).
    """
    try:
        response = await llm_client.post("/v1/chat/completions",
                                         json={ "model" : MODEL_NAME, "messages" : messages,
                                                "temperature" : 0.3,
                                                "max_tokens" : MAX_THINKING_TOKENS + MAX_ANSWER_TOKENS,
                                                "chat_template_kwargs" : { "enable_thinking" : True } })
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return (content or "").strip() or "(empty answer)"
    except Exception as e :
        return f"(LLM error: {e})"


def build_layout(info_lines : List[str], chat_lines : List[Tuple[str, str]],
                 chat_window : int) -> Layout:
    """
    The two-panel layout (the vertical split): the upper frame (1/3 of
    the height) is the info subterminal, the lower frame (2/3) is the
    chat (the "user> ..." / "LLM> ..." lines, word-wrapped - only the
    last `chat_window` lines are rendered, the older ones roll off).
    """
    info_text = Text("\n".join(info_lines))
    chat_text = Text()
    visible = chat_lines[-chat_window :]
    if len(chat_lines) > len(visible) :
        chat_text.append(f"(... {len(chat_lines) - len(visible)} older line(s) omitted ...)\n", style="dim")
    for (role, body) in visible :
        chat_text.append("user> " if role == "user" else "LLM> ",
                         style="bold cyan" if role == "user" else "bold green")
        chat_text.append(body)
        chat_text.append("\n")
    root = Layout()
    root.split_column(
        Layout(Panel(info_text, title="RAG INFO", border_style="cyan"), ratio=1),
        Layout(Panel(chat_text, title="CHAT", border_style="green"), ratio=2),
    )
    return root


async def run_chat() -> int:
    """
    The chat session: the two-panel loop (per question: the RAG query
    stage (the two-stages harness - the decision whether the corpus is
    needed + the query reformulation) -> the query embedding -> the
    top-chunk retrieval -> the base LLM answer on the retrieved context
    + the discussion history so far).

    Returns:
        0 on a clean exit (q / quit / exit / EOF), 1-3 on a failed
        connection / schema / corpus gate.
    """
    console = Console()
    # The panel windows (the rendered lines fit the frame's inner
    # height - a 1/3 : 2/3 split of the terminal height, the panel
    # borders and a safety margin deducted).
    info_window = max(2, console.height // 3 - 5)
    chat_window = max(4, console.height * 2 // 3 - 5)

    async with PostgresDBClient(readonly=True) as pg_client :
        if (await pg_client.connect()) != 0 :
            logger.error("Failed to connect Postgres.")
            return 1
        if (await pg_client.ensure_schema()) != 0 :
            logger.error("Failed to ensure the Postgres RAG schema.")
            return 2
        docs_count = await pg_client.get_doc_count()
        chunks_count = await pg_client.get_chunk_count()
        if docs_count < 0 or chunks_count < 0 :
            logger.error("The RAG corpus is unreadable.")
            return 3

        llm_client = httpx.AsyncClient(base_url=BASE_URL,
                                       headers={ "Authorization" : f"Bearer {BASE_API_KEY}" },
                                       timeout=LLM_READ_TIMEOUT_S)
        emb_client = httpx.AsyncClient(base_url=EMBEDDING_BASE_URL,
                                       headers={ "Authorization" : f"Bearer {EMBEDDING_API_KEY}" },
                                       timeout=EMBED_TIMEOUT_S)
        # The RAG query stage (the generic two-stages agentic harness
        # wired for the query decision / reformulation): one compiled
        # graph for the whole session, the attempt budget + the LLM
        # client ride the per-invocation context.
        rag_query_llm = ChatOpenAI( base_url=f"{BASE_URL}/v1",
                                    api_key=BASE_API_KEY,
                                    model_name=MODEL_NAME,
                                    temperature=0.1,
                                    max_tokens=MAX_THINKING_TOKENS + RAG_QUERY_MAX_OUTPUT_TOKENS,
                                    timeout=httpx.Timeout(connect=10.0, read=LLM_READ_TIMEOUT_S,
                                                          write=300.0, pool=10.0),
                                    max_retries=0,
                                    # chat_template_kwargs (vLLM) go through extra_body: a
                                    # per-call RunnableConfig is NOT forwarded to the
                                    # OpenAI-compatible API by langchain_openai, so the
                                    # thinking flag would otherwise never reach the server.
                                    extra_body={ "chat_template_kwargs" :
                                                 { "enable_thinking" : True,
                                                   "reasoning_effort" : "high" } }, )
        rag_query_graph = build_graph(build_process_prompt=build_rag_query_process_prompt,
                                      build_audit_prompt=build_rag_query_audit_prompt,
                                      det_audit=_rag_query_deterministic_audit)
        # The discussion history (the multi-turn context of the answer
        # calls - the user's questions and the LLM's answers, appended
        # per turn; the oldest complete pairs drop off under
        # MAX_HISTORY_TOKENS - the rolling trim).
        history : List[Dict[str, str]] = []
        chat_lines : List[Tuple[str, str]] = [
            ("LLM", f"RAG chat ready - the corpus holds {docs_count} docs / {chunks_count} chunks. "
                    f"Ask a question about the news (q to quit).") ]
        info_lines : List[str] = [
            f"Corpus: {docs_count} docs / {chunks_count} chunks (rag_db, readonly)",
            "Turns: 0 | 0 chunks retrieved | ~0 tok history",
            "" ]
        total_chunks = 0
        turn_no = 0
        live = Live(build_layout(info_lines, chat_lines, chat_window),
                    console=console, refresh_per_second=4)
        live.start()
        try :
            while True :
                # The question (the display stops while the line is
                # typed, so the prompt and the typed text sit right
                # below the last frame; the display restarts before the
                # turn runs).
                live.stop()
                try :
                    question = input("user> ").strip()
                except (EOFError, KeyboardInterrupt) :
                    break
                live.start()
                if not question :
                    continue
                if question.lower() in ("q", "quit", "exit") :
                    break
                turn_no += 1
                chat_lines.append(("user", question))
                info_lines.append(f'#{turn_no} "{_truncate(question, 44)}" -> querying...')
                live.update(build_layout(info_lines, chat_lines, chat_window))
                try :
                    # The rolling trim of the discussion history (the
                    # estimated tokens stay under MAX_HISTORY_TOKENS -
                    # the oldest complete (user, answer) pairs drop off
                    # first, the newest turns are always kept).
                    while len(history) >= 2 :
                        if sum(_est_tokens(m["content"]) for m in history) <= MAX_HISTORY_TOKENS :
                            break
                        del history[:2]
                    history_tokens = sum(_est_tokens(m["content"]) for m in history)
                    # The RAG query stage (the two-stages harness): the
                    # PROCESS stage decides whether the question needs
                    # the news corpus and, when it does, reformulates
                    # the turn into a self-contained retrieval query;
                    # the audits (the deterministic contract / degenerate
                    # gates + the semantic LLM audit - the terminal
                    # verdict) gate the attempt, the harness's retry edge
                    # replays the compiled feedback on a fail.
                    t_q0 = time.monotonic()
                    state = await rag_query_graph.ainvoke(
                        { "document" : _rag_query_document(history, question),
                          "document_name" : f"turn {turn_no}" },
                        context={ "llm_base" : rag_query_llm,
                                  "retries_max" : RAG_QUERY_ATTEMPTS_MAX })
                    t_q1 = time.monotonic()
                    verdict = state["llm_audit_history"][-1]["verdict"]
                    candidate = state["process_history"][-1]
                    attempts = len(state["process_history"])
                    if verdict == "pass" and not candidate.get("error") :
                        rag_needed = bool(candidate["rag_needed"])
                        rag_query = str(candidate.get("query") or "").strip()
                        rag_note = f"query stage ok ({attempts} attempt(s))"
                    elif isinstance(candidate.get("rag_needed"), bool) \
                            and str(candidate.get("query") or "").strip() :
                        # The attempt budget was exhausted without an
                        # accepted decision: degrade, never block the
                        # chat - the last parseable candidate (not
                        # accepted by the terminal verdict) is still a
                        # usable retrieval query.
                        rag_needed = bool(candidate["rag_needed"])
                        rag_query = str(candidate["query"]).strip()
                        rag_note = f"query stage rejected - last candidate ({attempts} attempt(s))"
                        logger.warning(f"Turn {turn_no}: the RAG query stage ended on '{verdict}' "
                                       "after the attempt budget - the last candidate is used "
                                       "unaudited.")
                    else :
                        # No usable candidate at all: the pre-harness
                        # behaviour (retrieve on the raw question).
                        rag_needed = True
                        rag_query = question
                        rag_note = f"query stage failed - raw question ({attempts} attempt(s))"
                        logger.warning(f"Turn {turn_no}: the RAG query stage produced no usable "
                                       "candidate after the attempt budget - the retrieval falls "
                                       "back to the raw question.")
                    info_lines[-1] = f'#{turn_no} "{_truncate(question, 44)}" -> {rag_note}'
                    live.update(build_layout(info_lines, chat_lines, chat_window))
                    # The retrieval (skipped when the corpus is not
                    # needed - the answer works on the discussion
                    # history only). The ingester holds the RAG index
                    # maintenance lock only around the post-batch BM25
                    # merge (it is not held during the data ingestion):
                    # a retrieval landing on the merge waits it out (the
                    # LLM answer below works on the already-retrieved
                    # context - no lock during the generation).
                    indexer_running = False
                    embedding : Optional[List[float]] = None
                    rows : List[Dict[str, Any]] = []
                    if rag_needed and rag_query :
                        waited = 0.0
                        while await pg_client.is_index_locked() and waited < INDEX_LOCK_WAIT_S :
                            info_lines[-1] = (f'#{turn_no} "{_truncate(question, 44)}" -> '
                                              f'index merge in progress, waiting ({waited:.0f}s)...')
                            live.update(build_layout(info_lines, chat_lines, chat_window))
                            await asyncio.sleep(INDEX_LOCK_POLL_S)
                            waited += INDEX_LOCK_POLL_S
                        if await pg_client.is_index_locked() :
                            # The wait was capped: proceed anyway (the
                            # reads are snapshot-consistent - the corpus
                            # may be mid-update, the info line says so).
                            indexer_running = True
                            logger.warning(f"The index merge still holds the index lock after "
                                           f"{INDEX_LOCK_WAIT_S:.0f}s; the retrieval proceeds on the current snapshot.")
                        t_r0 = time.monotonic()
                        embedding = await embed_query(emb_client, rag_query)
                        rows = await pg_client.search_chunks(rag_query, embedding, top_k=TOP_K)
                        t_r1 = time.monotonic()
                    else :
                        t_r0, t_r1 = t_q1, t_q1
                    docs_hit = len({ r["doc_id"] for r in rows })
                    if rows and any(r["bm25_rank"] is not None for r in rows) :
                        paths = "bm25+vec"
                    elif rows and embedding is None :
                        paths = "bm25"
                    elif rows :
                        paths = "vec"
                    else :
                        paths = "none"
                    # The context room of the answer call's input (the
                    # input budget minus the system prompt, the
                    # discussion history and the question - the margin
                    # inside INPUT_TOKEN_BUDGET covers the prompt's glue
                    # lines).
                    if rag_needed :
                        context, n_in_ctx = build_context(
                            rows, INPUT_TOKEN_BUDGET - _est_tokens(SYSTEM_PROMPT)
                            - history_tokens - _est_tokens(question))
                    else :
                        context, n_in_ctx = None, 0
                    turn_hdr = f'#{turn_no} "{_truncate(question, 44)}" -> {rag_note}'
                    if rag_needed :
                        turn_hdr += f" -> {len(rows)} chunks / {docs_hit} docs [{paths}]"
                        # Fewer chunks fit the context room than were
                        # retrieved - the info panel shows it.
                        if n_in_ctx < len(rows) :
                            turn_hdr += f" ({n_in_ctx} in context)"
                        turn_hdr += _top_rank_suffix(rows)
                    else :
                        turn_hdr += " -> no RAG (the discussion answers it)"
                    if indexer_running :
                        turn_hdr += " (merge still running)"
                    info_lines[-1] = turn_hdr
                    # The per-chunk log lines (the doc title + the two
                    # paths' rank/score pairs), one per pulled chunk -
                    # as many as fit the frame (the header states the
                    # total, the older lines roll off).
                    hdr_idx = len(info_lines) - 1
                    for r in rows[: max(1, info_window - 4)] :
                        info_lines.append(_chunk_log_line(r, console.width))
                    live.update(build_layout(info_lines, chat_lines, chat_window))
                    # The answer call: the system prompt (+ the
                    # retrieved context when the corpus was needed) +
                    # the discussion history so far + the current
                    # question.
                    system = SYSTEM_PROMPT
                    if rag_needed :
                        system += "\n\nContext (retrieved from the RAG index):\n\n" + context
                    messages = [ { "role" : "system", "content" : system },
                                  *history,
                                  { "role" : "user", "content" : question } ]
                    info_lines[hdr_idx] += " | generating..."
                    live.update(build_layout(info_lines, chat_lines, chat_window))
                    t_a0 = time.monotonic()
                    answer = await ask_llm(llm_client, messages)
                    t_a1 = time.monotonic()
                    # The answer joins the discussion history (the
                    # multi-turn context of the next turns).
                    history.append({ "role" : "user", "content" : question })
                    history.append({ "role" : "assistant", "content" : answer })
                    total_chunks += len(rows)
                    info_lines[hdr_idx] += (f"  q {t_q1 - t_q0:.1f}s | ret {t_r1 - t_r0:.1f}s "
                                            f"| llm {t_a1 - t_a0:.1f}s")
                    history_tokens = sum(_est_tokens(m["content"]) for m in history)
                    info_lines[1] = (f"Turns: {turn_no} | {total_chunks} chunks retrieved "
                                     f"| ~{history_tokens} tok history")
                    # The info window rolls off the oldest turn lines.
                    if len(info_lines) > 3 + info_window :
                        del info_lines[3 : -info_window]
                    chat_lines.append(("LLM", answer))
                    live.update(build_layout(info_lines, chat_lines, chat_window))
                except KeyboardInterrupt :
                    chat_lines.append(("LLM", "(interrupted - the turn was skipped)"))
                    break
                except Exception as e :
                    chat_lines.append(("LLM", f"(turn failed: {e})"))
                    logger.error(f"Turn failed: {e}")
            chat_lines.append(("LLM", "bye."))
            if not live.is_started :
                live.start()
            live.update(build_layout(info_lines, chat_lines, chat_window))
        finally :
            if live.is_started :
                live.stop()
            await llm_client.aclose()
            await emb_client.aclose()
        return 0

# Examples:
# "Which company is leading the oil exploration bids off Newfoundland?"
# "What did the Canadian budget do with the higher oil revenue?"
# "What's the news between Amazon and Uber Eats?"
# "Why are more first-time homebuyers paying in cash?"
# "Who won the 2026 World Cup?" (negative test)

if __name__ == "__main__":
    """
    Main entry point for command-line execution.

    Launches the RAG chat terminal (the two-panel loop: the retrieval
    summary in the upper frame, the user> / LLM> chat in the lower).
    """
    return_code = asyncio.run(run_chat())
    logger.info(f"Agent finished with {return_code} error code.")
