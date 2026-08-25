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
# file_handler = logging.FileHandler('web_analysis_agent.log')
# file_handler.setLevel(logging.DEBUG)
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)

from typing import TypedDict, Optional, List, Dict, Tuple, Any, Protocol, Set
import os
import sys
import asyncio
from contextlib import asynccontextmanager
import re
import json
import datetime
import uuid
import gzip
import httpx

# Make the parent 'src' directory importable so that the module moved there
# (rabbit_driver) can be imported as a top-level module.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, RemoveMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import ClientSession
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition


# =====================================================================
# Batch HTML -> story pipeline
# ---------------------------------------------------------------------
# Purpose:
#   Consumes references to already-downloaded raw HTML files (one story
#   per file) from the crawler's RabbitMQ queue and rewrites each page
#   into a clean, dense "gold standard" story via an LLM call. There is
#   no live browsing in this module - web navigation belongs to the
#   separate crawler; this is the offline extraction / cleaning stage.
#
# Data flow:
#   crawler_json_queue (RabbitMQ, published by the crawler - every
#        |              message names a stored .html.gz file + source url)
#        v
#   LangGraph rewrite node (two LLM stages per file, with retries)
#        |
#        +--> docs_folder_name/      (saved story documents)
#        +--> scrapper_json_queue    (story JSON published to the queue
#                                     for processed stories)
#
# NOTE: in DEBUG (readonly rabbit) the client still connects to the real
#       broker, but the input queue is read one message at a time and each
#       message is put back (never drained), so a run can be repeated; the
#       output publish to scrapper_json_queue is skipped - progress is
#       logged instead. Since the peeked head message is requeued to the
#       head, a DEBUG run processes the head message once and stops as soon
#       as the queue cycles back to an already-seen file.
#
# =====================================================================

BASE_URL="http://ifo4:8000"
BASE_API_KEY="alex_llm_qwen"
#MODEL_NAME="poolside/Laguna-XS-2.1-NVFP4"
#MODEL_NAME="QuantTrio/Qwen3.6-27B-AWQ"
MODEL_NAME="unsloth/Qwen3.8-27B-NVFP4"
LLM_BASE_CONTEXT_LEN = 163840 # 131072 # 196608 
LLM_MAX_OUTPUT_TOKENS = 32768
#python3  -m vllm.entrypoints.openai.api_server --model unsloth/Qwen3.8-27B-NVFP4 --host 0.0.0.0 --port 8000 --api-key alex_llm_qwen \
#        --max-model-len 163840 --kv-cache-dtype fp8 --gpu-memory-utilization 0.96 --max-num-seqs 1 --max-num-batched-tokens 8192    \
#        --enable-auto-tool-choice --tool-call-parser qwen3_xml --enable-prefix-caching --reasoning-parser qwen3 --trust-remote-code \
#        --enable-chunked-prefill --limit-mm-per-prompt '{"image": 0, "video": 0, "audio": 0}'


# --- DEBUG FLAG: Use a readonly implementation for testing/debugging ---
# Set the DEBUG environment variable to 'true', '1', or 'yes' (case-insensitive)
# to use the readonly implementation of the RabbitMQ client: it still
# connects to the real broker, but reads are non-destructive (messages are
# never removed from the queue) and writes are buffered instead of published.
DEBUG = os.environ.get('DEBUG', '').lower() in ('true', '1', 'yes')

if DEBUG:
    logger.debug("INFO. Running in DEBUG mode.")

# Import the unified client with readonly support
# This module was moved to the src/ directory root; it is importable
# thanks to the sys.path insertion above.
from rabbit_driver import RabbitMQClient

# The RabbitMQ queues of the pipeline (see the data flow above):
CRAWLER_JSON_QUEUE = "crawler_json_queue"     # input - raw-HTML file references published by the crawler
PROCESSED_STORY_QUEUE = "scrapper_json_queue" # output - the queue for processed stories


#   --------   A G E N T I C   P A R T   -------- 


class AgenticState(TypedDict):
    """
    TypedDict defining the state structure for the batch HTML->story pipeline.
    One run of the pipeline processes the raw-HTML files referenced by the
    crawler's queue messages; LangGraph threads this state through the nodes
    while the driver loop feeds one file per cycle and the agent updates it
    as files are processed.
    Attributes:
        file_content (List[Optional[str]]): Single-slot holder for the raw HTML uploaded
            from the current input file; consumed by the rewrite agent. The slot is set
            to None when the rewrite of the file fails.
        rewrite_history (List[Optional[str]]): Single-slot holder for the rewrite produced
            for the current file: an empty string while the rewrite cycle is in progress,
            the story text once the rewrite succeeded, None once the retries are exhausted
            or there was nothing to rewrite.
        pages_number (List[int]): Number of files uploaded/processed so far during the
            current run.
        pages_saved (List[int]): Counter of successfully saved stories; element [0] is
            checked against the pages limit to stop the driver loop.
        retries_max (int): Maximum rewrite attempts allowed for a failed page/LLM access.
        retry_counter (List[int]): Per-file retry counter, incremented on failed access
            until retries_max is exhausted.
        retry_history (List[str]): Single-slot holder for the per-file log of failed
            rewrite attempts (one line per attempt with its reason); an empty string
            means no failures yet. Replayed to the LLM in the prompt of the next
            retry so it can see what went wrong.
        rabbit_manager (RabbitMQClient): Message queue facade; consumes the raw-HTML
            file references from the crawler's queue (crawler_json_queue) and
            publishes the extracted story JSON to the processed-stories queue
            (scrapper_json_queue).
    """
    file_content     : List[Optional[str]]
    rewrite_history  : List[Optional[str]]
    pages_number     : List[int]
    pages_saved      : List[int]
    retries_max      : int
    retry_counter    : List[int]
    retry_history    : List[str]
    rabbit_manager   : RabbitMQClient


# The agentic system
# ------------------------------------------------------------------
# Core batch driver of the pipeline. Workflow:
#   1. Define the LangGraph node(s) - here the story rewrite agent.
#   2. Assemble and COMPILE the graph exactly once: it is a reusable
#      state machine for a single file-processing cycle.
#   3. Seed the initial AgenticState (retry budget, RabbitMQ manager).
#   4. External driver loop: feed each file named in a message of the
#      crawler's queue through the compiled graph, bump the saved
#      counter, and stop at the pages limit or when the queue is
#      exhausted.
#   5. Log a summary and return the run error code.
# ------------------------------------------------------------------
async def run_agentic_scraper(llm_base : ChatOpenAI, pages_max : int, retries_max : int,
                              docs_folder_name : str, rabbit_manager : RabbitMQClient) -> int :

    # ----------------------------   A G E N T S   ----------------------------

    # The agent to create the rewrite of the story in two stages.
    # Per-file workflow:
    #   * the driver uploads the raw HTML of the current file (the
    #     .html.gz archive named in the queue message) into
    #     state["file_content"];
    #   * stage 1 (extraction): the agent sends it to the LLM with the
    #     dense-extraction prompt (thinking enabled, high reasoning
    #     effort) asking for the single largest coherent story block -
    #     verbatim, no boilerplate / ads / menus;
    #   * stage 2 (refine): the extracted text goes through a light
    #     editorial-polish prompt (Strict Literary Archivist) that repairs
    #     tag-stripping artifacts without ever simplifying or flattening
    #     the author's prose density (the high perplexity is preserved);
    #   * on success the agent appends the rewrite to rewrite_history and
    #     the driver persists it (story document, queue publish or DEBUG
    #     progress log, saved counter); on a failure in either stage it
    #     logs which stage went wrong to retry_history (replayed to the
    #     LLM on the next retry), retries up to retries_max using
    #     retry_counter, and once the retries are exhausted the
    #     file_content slot is set to None.
    async def initial_rewrite_agent(state : AgenticState) -> None :

        # Nothing uploaded (unreadable or empty input file): skip this file.
        if not state["file_content"][0] :
            logger.error(f"No content uploaded for {html_file_name}. Skipping the file.")
            state["file_content"][0] = None
            state["rewrite_history"][0] = None
            state["retry_counter"][0] = 0
            return None

        # Replay the log of failed attempts so the LLM can see what went wrong
        retry_attempts_section = ""
        if state["retry_history"][0] :
            retry_attempts_section = (
                f"# PREVIOUS FAILED ATTEMPTS (adjust your approach accordingly)\n"
                f"{''.join(f'{i+1}. {s}\n' for i, s in enumerate(state['retry_history'][0].splitlines()))}\n\n"
            )

        # Enhanced prompt for investigating web pages and extracting content
        initial_rewrite_agent_prompt = (
            f"# ROLE:\n"
            f"You are an expert dense content-extraction engine. Your task is to analyze a raw HTML document and isolate only the primary dense story block contained within it.\n"
            f"The page may contain:\n"
            f" * navigation menus\n"
            f" * summaries\n"
            f" * links to other stories\n"
            f" * sidebars\n"
            f" * footers\n"
            f" * advertisements\n"
            f" * embedded widgets\n"
            f" * fragments of unrelated text\n\n"
            f"# OBJECTIVE:\n"
            f"Your goal is to identify and return the single largest coherent block of continuous text, optimized for linguistic density: the longest continuous block of natural-language prose with high vocabulary information density (entropy) forming a narrative or article—not boilerplate, not metadata, not UI text.\n\n"
            f"# RULES:\n"
            f" 1. Ignore all HTML tags, scripts, styles, and attributes. Extract the raw text stream.\n"
            f" 2. Ignore all text that is clearly:\n"
            f"     * menus\n"
            f"     * headers/footers\n"
            f"     * cookie notices\n"
            f"     * ads or sponsored content\n"
            f"     * \"related stories\" lists\n"
            f"     * comments\n"
            f"     * captions\n"
            f"     * author bios\n"
            f"     * newsletter signup prompts\n"
            f" 3. If multiple story-like blocks exist, choose the one that maximizes text surface area: \n"
            f"     * longest in total word count\n"
            f"     * most internally coherent across paragraph boundaries\n"
            f"     * rich vocabulary and intricate, natural narrative flow\n"
            f" 4. Do not summarize, omit paragraphs, or condense. Return the full story verbatim as plain text.\n"
            f" 5. Do not include any text outside the main story.\n\n"
            f"# OUTPUT FORMAT:\n"
            f" * Return only the extracted story as plain text.\n"
            f" * No explanations.\n"
            f" * No HTML.\n"
            f" * No headings.\n"
            f" * No commentary.\n\n"
            f"{retry_attempts_section}"
            f"USER INPUT:\n"
            f"{state['file_content'][0]}"
        )

        # Stage 1: dense extraction of the primary story block
        # (thinking/reasoning effort are enabled on the llm_base client itself
        # via extra_body: vLLM chat_template_kwargs - a per-call RunnableConfig
        # is NOT forwarded to the API by langchain_openai)
        extraction = ""
        error_message = None
        try :
            input_messages = [ SystemMessage(content=initial_rewrite_agent_prompt),
                               HumanMessage(content="Rewrite story to remove all unrelated pieces.") ]
            response = await llm_base.ainvoke( input_messages )
            extraction = response.content.strip() if ( response is not None and response.content ) else ""
        except Exception as e :
            logger.error(f"LLM extraction failed for {html_file_name}, error: {e}")
            error_message = str(e)

        if not extraction :
            # Failed extraction: report which stage went wrong for the next
            # retry, then retry up to retries_max using retry_counter
            state["retry_counter"][0] += 1
            state["retry_history"][0] += (
                f"Attempt {state['retry_counter'][0]}: Extraction stage: LLM call failed with error: {error_message}\n"
                if error_message else
                f"Attempt {state['retry_counter'][0]}: Extraction stage: LLM returned an empty or non-text response\n" )
            if state["retry_counter"][0] >= state["retries_max"] :
                logger.error(f"Exhausted {state['retries_max']} rewrite attempts for {html_file_name}.")
                state["file_content"][0] = None
                state["rewrite_history"][0] = None
                state["retry_counter"][0] = 0
            return None

        # Stage 2: light editorial polish of the extracted story
        refine_agent_prompt = (
            f"# ROLE:\n"
            f"You are a Strict Literary Archivist and Text Restoration Engine. You receive a story already extracted from a raw HTML page and produce its final clean edition. Your objective is to make the text flawless while preserving 100% of the author's original linguistic complexity and perplexity profile.\n\n"
            f"# CORE HEURISTIC:\n"
            f"The value of this text lies in its exact lexical and micro-syntactic fidelity. Do not smooth out, modernize, simplify, or normalize the prose. The original rare vocabulary, varying sentence lengths, erratic punctuation quirks, and grammatical idiosyncrasies must remain completely untouched - you clean the text, you never rewrite it.\n\n"
            f"# YOU MAY:\n"
            f" 1. Repair grammar and punctuation broken by tag stripping (sentences split across layout boundaries, orphaned fragments, doubled punctuation). If a sentence was broken across HTML tag boundaries, stitch it back seamlessly - DO NOT hallucinate or insert transitional words or conjunctions to bridge the gap.\n"
            f" 2. Decode HTML entities and mojibake (&nbsp;, &#39;, double-encoded UTF-8, ...) back into the proper characters.\n"
            f" 3. Remove residual boilerplate fragments: ad leftovers, navigation crumbs, cookie banners, 'read more' / 'continue reading' tails, orphaned captions, duplicated paragraphs.\n"
            f" 4. Normalize whitespace (single spaces, single blank lines between paragraphs).\n\n"
            f"# YOU MUST NOT:\n"
            f" 1. Summarize, shorten, or condense. Every fact, figure, quote, ticker symbol, and entity stays.\n"
            f" 2. Add any content: no invented words, transitions, corrections of the author's wording, or commentary.\n"
            f" 3. Translate or simplify. Never flatten the author's prose density - high perplexity, obscure vocabulary, and unusual phrasings are intentional assets and must be preserved.\n"
            f" 4. Use Markdown or any markup: no headings, bullets, bold, links, or block quotes.\n\n"
            f"# OUTPUT FORMAT:\n"
            f" * Return ONLY the final clean story as plain text.\n"
            f" * Zero preamble: you are a node in an automated processing pipeline - no explanations, no introductory or concluding remarks (do not say \"Here is the story\").\n\n"
            f"{retry_attempts_section}"
            f"USER INPUT (the extracted story to refine):\n"
            f"{extraction}"
        )

        story = ""
        error_message = None
        try :
            input_messages = [ SystemMessage(content=refine_agent_prompt),
                               HumanMessage(content="Refine the extracted story per the rules above.") ]
            response = await llm_base.ainvoke( input_messages,
                                               config={"configurable": {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "high"}}} )
            story = response.content.strip() if ( response is not None and response.content ) else ""
        except Exception as e :
            logger.error(f"LLM refine failed for {html_file_name}, error: {e}")
            error_message = str(e)

        if not story :
            # Failed refine: report which stage went wrong for the next
            # retry, then retry up to retries_max using retry_counter
            state["retry_counter"][0] += 1
            state["retry_history"][0] += (
                f"Attempt {state['retry_counter'][0]}: Refine stage: LLM call failed with error: {error_message}\n"
                if error_message else
                f"Attempt {state['retry_counter'][0]}: Refine stage: LLM returned an empty or non-text response\n" )
            if state["retry_counter"][0] >= state["retries_max"] :
                logger.error(f"Exhausted {state['retries_max']} rewrite attempts for {html_file_name}.")
                state["file_content"][0] = None
                state["rewrite_history"][0] = None
                state["retry_counter"][0] = 0
            return None

        # Successful rewrite: hand the story over to the driver, which saves
        # the story document and publishes the story JSON.
        state["rewrite_history"][0] = story
        return None


    # --- Graph Assembly ---
    # This section constructs the LangGraph state machine that orchestrates the crawler agents.
    # The graph defines the workflow for a SINGLE file-processing cycle: the START
    # edge enters the rewrite node, which loops back onto itself until a non-empty
    # rewrite has been produced; once the rewrite exists the cycle exits to END.
    # The compiled graph is then reused by the external driver loop for every
    # file named in the crawler's queue messages.
    #
    #    EXTERNAL WHILE FILES LOOP :
    #           .--------------------------.
    #           V                          |
    #    START -.-> initial_rewrite_agent -.->  END
    #
    builder = StateGraph(AgenticState)
    builder.add_node("initial_rewrite_agent",   initial_rewrite_agent)
    builder.add_edge(START, "initial_rewrite_agent")
    builder.add_conditional_edges("initial_rewrite_agent", lambda state : str(state["rewrite_history"][0] == ""),
                                  {"True" : "initial_rewrite_agent", "False" : END})
    
    # Compile ONCE outside the loop
    graph = builder.compile()

    # Seed the initial state for the driver loop.
    state = AgenticState({ "file_content"     : [None],
                           "rewrite_history"  : [""],
                           "pages_number"     : [0],
                           "pages_saved"      : [0],
                           "retries_max"      : retries_max,
                            "retry_counter"    : [0],
                            "retry_history"    : [""],
                            "rabbit_manager"   : rabbit_manager,
                        })

    # External driver loop: reuses the compiled 'graph' across all file
    # iterations. The work items are the pending messages of the crawler's
    # queue (each one names a stored raw HTML file via file_path), taken
    # one at a time:
    #   * readonly/debug mode - every message is put back into the queue
    #     as soon as it is read (the queue is never drained), so the run
    #     can be repeated. Because the peeked head message is requeued to
    #     the head, the same file would otherwise be observed forever: the
    #     run therefore stops as soon as a file seen in this run is
    #     re-observed (the queue has cycled back to it).
    #   * production mode     - a message is consumed (removed) from the
    #     queue as soon as it is picked up.
    # For every message: the raw HTML is uploaded from the file into the
    # state, the graph runs one rewrite cycle (the agent reads the current
    # file name from the loop variable). On success the driver saves the
    # story document to docs_folder_name, publishes the story JSON to the
    # processed-stories queue (or logs progress in readonly/debug mode);
    # then the slots are cleared. The loop stops when the saved-pages
    # counter reaches the limit, the queue is exhausted, or (readonly)
    # the queue cycles back to an already-seen file.
    async def iter_queue_messages() :
        seen_file_paths : set = set()
        while True :
            if ( pages_max > 0 and state["pages_saved"][0] >= pages_max ) : break  # break on pages limit
            status, message = await rabbit_manager.receive_json(queue_name=CRAWLER_JSON_QUEUE)
            if status != 1 or message is None : break  # the queue is exhausted (or not connected)
            if rabbit_manager.readonly :
                # The readonly peek always re-reads the (requeued) head
                # message: if we have seen this file in this run already,
                # the queue cycled back to it - stop instead of looping
                # forever on the same file.
                file_path_key = message.get("file_path")
                if file_path_key in seen_file_paths : break
                seen_file_paths.add(file_path_key)
            yield message

    html_file_name = ""
    async for message in iter_queue_messages() :
        source_file_path = message.get("file_path")
        if not source_file_path :
            logger.error(f"Malformed queue message (no file_path): {message}")
            continue
        html_file_name = os.path.basename(source_file_path)
        file_name = html_file_name.removesuffix(".gz")  # the crawler stores .html.gz archives
        logger.debug(f"Read queue message: {html_file_name} (url: {message.get('url', '')})")

        # Upload: read the raw HTML of the current file from disk
        try :
            with gzip.open(source_file_path, "rt", encoding="utf-8") as f :
                file_content = f.read()
        except Exception as e :
            logger.error(f"Failed to upload {source_file_path}, error: {e}")
            continue

        state["file_content"][0]  = file_content
        state["pages_number"][0] += 1

        state = await graph.ainvoke(state) # Call the agent 

        story = state["rewrite_history"][0]
        if story :
            # Save the story document to the docs directory
            docs_file_path = os.path.join(docs_folder_name, os.path.splitext(file_name)[0] + ".md")
            os.makedirs(docs_folder_name, exist_ok=True)
            with open(docs_file_path, "w", encoding="utf-8") as f :
                f.write(story)

            story_payload = { "url"       : message.get("url", ""),
                              "file_path" : docs_file_path,
                              "story"     : story,
                              "timestamp" : datetime.datetime.now().strftime('%Y-%m-%d'), }
            if rabbit_manager.readonly :  # debug: do not write to the processed-stories queue, just log the progress
                logger.info(f"DEBUG. Saved story to {docs_file_path} (url: {story_payload['url']}, {len(story)} chars); "
                            f"the processed-stories queue is NOT written in DEBUG mode.")
                logger.debug(f"DEBUG. Processed-stories message (not published): url={story_payload['url']}, "
                             f"file_path={story_payload['file_path']}, timestamp={story_payload['timestamp']}")
            else :
                # Publish the story JSON to the processed-stories queue
                await state["rabbit_manager"].publish_json(story_payload)

            state["pages_saved"][0] += 1
        state["rewrite_history"][0]  = ""
        state["retry_history"][0]    = ""
        state["retry_counter"][0]    = 0

    logger.info(f"Processed pages: {state['pages_number'][0]}")
    logger.info(f"Saved pages:     {state['pages_saved'][0]}")
    logger.info("======================================================")

    return 0


async def run_stealth_graph(docs_folder_name : str = "../docs/", pages_max : int = -1, retries_max : int = 1,
                            rabbit_host : str = "localhost", rabbit_port : int = 5672) :
    # ------------------------------------------------------------------
    # System bootstrap. Workflow:
    #   1. Plug in the LLM (OpenAI-compatible vLLM endpoint) with tool support.
    #   2. Health gate: minimal "ping" completion with a 30s timeout; any
    #      failure aborts the run with error code -1.
    #   3. Connect RabbitMQ (reads the crawler's queue, publishes to the
    #      processed-stories queue; readonly mode: non-destructive reads,
    #      no writes); hard-fail with code 1 if unavailable.
    #   4. Launch the agentic batch driver with the manager wired in.
    #   The manager is an async context manager: the connection is released
    #   automatically on the way out, and the driver's error code becomes
    #   the return value of this function.
    # ------------------------------------------------------------------
    # Plug-in LLM.
    # chat_template_kwargs (vLLM) are set on the client via extra_body:
    # a per-call RunnableConfig is NOT forwarded to the OpenAI-compatible API
    # by langchain_openai, so thinking/reasoning effort would otherwise
    # silently never reach the server.
    llm_base = ChatOpenAI( base_url=f"{BASE_URL}/v1",
                           api_key=BASE_API_KEY,
                           model_name=MODEL_NAME,
                           max_tokens=LLM_MAX_OUTPUT_TOKENS,
                           temperature=0.1,
                           extra_body={ "chat_template_kwargs" : { "enable_thinking" : True, "reasoning_effort" : "high" } }, )
    # Test LLM connectivity with a simple ping (a trivial prompt; a strict
    # timeout catches a dead endpoint)
    try :
        response = await asyncio.wait_for( llm_base.ainvoke("ping"), timeout=30.0 )
        assert bool(response and response.content) , "Failed to connect LLM!"
    except Exception as e :
        logger.error(f"LLM Failed with error code {e}")
        return -1

    # Link RabbitMQ 
    async with RabbitMQClient(host=rabbit_host, port=rabbit_port,
                              publish_queue_name=PROCESSED_STORY_QUEUE, receive_queue_name=CRAWLER_JSON_QUEUE,
                              username="scrapper", password="scrapper") as rabbit_manager :
        rabbit_err_code = await rabbit_manager.connect()
        if rabbit_err_code:
            logger.error(f"Failed to connect RabbitMQ at host {rabbit_manager.host} port {rabbit_manager.port}")
            return 1

        if rabbit_manager.readonly :
            logger.info(f"DEBUG. RabbitMQ readonly mode: reads from {CRAWLER_JSON_QUEUE} are non-destructive "
                        f"(messages are never removed); the publish to {PROCESSED_STORY_QUEUE} is skipped, "
                        f"progress is logged instead.")

        # Launch agents
        agentic_error_code = await run_agentic_scraper(llm_base=llm_base, pages_max=pages_max, retries_max=retries_max,
                                                       docs_folder_name=docs_folder_name, rabbit_manager=rabbit_manager)

    return agentic_error_code


if __name__ == "__main__":
    """
    Main entry point for command-line execution.

    Parses command-line arguments and launches the batch HTML->story pipeline
    with user-specified output folder, page limit, retry budget and
    connection parameters.
    """
    import argparse
    parser = argparse.ArgumentParser(description="The batch HTML->story scraper.")

    parser.add_argument("--docs_folder_name",  "-dn",  type=str,  default="../docs/",   dest="docs_folder_name",  help="Name of docs dir.")
    parser.add_argument("--pages_max",         "-n",  type=int,  default=-1,           dest="pages_max",         help="Maximal number of pages to extract.")
    parser.add_argument("--retries_max",        "-m",  type=int,  default=1,            dest="retries_max",       help="Maximal number of retries.")
    parser.add_argument("--rabbit_host",       "-bh",  type=str,  default="localhost",  dest="rabbit_host",       help="RabbitMQ host.")
    parser.add_argument("--rabbit_port",       "-bp",  type=int,  default=5672,        dest="rabbit_port",       help="RabbitMQ port.")
    args = parser.parse_args()

    return_code = asyncio.run(run_stealth_graph(
        docs_folder_name=args.docs_folder_name,
        pages_max=args.pages_max,
        retries_max=args.retries_max,
        rabbit_host=args.rabbit_host,
        rabbit_port=args.rabbit_port))
    logger.info(f"Agent finished with {return_code} error code.")
    
# Backlog:
# Consider hierarchical merging
