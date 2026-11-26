"""
Specialized Yahoo Finance news crawler.

A streamlined clone of crawler_agent.py that uses hard-coded URL shortcuts
instead of generic traversal heuristics:
  * the entry point defaults to https://ca.finance.yahoo.com/
  * only links whose HOSTNAME is finance.yahoo.com / *.finance.yahoo.com AND
    whose PATH contains "/news/" are extracted / traversed
    (e.g. https://ca.finance.yahoo.com/news/elon-musk-...-131435467.html,
     https://ca.finance.yahoo.com/news/gop-counts-farmers-votes-costly-115740788.html)
  * the entry (zero-level) page is a STRUCTURAL HUB: it is visited as-is,
    never saved, never marked visited (revisitable, EXCLUDED from the visited
    set); it is always seeded directly by the driver, and its /news/ links are
    always pushed to the DFS stack - hub detection is the URL shortcut's job,
    not the classifier's
  * every deeper page (only /news/ candidates are ever pushed) gets a binary
    CONTENT verdict STORY | IRRELEVANT instead of the generic
    STORY / AGGREGATOR / IRRELEVANT three-way split:
        STORY iff the page has ENOUGH CONTENT (a substantive single article)
        AND FINANCES (a financial subject) AND a FINISHED, CONSISTENT PLOT;
        IRRELEVANT is everything that is not STORY
    * before classification, a deeper article page gets a mechanical (NO LLM)
      first-story "more content" check: the article is a single <article> and a
      truncated body is hidden behind a "Story Continues" <button> + a
      <div class="read-more-wrapper"> inside it; the button's reveal handler only
      fires for a TRUSTED click (execute_script cannot produce one), so the check
      force-reveals that hidden continuation (scoped to the first <article> so a
      later story's content is never touched), re-fetches the sanitized content,
      and uses the expanded content when it grew (otherwise the flow continues
      unchanged)
   * "/news/" links are pushed to the DFS stack as soon as they are extracted
    (every page, the entry hub included), and only while the current depth is
    below deepness_max (deepness_max < 0 = unlimited depth); the (polled)
    link extraction is skipped entirely once the depth is reached, since no
    push is possible then
  * the DFS stack traversal (pruning empty levels, popping the next URL,
    skipping already-visited URLs) is owned by the driver loop
    (run_agentic_crawler_yahoo); the page node only fetches / classifies the
    current URL and pushes newly found links in place
  * the LLM is NEVER given browser tools: the script alone controls the page
    access (navigate / content / links), so no stealth-browser MCP tool data
    appears in any LLM prompt; the classifier runs on the tools-free llm_base
Everything else (stealth MCP browser, LangGraph single-node page cycle +
external DFS loop, RabbitMQ story publishing, Redis visited-URL set, gzip
HTML archiving) mirrors the generic script.

STATUS: implemented (phase 2).
"""

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
# file_handler = logging.FileHandler('crawler_agent_yahoo.log')
# file_handler.setLevel(logging.DEBUG)
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)

from typing import TypedDict, Optional, List, Dict, Any
import os
import sys
import asyncio
import re
import json
import datetime
import uuid
import gzip
import httpx
from urllib.parse import urlparse

# Make the parent 'src' directory importable so that modules moved there
# (rabbit_driver, redis_driver) can be imported as top-level modules.
_SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC_ROOT)
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, RemoveMessage
from langgraph.graph import StateGraph, START, END

from web_tools import clean_dom, BrowserProxy, StealthMCPManager
from llm_config import BASE_URL, BASE_API_KEY, MODEL_NAME

from rabbit_driver import RabbitMQClient
from redis_driver import RedisDBClient


# --- Yahoo Finance crawl defaults ------------------------------------------
# Entry point of the crawl (overridable via the CLI -u flag).
DEFAULT_ENTRY_URL = "https://ca.finance.yahoo.com/"
# Path marker of a Yahoo Finance news candidate URL.
NEWS_PATH_MARK = "/news/"
# Hostname suffix of the Yahoo Finance domain: matches finance.yahoo.com and
# every regional subdomain (e.g. ca.finance.yahoo.com, uk.finance.yahoo.com).
FINANCE_HOST_SUFFIX = ".finance.yahoo.com"

# --- LLM (vLLM) settings - shared endpoint constants (BASE_URL / BASE_API_KEY
# / MODEL_NAME) come from llm_config; only the context/output budgets are
# yahoo-specific ------------------------------------------------------------
LLM_BASE_CONTEXT_LEN = 188416
LLM_MAX_OUTPUT_TOKENS = 32768

# --- SPA hydration delays -----------------------------------------------------
# The Yahoo SPA can still be hydrating right after navigation, so: the link
# extraction is polled up to retries_max times (LINKS_POLL_DELAY_SECONDS
# apart; stops early as soon as any link is found), and the empty-content
# refetches of the fetch loop are CONTENT_RETRY_DELAY_SECONDS apart (the SPA
# needs a moment to hydrate between the re-navigation and the next fetch).
LINKS_POLL_DELAY_SECONDS = 1.5
CONTENT_RETRY_DELAY_SECONDS = 1.5

# --- DEBUG MODE: readonly (non-destructive) back-end semantics ----------------
# When DEBUG is set, the RabbitMQ and Redis clients are constructed in readonly
# mode: the crawler runs end-to-end, but no data is pushed into the real
# redis/rabbit pipelines (writes are buffered in memory and lost on exit), so
# debug runs are repeatable and non-destructive. The flag is re-read at
# client-construction time (run_stealth_graph_yahoo), NOT at import time: the
# env var may be set after the module import (e.g. by a test harness).
def _debug_mode() -> bool :
    return os.environ.get('DEBUG', '').lower() in ('true', '1', 'yes')


def is_yahoo_news_url(url: str) -> bool:
    """
    The URL shortcut of the Yahoo Finance crawl.

    Returns True iff `url` is an on-site news candidate:
      1. its hostname is finance.yahoo.com or ends with FINANCE_HOST_SUFFIX
         (i.e. any regional Yahoo Finance domain, e.g. ca.finance.yahoo.com); AND
      2. its PATH contains the NEWS_PATH_MARK marker
         (e.g. /news/elon-musk-says-...-131435467.html).
    Matching is done on the path component only (via urlparse), so query
    strings / fragments (e.g. "...?ref=news") never match.
    The zero-level (entry) URL is EXEMPT from this check by the caller - this
    predicate is applied only to URLs popped from / pushed to the recursion
    lists, never to the entry URL itself.
    """
    parsed = urlparse(url)
    if not parsed.hostname:
        return False
    host_ok = (parsed.hostname == "finance.yahoo.com"
               or parsed.hostname.endswith(FINANCE_HOST_SUFFIX))
    return host_ok and NEWS_PATH_MARK in parsed.path


async def get_prompt_tokens_length(prompt: str) -> int :
    """
    Count the tokens of a string via the vLLM /tokenize endpoint.
    The Yahoo-crawler-local copy of the crawler_agent token counter: this
    crawler's LLM is tools-free (no bound tools, so the tools token count
    is 0), so this helper counts the classification prompt itself and the
    shared tools-flavored get_tools_tokens_length helper is not used here.
    """
    headers = {"Authorization": f"Bearer {BASE_API_KEY}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(base_url=BASE_URL, headers=headers, timeout=60.0) as client :
        response = await client.post("/tokenize", json={"model": MODEL_NAME, "prompt": prompt})
        response.raise_for_status()
    return response.json().get("count", len(response.json().get("tokens", [])))


# --- First-story "more content" expansion (Yahoo-specific, NO LLM) -----------
# A Yahoo /news/ article page is a single <article data-testid=
# "article-content-wrapper"> (the FIRST story). A truncated article hides the
# rest of the body behind a "Story Continues" <button class="... readmore-button
# ..."> inside that <article>; the hidden continuation lives in
# <div class="read-more-wrapper" data-testid="read-more" style="display:none">.
#
# The button's reveal handler is a React synthetic-event handler that only fires
# for TRUSTED user clicks; a programmatic el.click() or a dispatched synthetic
# MouseEvent (isTrusted=false) does NOT reveal the content (verified: the DOM is
# byte-for-byte unchanged after such a click, even after several seconds). So
# instead of pressing the button, the routine FORCE-REVEALS the hidden
# continuation container directly: it overrides the hiding CSS (display /
# height / max-height / overflow / visibility / opacity, all !important) so the
# container becomes a visible, non-zero-sized block, and the re-fetched
# sanitized content (which drops display:none / 0x0 elements) GROWS. Because
# the page's own JS can re-hide the container (a re-render) between the reveal
# and the re-fetch, a MutationObserver keeps it revealed (re-applying the
# overrides if it is hidden again); the observer is discarded on the next
# navigation. The "Story Continues" button is also hidden (cosmetic) so its
# label does not linger in the saved content.
#
# LATER stories live OUTSIDE the <article> (in related-content-stream) and carry
# no read-more markup, so scoping the reveal to the first <article> guarantees a
# later story's content is never touched. The check is purely mechanical (no
# LLM, classifier untouched): detect the hidden continuation, force-reveal it,
# and compare the sanitized content size before / after the reveal.
EXPANSION_SETTLE_SECONDS = 2.0
MORE_CONTENT_PHRASES = (
    "story continues", "story continue", "read full story", "read the full story",
    "full story", "continue reading", "continue to read", "continue story",
    "read more", "show more", "see more", "load more", "view full article",
    "full article", "show full article", "expand article", "more of this story",
)

# In-browser JS (self-contained IIFE, no args): scope to the FIRST story (the
# first <article>), FORCE-REVEAL any hidden continuation container (the
# button's reveal handler needs a trusted click, which execute_script cannot
# produce), and hide the "Story Continues" button (cosmetic). Returns a JSON
# string {"revealed": N, "label": L} where N is the number of hidden containers
# revealed (0 = nothing hidden to reveal) and L is a short preview of the
# continuation text (the first revealed container's label) or, failing that,
# the "Story Continues" button label ("" when none matched).
_FIRST_STORY_REVEAL_JS = r"""
(() => {
    if (!document.body) return JSON.stringify({revealed: 0, label: ""});
    // First story = the first <article> (the article-content-wrapper on Yahoo).
    const article = document.querySelector('article[data-testid="article-content-wrapper"]')
        || document.querySelector('article');
    if (!article) return JSON.stringify({revealed: 0, label: ""});
    const phrases = __PHRASES__;
    const norm = (s) => (s || '').replace(/\s+/g, ' ').trim().toLowerCase();
    // The "Story Continues" button (cosmetic: hide it so its label does not
    // linger in the saved content). Primary = the stable read-more class,
    // fallback = the first VISIBLE button whose label matches a phrase.
    let firstBtn = article.querySelector('button.readmore-button');
    if (!firstBtn) {
        const cands = article.querySelectorAll('button, a, [role="button"], input[type="button"], input[type="submit"]');
        for (const el of cands) {
            const style = window.getComputedStyle(el);
            if (style.display === 'none' || style.visibility === 'hidden' || el.getClientRects().length === 0) continue;
            const label = norm(el.innerText || el.getAttribute('aria-label') || el.getAttribute('title'));
            if (label && phrases.some(p => label.includes(p))) { firstBtn = el; break; }
        }
    }
    const labelOf = (el) => el ? norm(el.innerText || el.getAttribute('aria-label') || el.getAttribute('title')) : "";
    // Force a continuation container visible (!important beats the CSS class
    // that hides it).
    const forceReveal = (el) => {
        ['display:block', 'visibility:visible', 'opacity:1', 'height:auto', 'max-height:none', 'overflow:visible']
            .forEach((d) => { const kv = d.split(':'); el.style.setProperty(kv[0], kv[1], 'important'); });
    };
    const isHidden = (el) => {
        const style = window.getComputedStyle(el);
        const rect = el.getBoundingClientRect();
        return el.hidden || el.getAttribute('aria-hidden') === 'true'
            || style.display === 'none' || style.visibility === 'hidden'
            || parseFloat(style.opacity) === 0 || (rect.width === 0 && rect.height === 0);
    };
    // Force-reveal hidden continuation containers (display:none / 0x0) inside
    // the first article, and KEEP them revealed: the page's JS may re-hide a
    // container (re-render), so a MutationObserver re-applies the reveal if one
    // is hidden again. The observers are attached to the document's nodes and
    // are discarded on the next navigation, so no manual cleanup is needed.
    const containers = article.querySelectorAll('.read-more-wrapper, [data-testid="read-more"]');
    let revealed = 0;
    const observers = window.__yf_readmore_observers = window.__yf_readmore_observers || [];
    containers.forEach((el) => {
        if (!isHidden(el)) return;
        forceReveal(el);
        revealed++;
        try {
            const mo = new MutationObserver(() => { if (isHidden(el)) forceReveal(el); });
            mo.observe(el, { attributes: true, attributeFilter: ['style', 'hidden', 'class'] });
            observers.push(mo);
        } catch (e) { /* observer unavailable - the one-shot reveal still applies */ }
    });
    let label = "";
    for (const c of containers) { label = labelOf(c); if (label) break; }
    if (!label && firstBtn) label = labelOf(firstBtn);
    if (firstBtn) firstBtn.style.setProperty('display', 'none', 'important');
    return JSON.stringify({revealed: revealed, label: label});
})()
""".replace("__PHRASES__", json.dumps(list(MORE_CONTENT_PHRASES)))


# BrowserProxy (the MCP proxy browser) is NOT cloned here: the single
# canonical copy lives in web_tools.py and is imported above. It provides
# direct single-round-trip session calls (instance_id injected, transport
# errors tolerated as None), retrying get_content() / get_sanitized_content()
# with offloaded-file handling and JS-based get_links() that filters out
# static assets.


# StealthMCPManager (the MCP server / session / browser lifecycle manager)
# is NOT cloned here either: the single canonical copy lives in web_tools.py
# and is imported above.


#   --------   A G E N T I C   P A R T   --------

# The LLM endpoint constants are NOT cloned here: the single canonical copies
# live in llm_config.py and are imported above. The prompt token counter
# (get_prompt_tokens_length) is defined locally: this crawler's LLM has no
# tools, so the shared tools-flavored helper does not apply.


class YahooAgenticState(TypedDict):
    """
    TypedDict defining the complete state structure of the Yahoo Finance news
    crawler. Mirrors AgenticState from crawler_agent.py minus the
    access-agent bookkeeping (page_history / retry_counter): this crawler has
    no LLM tool loop - the script alone drives the browser - so the single
    page_classify_agent node both fetches the page content and issues the
    verdict. Single-element lists hold mutable scalars (LangGraph convention
    of this codebase: the graph nodes read/write state in place); url_current
    is the exception - a plain str written only by the driver loop between
    graph invocations (the node treats it as read-only).
    Attributes:
        date_str (str): The date when the crawler started.
        urls (List[List[str]]): Nested DFS stack of candidate news URLs. Each
            inner list is a depth level; ONLY URLs satisfying
            is_yahoo_news_url() are ever pushed (the entry URL is exempt);
            URLs are popped (LIFO) as visited. All traversal mutations
            (prune / pop / visited-skip) happen in the driver loop
            (run_agentic_crawler_yahoo); the node only reads len(urls) (hub
            check / depth gate) and appends new levels in place.
        url_current (str): Current active URL. Plain str written only by the
            driver loop between graph invocations (traversal hand-off); the
            node treats it as read-only.
        deepness_max (int): Maximum depth allowed for tree exploration.
        pages_saved (List[int]): Counter of saved financial stories.
        retries_max (int): Maximum refetch attempts for empty page content.
        browser_state (List[str]): Current browser page content (cleaned DOM);
            always emptied at the end of a page cycle.
        html_folder_name (str): Folder to store the downloaded html files.
        mcp_manager (StealthMCPManager): Handler for MCP / stealth-browser tools.
        rabbit_manager (RabbitMQClient): Handler for message queue operations
            (publishing saved stories to crawler_json_queue).
        redis_manager (RedisDBClient): Handler for visited URL tracking
            (shared crawler_visited_urls set).
    """
    date_str         : str
    urls             : List[List[str]]
    url_current      : str
    deepness_max     : int
    pages_saved      : List[int]
    retries_max      : int
    browser_state    : List[str]
    html_folder_name : str
    mcp_manager      : StealthMCPManager
    rabbit_manager   : RabbitMQClient
    redis_manager    : RedisDBClient


# --- Crawl abort signal -------------------------------------------------------------
class CrawlerPublishError(Exception):
    """
    CRITICAL crawl-abort signal: a STORY page was saved to disk but its
    RabbitMQ publish failed (publish_json returned 0). The page is neither
    counted nor marked visited (the URL stays eligible for a future run),
    the story file stays on disk (the broker may hold the message without
    an ack - the path is logged), and the driver loop stops the crawl,
    returning error code 2.
    """


# The agentic system
async def run_agentic_crawler_yahoo(url_string : str, mcp_manager : StealthMCPManager,
                                    llm_base_context_len : int, llm_base : ChatOpenAI,
                                    deepness_max : int, pages_max : int, retries_max : int,
                                    html_folder_name : str, rabbit_manager : RabbitMQClient, redis_manager : RedisDBClient) -> int :
    """
    Main entry point of the Yahoo Finance news crawler.
    Specialized clone of the run_agentic_crawler function (crawler_agent.py)
    that orchestrates a depth-first tree-search of Yahoo Finance news pages
    using the MCP-controlled stealth browser and a tools-free LLM classifier,
    with the following shortcuts:
       1. The zero-level entry URL is a structural hub: visited as-is, never
          saved, never marked visited (revisitable, EXCLUDED from the visited
          set); it is seeded directly by the driver; its /news/ links are
          always pushed.
      2. Every URL popped from / pushed to the recursion lists satisfies
         is_yahoo_news_url() (the filter is applied at push time).
      3. Classification of deeper pages is a binary CONTENT verdict
         (STORY | IRRELEVANT); hub/aggregator handling is absorbed by the
         URL shortcut, not the classifier.
       4. "/news/" links are pushed to the DFS stack right at the extraction
          step (every page, the entry hub included), and only while
          len(urls) < deepness_max; the polled link extraction is skipped
          entirely at the depth limit.
       5. The LLM never calls browser tools: page access (navigate / content
          / links) is controlled by this script, and no MCP tool data is ever
          placed in an LLM prompt.
        6. The DFS stack traversal (prune / pop / visited-skip) is owned by
           this driver loop: it seeds url_current with the entry hub
           (urls[0][0]), ainvokes the page cycle, then traverses the stack for
           the next URL of the next iteration (no state re-assignment); the
           node only pushes newly found links in place.
         7. Before classification, a deeper article page gets a mechanical (NO
            LLM) first-story "more content" check: the article is a single
            <article> and a truncated body is hidden behind a "Story Continues"
            <button> + a <div class="read-more-wrapper"> inside it. The button's
            reveal handler only fires for a TRUSTED click (execute_script cannot
            produce one), so the check force-reveals that hidden continuation
             (scoped to the first <article> so a later story's content is never
             touched) via the managed browser, re-fetches the sanitized content,
             and uses the expanded content when it grew (otherwise the flow
             continues unchanged).
    Args:
        url_string (str): The entry point URL (defaults to ca.finance.yahoo.com
            at the run_stealth_graph_yahoo / CLI level).
        mcp_manager (StealthMCPManager): Manager for MCP server and browser lifecycle.
        llm_base_context_len (int): Token budget for the classification prompt.
        llm_base (ChatOpenAI): Tools-free LLM instance for the classification call.
        deepness_max (int): Maximum depth for tree exploration (-1 = unlimited).
        pages_max (int): Maximum number of saved stories (-1 = unlimited).
        retries_max (int): Maximum refetch attempts for empty page content.
        html_folder_name (str): Root folder for the saved html archives
            (stories are stored under html_folder_name/date_str/).
        rabbit_manager (RabbitMQClient): Message queue client for publishing
            saved stories to the downstream processors.
        redis_manager (RedisDBClient): Visited-URL tracking (shared
            crawler_visited_urls set, also used by the generic crawler).
    Returns:
        int: Error code (0 = success, 2 = critical story-publish failure).
    Internal Components:
        - page_classify_agent: fetch + binary STORY | IRRELEVANT content node
        - LangGraph single-node page cycle + external driver loop, which owns
          the DFS stack traversal (prune / pop / visited-skip): seeds
          url_current with the entry hub (urls[0][0]), re-invokes the graph
          per page cycle without state re-assignment, breaks on pages_max /
          a drained stack, aborts on CrawlerPublishError (a critical failed
          story publish -> error code 2), and asserts the clean-up invariant
          (empty browser_state) after each cycle
    """

    # ----------------------------   A G E N T S   ----------------------------

    # This node fetches the popped page and classifies its content.
    # Page access is script-controlled (navigate / get_sanitized_content / get_links):
    # the LLM is only asked for the final content verdict, with no tools.
    async def page_classify_agent(state : YahooAgenticState) -> None :
        """
        Fetch + classify node of the Yahoo Finance news crawler.
        Replaces the generic two-node page_access_agent + page_classify_agent
        pair (crawler_agent.py) with a single node, because the site is
        assumed cooperative: no LLM-driven access actions are needed.
        Workflow:
            1. The driver (run_agentic_crawler_yahoo) has already traversed
               the DFS stack (pruning empty levels, popping the next URL,
               skipping URLs already visited in Redis) and set url_current:
                this node deals only with the current URL. Navigate and fetch
                the sanitized content via BrowserProxy.get_sanitized_content();
                a failed navigation, a fetch failure (exception / transport
                None) or empty content triggers up to retries_max refetch
                attempts (re-navigate + get_sanitized_content, spaced
                CONTENT_RETRY_DELAY_SECONDS apart). Returns without a page if
                the content stays empty.
              2. First-story "more content" check (Yahoo-specific, NO LLM), run
                  only for non-root article pages (is_root is captured here):
                  the first story's hidden "read-more" continuation is
                  force-revealed via execute_script (the "Story Continues"
                  button's reveal handler only fires for a TRUSTED click, which
                  execute_script cannot produce) - scoped to the first <article>
                  so a later story's content is never touched - the sanitized
                  content is then re-fetched and compared to the pre-reveal
                  content. If the content GREW, the expanded content replaces
                  browser_state and the flow continues; otherwise (nothing
                  hidden to reveal, or a transport/JS failure) the flow
                  continues unchanged.
               3. Extract the page links (polled up to retries_max times while
                 the SPA is still hydrating) and filter them through
                 is_yahoo_news_url() (the URL shortcut); the filtered links are
                 pushed onto the DFS stack right here when non-empty - nothing
                 downstream carries links over. Skipped entirely when the
                 depth has reached deepness_max (no push is possible then).
               4. Entry level (structural hub, exempt from the verdict):
                  never saved, never marked visited (revisitable, EXCLUDED from
                  the visited set); its links were already pushed in step 3.
              5. Deeper levels: run the tools-free LLM with the content-focused
                classifier prompt (the SystemMessage carries only the static
                instructions, the HumanMessage carries the page data - URL +
                pruned HTML - and the format reminder) and parse the strict
                verdict:
                    VERDICT: <STORY|IRRELEVANT>
                 STORY (enough content AND finances AND a finished, consistent
                  plot): save the raw HTML (gzip), publish {url, file_path,
                  timestamp} to RabbitMQ, pages_saved += 1, mark the URL visited
                   in Redis (check_and_add). A failed publish (publish_json
                   returned 0) is CRITICAL: the crawl is aborted with error
                   code 2 (CrawlerPublishError) - the page is neither counted
                   nor marked visited (the URL stays eligible for a future run).
               IRRELEVANT (everything else): mark the URL visited in Redis and
               discard. Unparseable / missing verdict defaults to IRRELEVANT.
               Oversized prompt (prompt_tokens > llm_base_context_len): warn,
               hard-skip the page (clear state) and continue.
             6. browser_state is always emptied before returning.
        Returns:
            None (modifies state in-place).
        """

        # --- 1. PAGE FETCH: navigate and fetch the content of the current URL ---
        # The driver (run_agentic_crawler_yahoo) already traversed the DFS
        # stack (pruning empty levels, popping the next URL, skipping URLs
        # already visited in Redis) and set url_current: this node deals only
        # with the current URL; newly found links are pushed straight onto
        # the stack in the extraction block below.
        assert state["url_current"] , "url_current is not set - the driver must pop the next URL before each graph invocation!"
        # Script-controlled page access (no LLM tools): navigate, then fetch
        # sanitized DOM via in-browser JS; up to retries_max refetches on a
        # failed navigation, a fetch failure (exception / transport None from
        # get_sanitized_content) or empty content. The attempts are spaced
        # CONTENT_RETRY_DELAY_SECONDS apart.
        for attempt in range(state["retries_max"] + 1) :
            if attempt > 0 :
                await asyncio.sleep(CONTENT_RETRY_DELAY_SECONDS)
            nav_res = await state["mcp_manager"].managed_browser.navigate(state["url_current"])
            if nav_res is None :
                # A failed navigation would make get_sanitized_content() return the
                # PREVIOUS page's content: never trust it - skip the fetch and
                # let the next attempt re-navigate.
                state["browser_state"][0] = ""
                logger.warning(f"NAVIGATE FAILED. attempt {attempt + 1}/{state['retries_max'] + 1}: {state['url_current']}")
                continue
            try :
                _sanitized_fetch = await state["mcp_manager"].managed_browser.get_sanitized_content()
            except Exception as e :
                _sanitized_fetch = None
                logger.warning(f"FETCH EXCEPTION. {e} on: {state['url_current']}")
            if _sanitized_fetch is None :
                # get_sanitized_content() returns None only on a transport-level failure
                # (session error / page reload failed): never trust it - the
                # next attempt re-navigates.
                state["browser_state"][0] = ""
                logger.warning(f"FETCH FAILED (transport). attempt {attempt + 1}/{state['retries_max'] + 1}: {state['url_current']}")
                continue
            state["browser_state"][0] = _sanitized_fetch
            logger.debug(f"FETCH. attempt {attempt + 1}/{state['retries_max'] + 1}.clean len={len(state['browser_state'][0])} on: {state['url_current']}")
            if state["browser_state"][0] :
                break
        if not state["browser_state"][0] :
            logger.warning(f"EMPTY PAGE after {state['retries_max'] + 1} attempts. Moving on: {state['url_current']}")
            return None # Exhausted refetches

        # --- 2. FIRST-STORY "MORE CONTENT" CHECK (Yahoo-specific, NO LLM) ---
        # is_root is captured here, BEFORE the link-extraction push below (the
        # push appends a level and so changes len(urls)): the entry hub is a
        # structural page, never an article, so the more-content check only
        # runs for deeper article pages.
        is_root = (len(state["urls"]) == 1)
        if (not is_root) and state["browser_state"][0] :
            # The "Story Continues" reveal handler only fires for a TRUSTED
            # click (execute_script cannot produce one), so the hidden
            # continuation is force-revealed instead (see _FIRST_STORY_REVEAL_JS);
            # the sanitized content is re-fetched and used only if it grew.
            try :
                await state["mcp_manager"].managed_browser("execute_script", {"script": _FIRST_STORY_REVEAL_JS})
            except Exception as e :
                logger.warning(f"FIRST-STORY REVEAL transport error (continuing): {e}")
            else :
                await asyncio.sleep(EXPANSION_SETTLE_SECONDS)  # let the layout settle after the reveal
                expanded_content = await state["mcp_manager"].managed_browser.get_sanitized_content()
                if expanded_content is not None and len(expanded_content) > len(state["browser_state"][0]) :
                    logger.info(f"MORE-CONTENT EXPANDED: {state['url_current']}")
                    state["browser_state"][0] = expanded_content

        # --- 3. Extract the page links (traversal only, never fed to the LLM) ---
        # is_root was hoisted into the more-content check step above (it must be
        # captured BEFORE the push below, which appends a level and so changes
        # len(urls)); the entry level is the structural hub (block 4).
        # The push is only possible below deepness_max (deepness_max < 0 = unlimited:
        #  the plain comparison would always be False then); at the depth limit the
        #  polled extraction (up to retries_max * LINKS_POLL_DELAY_SECONDS) is skipped entirely.
        if state["deepness_max"] < 0 or len(state["urls"]) < state["deepness_max"] :
            # get_links() calls the session directly (single round-trip, no pre-flight
            # probe); the proxy injects instance_id and tolerates transport errors as None
            assert state["mcp_manager"].managed_browser is not None , "Managed browser is not available!"
            # The page can still be hydrating right after navigation: poll up
            # to retries_max times until the DOM yields links (attempts are
            # LINKS_POLL_DELAY_SECONDS apart; stops early once any link is
            # found; the last attempt never sleeps).
            extracted_links = []
            for poll in range(1, state["retries_max"] + 1) :
                extracted_links = await state["mcp_manager"].managed_browser.get_links() or []
                logger.debug(f"LINKS POLL {poll}/{state['retries_max']}. {len(extracted_links)} links found on: {state['url_current']}")
                if extracted_links :
                    break
                await asyncio.sleep(LINKS_POLL_DELAY_SECONDS)
            # The URL shortcut: keep only Yahoo Finance news candidates - and
            # push them onto the DFS stack right here; nothing downstream
            # carries links over.
            news_links = [link for link in extracted_links if is_yahoo_news_url(link)]
            if news_links :
                state["urls"].append(news_links)
                logger.info(f"Pushed {len(news_links)} news links to a new depth level (of {len(extracted_links)} extracted): {state['url_current']}")
        else :
            logger.debug(f"DEPTH LIMIT REACHED (depth {len(state['urls'])} >= deepness_max {state['deepness_max']}). Link extraction skipped: {state['url_current']}")

        # --- 4. ENTRY LEVEL: structural hub, exempt from the content verdict ---
        if is_root :
            # Never saved, never marked visited (revisitable, like the generic
            # AGGREGATOR): the entry hub is EXCLUDED from the visited set. Its
            # news links were already pushed as a new DFS level in the
            # extraction block above.
            logger.info(f"Root page (entry): {state['url_current']}")
            state["browser_state"][0] = ""
            return None

        # --- 5. DEEPER LEVELS: binary content verdict STORY | IRRELEVANT ---
        # Prompt structure: the SystemMessage carries ONLY the static
        # instructions (identical for every page -> a fully cacheable prefix
        # for the vLLM prefix cache); the HumanMessage carries the page data
        # (URL + pruned HTML) and repeats the output-format reminder at the
        # very end, right where the verdict is generated.
        classifier_system_prompt = (
            f"# ROLE\n"
            f"You are a senior financial news editor. You are given the cleaned HTML of ONE web page\n"
            f"fetched from Yahoo Finance (the URL contains the '/news/' segment).\n"
            f"Your task is a strict binary content check: does this page carry a finished financial STORY, or not?\n"
            f"You never act on the page - you only issue a verdict.\n\n"
            f"# NOTE ON THE HTML\n"
            f"The HTML is a PRUNED DOM: only h1-h4, p, a, li, and table tags survive the cleaning, so site\n"
            f"chrome, widgets, and the <article> element may be absent. Judge the article body on its own\n"
            f"merits and ignore navigation, related-news lists, and quote widgets when assessing the content.\n\n"
            f"# DECISION RULES\n"
            f"STORY if and only if ALL THREE conditions hold. Evaluate them in this order; the first\n"
            f"condition that fails means IRRELEVANT:\n"
            f"1. **ENOUGH CONTENT** - the page holds a substantive single article: one dominant headline and a\n"
            f"   multi-paragraph narrative body. Not a stub, teaser, listing, or one-liner.\n"
            f"2. **FINANCES** - the article is about finance, markets, or the economy: companies and their numbers\n"
            f"   (earnings, revenue, guidance, margins, deals, M&A, regulation, layoffs), markets (stocks, bonds,\n"
            f"   forex, commodities, crypto, indices), macro (central banks, rates, inflation, jobs data), or\n"
            f"   tickers (AAPL, MSFT, TSLA, GOOGL, ...).\n"
            f"3. **FINISHED, CONSISTENT PLOT** - the article reads as a complete, coherent story: a clear subject,\n"
            f"   a logical flow from lead to close, no cut-off sentences, no jumbled fragments, and the full body\n"
            f"   is present (not replaced by a paywall or sign-in wall).\n"
            f"IRRELEVANT - everything that is NOT a STORY: hubs/feeds of many headlines, indexes, watchlists or\n"
            f"quote pages without an article; non-financial topics (sports-only, celebrity gossip, lifestyle,\n"
            f"politics without a financial angle); boilerplate (login, error/404, legal, cookie, empty pages);\n"
            f"truncated or teaser-only content; pages whose text is fragmented or incoherent.\n\n"
            f"# EDGE CASES\n"
            f"* A short but complete wire brief (AP/Reuters style, 2-5 paragraphs on finance) IS a STORY.\n"
            f"* Embedded tickers, quote tables, or data blocks are STRONG STORY signals when part of a coherent article.\n"
            f"* Mentioning finance somewhere is NOT enough: a single finished article is mandatory.\n\n"
            f"# OUTPUT FORMAT - STRICT\n"
            f"Your reply must consist of EXACTLY these two lines, in this order, with nothing before or after\n"
            f"(no markdown fences, no preamble):\n"
            f"VERDICT: <STORY|IRRELEVANT>\n"
            f"Analysis: [2-3 sentences: the page structure you observed, the financial subject (if any), and the deciding condition(s)]\n"
        )
        classifier_user_prompt = (
            f"# PAGE TO CLASSIFY\n"
            f"URL: {state['url_current']}\n\n"
            f"```html\n{state['browser_state'][0]}\n```\n\n"
            f"# TASK\n"
            f"Classify the page above. Reply with exactly the two lines specified in the output format:\n"
            f"VERDICT: <STORY|IRRELEVANT>\n"
            f"Analysis: ...\n"
        )

        # BACKLOG: Oversized pages are hard-skipped for now (page is discarded and not saved),
        # as in the generic crawler.
        try :
            prompt_tokens = await get_prompt_tokens_length(classifier_system_prompt + classifier_user_prompt)
        except Exception as e : # The token counter is best-effort: on failure, let the LLM call decide.
            prompt_tokens = 0
            logger.warning(f"TOKEN COUNT FAILED (continuing with the LLM call): {e}")
        if prompt_tokens > llm_base_context_len :
            logger.warning(f"OVERSIZED PAGE SKIPPED. url={state['url_current']}, prompt_tokens={prompt_tokens} > context_budget={llm_base_context_len}")
            state["browser_state"][0] = ""
            return None

        # Query the tools-free vLLM generation (with a hard timeout: a hung
        # request must not block the crawl forever). On any failure the page
        # is discarded WITHOUT being marked visited: a failed call is not a
        # verdict (the visited DB is updated only in the verdict branches), so
        # the URL stays eligible for a future run.
        input_messages = [ SystemMessage(content=classifier_system_prompt), HumanMessage(content=classifier_user_prompt) ]
        try :
            response = await asyncio.wait_for(
                llm_base.ainvoke( input_messages,
                                  config={"configurable": {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "high"}}} ),
                                  timeout=300.0 )
        except Exception as e :
            logger.error(f"LLM CLASSIFICATION FAILED (page discarded, NOT marked visited): url={state['url_current']}, error={e}")
            state["browser_state"][0] = ""
            return None

        # Parse the strict verdict; unparseable / missing / non-str content
        # defaults to IRRELEVANT. A list-valued content (langchain multi-part
        # payloads) is stringified first: its str form still carries the
        # VERDICT line, so the regex keeps matching.
        content = response.content
        if not isinstance(content, str) :
            content = str(content)
        try :
            verdict_match = re.search(r'VERDICT:\s*(STORY|IRRELEVANT)', content, re.IGNORECASE) if content else None
            if verdict_match :
                verdict = verdict_match.group(1).upper()
            else                :
                verdict = "IRRELEVANT"
                logger.warning(f"UNPARSEABLE VERDICT (defaulted to IRRELEVANT): {state['url_current']}. Response tail: {(response.content or '')[-300:]!r}")
        except Exception as e :
            verdict_match = None
            logger.warning(f"VERDICT PARSE FAILED (defaulted to IRRELEVANT): {state['url_current']}, error={e}")

        if verdict == "STORY" :  # STORY  --  save it at hdd
            save_dir = os.path.join(state["html_folder_name"], state["date_str"])
            os.makedirs(save_dir, exist_ok=True)
            # Generate a unique filename - keep trying until file creation succeeds
            compressed_browser_state = gzip.compress(state["browser_state"][0].encode("utf-8"))
            while True:
                unique_fname = str(uuid.uuid4())
                file_path = os.path.join(save_dir, unique_fname + ".html.gz")
                if os.path.exists(file_path):
                    continue  # File exists, try another name
                try :
                    # Attempt to create the file exclusively ('xb' is required for gzip bytes)
                    with open(file_path, 'xb') as f:
                        f.write(compressed_browser_state)
                    break  # File created successfully!
                except FileExistsError :
                    continue  # Another process created it, try another name
            rabbit_res = await state["rabbit_manager"].publish_json({"url": state["url_current"],
                                                                     "file_path": file_path,
                                                                     "timestamp": state["date_str"],})
            if not rabbit_res :
                # CRITICAL: the story file is on disk but was never delivered.
                # The page is neither counted nor marked visited (the URL stays
                # eligible for a future run) and the driver loop aborts the
                # crawl with error code 2.
                logger.error(f"PUBLISH FAILED - CRITICAL. Aborting crawl: url={state['url_current']}, file={file_path}")
                state["browser_state"][0] = ""
                raise CrawlerPublishError(f"publish_json failed for url={state['url_current']}, file={file_path}")
            logger.info(f"STORY page. Published to RabbitMQ (publish_status={rabbit_res}): url={state['url_current']}, file={file_path}")
            state["pages_saved"][0] += 1
            # Store visited URL in RedisDB
            redis_res = await state["redis_manager"].check_and_add(state["url_current"])
            logger.info(f"STORY page. Marked as visited in Redis (add_status={redis_res}): {state['url_current']}")
        else                       :  # IRRELEVANT  --  discard the page, but still mark it as visited
            redis_res = await state["redis_manager"].check_and_add(state["url_current"])
            logger.info(f"IRRELEVANT page. Marked as visited in Redis (add_status={redis_res}): {state['url_current']}")
        state["browser_state"][0] = ""
        return None

    # --- Graph Assembly ---
    # The Yahoo crawl runs a SINGLE node per page cycle: the entry hub and the
    # story/irrelevant verdict are both handled by page_classify_agent.
    #
    #    EXTERNAL DRIVER LOOP (owns the DFS stack traversal) :
    #           url_current = urls[0][0]  (the entry hub)
    #           .------------------------.
    #           V                        |
    #    START -+-> page_classify_agent -o-> END
    #           (fetch / classify url_current; push new links in place)
    #           pages_max check -> traverse the stack for the next url_current
    #
    # The graph is compiled ONCE; the external loop seeds url_current with
    # the entry hub, re-invokes it per page cycle (no state re-assignment),
    # breaks on pages_max or a drained stack, aborts on a critical failed
    # story publish (CrawlerPublishError -> error code 2), and asserts the
    # clean-up invariant (empty browser_state) after each cycle.
    # State is initialized with urls=[[entry_url]].
    builder = StateGraph(YahooAgenticState)
    builder.add_node("page_classify_agent", page_classify_agent)
    builder.add_edge(START, "page_classify_agent")
    builder.add_edge("page_classify_agent", END)

    # Compile ONCE outside the loop
    graph = builder.compile()

    state = YahooAgenticState({ "date_str"         : datetime.datetime.now().strftime('%Y-%m-%d'),
                                 "urls"             : [[url_string,],],
                                 "url_current"      : "",
                                 "deepness_max"     : deepness_max,
                                "pages_saved"      : [0,],
                                "retries_max"      : retries_max,
                                "browser_state"    : ["",],
                                "html_folder_name" : html_folder_name,
                                "mcp_manager"      : mcp_manager,
                                "rabbit_manager"   : rabbit_manager,
                                "redis_manager"    : redis_manager,
                             })

    # External driver loop: owns the DFS stack traversal and the pages_max
    # limit; it reuses the compiled 'graph' across all page cycles. The graph
    # node processes url_current (a plain str the node never writes) and
    # pushes newly found links straight onto the stack.
    # Deliberately NO state re-assignment after ainvoke: the node mutates the
    # shared containers in place (urls / pages_saved / browser_state - the
    # channel values are those same objects) and url_current is
    # driver-managed, so the original state dict stays authoritative.
    #
    # url_current is the loop condition: "" means the DFS stack is drained
    # (or the pages limit was hit) and the crawl ends. The first page is
    # seeded directly from the entry level - urls[0][0] is the entry hub,
    # which holds exactly one URL that is always visited, so no traversal
    # (and no visited check) is needed to pop it.
    state["url_current"] = state["urls"][0].pop()
    error_code = 0
    while state["url_current"] :
        if ( pages_max > 0 and state["pages_saved"][0] >= pages_max ) : break  # break on pages limit
        logger.info(f"NEW URL. Navigating to: {state['url_current']}")
        try :
            await graph.ainvoke(state)
        except CrawlerPublishError as e :
            # A critical failed story publish: the node already cleared the
            # state and left the URL unvisited (eligible for a future run);
            # stop the crawl here.
            logger.error(f"CRITICAL PUBLISH FAILURE. Crawl aborted: {e}")
            error_code = 2
            break
        assert not len(state["browser_state"][0]) , "Failure in agentic clean-up."
        # DFS stack traversal: prune empty tail levels, then pop the next
        # URL into url_current. Deeper URLs already visited in Redis are
        # skipped (read-only check_only - the visited DB is updated only in
        # the node's verdict branches); the entry level (len(urls) == 1) is
        # a structural hub and is always visited.
        state["url_current"] = ""  # Stack exhausted: end the crawl
        while state["urls"] and state["url_current"] == "" :
            if not state["urls"][-1] :
                state["urls"].pop()  # Remove empty tail levels
            else                     :
                url = state["urls"][-1].pop()
                if await state["redis_manager"].check_only(url) != 1 :
                   state["url_current"] = url
                else                                                 :
                    logger.info(f"SKIPPED URL. Already visited: {url}")

    logger.info(f"Saved pages:    {state['pages_saved'][0]}")
    logger.info("======================================================")

    return error_code


async def run_stealth_graph_yahoo(url_string : str = DEFAULT_ENTRY_URL, deepness_max : int = 3, pages_max :int = -1, retries_max : int = 3,
                                  html_folder_name : str = "./html/", rabbit_host : str = "localhost", rabbit_port : int = 5672,
                                  redis_host : str = "localhost", redis_port : int = 6379) :
    """
    Main orchestration function: wires all subsystems and runs the crawler.
    Specialized clone of run_stealth_graph (crawler_agent.py) - the LLM is
    tools-free (no bind_tools, no tools-token accounting) because page access
    is script-controlled in this crawler.

    Args:
        url_string (str, default=https://ca.finance.yahoo.com/): Entry URL.
        deepness_max (int, default=3): Maximum tree depth for exploration (the
            entry hub counts as level 1, so the default crawls the hub and two
            levels of /news/ pages; -1 = unlimited).
        pages_max (int, default=-1): Maximum stories to save (-1 = unlimited).
        retries_max (int, default=3): Maximum refetch attempts for empty page content.
        html_folder_name (str, default=./html/): Root folder for the archives.
        rabbit_host (str, default=localhost): RabbitMQ server hostname.
        rabbit_port (int, default=5672): RabbitMQ AMQP port.
        redis_host (str, default=localhost): Redis server hostname.
        redis_port (int, default=6379): Redis server port.

    DEBUG MODE:
        Set the DEBUG environment variable to 'true', '1', or 'yes' (case-insensitive).
        The flag is re-read at runtime (it may be set after the module import) and
        then passed explicitly as readonly=True to the RabbitMQ and Redis clients:
        the crawler runs end-to-end, but no data is pushed into the real
        redis/rabbit pipelines (writes are buffered in memory and lost on exit),
        so debug runs are repeatable and non-destructive.

    Returns:
        int: Return code (0 = success, 1 = MCP/RabbitMQ/Redis connection failure,
             -1 = LLM failure, 2 = critical story-publish failure - crawl aborted)

    Note:
        The function uses async context managers for automatic cleanup of resources.
        The MCP session is created within the manager context, and all connections
        are properly closed on exit or error.
    """
    # Evaluate DEBUG at runtime (the env var may have been set after import,
    # e.g. by a test harness) and pass it explicitly to the back-end clients.
    debug_mode = _debug_mode()
    if debug_mode :
        logger.info("DEBUG MODE. RabbitMQ/Redis clients run readonly: no data is pushed into the real pipelines.")

    # Start MCP server
    async with StealthMCPManager(logger) as mcp_manager :
        mcp_err_code = await mcp_manager.connect()
        if mcp_err_code :
            logger.error(f"Failed to connect MCP: {mcp_manager.ERROR_MCP_CODES_WORDS[mcp_err_code]}")
            return 1
        # Use create_managed_browser instead of the property managed_browser
        mcp_err_code = await mcp_manager.create_managed_browser(headless=False)
        if mcp_err_code :
            logger.error(f"Failed to create a managed browser: {mcp_manager.ERROR_MCP_CODES_WORDS[mcp_err_code]}")
            return 1

        # Plug-in the tools-free LLM (the classifier is the only LLM user)
        llm_base = ChatOpenAI( base_url=f"{BASE_URL}/v1",
                               api_key=BASE_API_KEY,
                               model_name=MODEL_NAME,
                               max_tokens=LLM_MAX_OUTPUT_TOKENS,
                               temperature=0.1,)
        # Test LLM connectivity with a simple ping. A completed ainvoke proves
        # the endpoint is reachable even if the visible content is empty (a
        # thinking-first model may answer in reasoning_content only); the
        # model keeps its default thinking behavior for the ping.
        try : # Request a minimal completion with a strict timeout
            response = await asyncio.wait_for(
                llm_base.ainvoke("ping"),
                timeout=30.0  # Define timeout value
            )
            assert response is not None , "Failed to connect LLM!"
        except Exception as e :
            logger.error(f"LLM Failed with error code {e}")
            return -1

        # Link RabbitMQ (readonly mode in DEBUG: buffered writes, the real
        # queue is never touched)
        async with RabbitMQClient(host=rabbit_host, port=rabbit_port,
                                  publish_queue_name="crawler_json_queue", receive_queue_name=None,
                                  username="crawler", password="crawler",
                                  readonly=debug_mode) as rabbit_manager:
            rabbit_err_code = await rabbit_manager.connect()
            if rabbit_err_code:
                logger.error(f"Failed to connect RabbitMQ at host {rabbit_manager.host} port {rabbit_manager.port}")
                return 1

            # Connect to RedisDB (readonly mode in DEBUG: the visited set is
            # buffered in memory, the real Redis set is never touched)
            async with RedisDBClient(host=redis_host, port=redis_port, db=0, readonly=debug_mode) as redis_manager:
                redis_err_code = await redis_manager.connect()
                if redis_err_code:
                    logger.error(f"Failed to connect RedisDB at host {redis_manager.host} port {redis_manager.port}")
                    return 1

                # Launch the crawler
                agentic_error_code = await run_agentic_crawler_yahoo(url_string=url_string, mcp_manager=mcp_manager,
                                                                     llm_base_context_len=(LLM_BASE_CONTEXT_LEN - LLM_MAX_OUTPUT_TOKENS - 1024 * 4),
                                                                     llm_base=llm_base,
                                                                     deepness_max=deepness_max, pages_max=pages_max, retries_max=retries_max,
                                                                     html_folder_name=html_folder_name, rabbit_manager=rabbit_manager, redis_manager=redis_manager)

    return agentic_error_code


# python3 ./src/crawler/crawler_agent_yahoo.py -u "https://ca.finance.yahoo.com/" -d 3 -f "./html" -n -1 -m 1
# python3 ./src/crawler/crawler_agent_yahoo.py -u "https://ca.finance.yahoo.com/topic/news/" -d 3 -f "./html" -n -1 -m 1
if __name__ == "__main__":
    """
    Main entry point for command-line execution (mirror of crawler_agent.py
    __main__). Parses the flags below and runs run_stealth_graph_yahoo via
    asyncio.run.
    Flags:
        -u/--url_string        (str, default=https://ca.finance.yahoo.com/) Entry point url.
        -d/--deepness_max      (int, default=3)  Deepness (in hops) of exploration.
        -f/--html_folder_name  (str, default=./html/) Folder for the downloaded html files.
        -n/--pages_max         (int, default=-1) Maximal number of stories to extract.
        -m/--retries_max       (int, default=3)  Maximal number of refetch attempts for empty page content.
        -bh/--rabbit_host      (str, default=localhost) RabbitMQ host.
        -bp/--rabbit_port      (int, default=5672) RabbitMQ AMQP port.
        -rh/--redis_host       (str, default=localhost) RedisDB host.
        -rp/--redis_port       (int, default=6379) RedisDB port.
    """
    import argparse
    parser = argparse.ArgumentParser(description="The Yahoo Finance news crawler.")

    parser.add_argument("--url_string",       "-u", type=str, default=DEFAULT_ENTRY_URL, dest="url_string",        help="Entry point url.")
    parser.add_argument("--deepness_max",     "-d", type=int, default=3,                               dest="deepness_max",      help="Deepness (in hops) of exploration.")
    parser.add_argument("--html_folder_name", "-f", type=str, default="./html/",                       dest="html_folder_name",  help="Name of the folder for storing downloaded html files.")
    parser.add_argument("--pages_max",        "-n", type=int, default=-1,                              dest="pages_max",         help="Maximal number of stories to extract.")
    parser.add_argument("--retries_max",      "-m", type=int, default=1,                               dest="retries_max",       help="Maximal number of refetch attempts for empty page content.")
    parser.add_argument("--rabbit_host",     "-bh", type=str, default="localhost",                     dest="rabbit_host",       help="RabbitMQ host.")
    parser.add_argument("--rabbit_port",     "-bp", type=int, default=5672,                            dest="rabbit_port",       help="RabbitMQ AMQP port.")
    parser.add_argument("--redis_host",      "-rh", type=str, default="localhost",                     dest="redis_host",        help="RedisDB host.")
    parser.add_argument("--redis_port",      "-rp", type=int, default=6379,                            dest="redis_port",        help="RedisDB port.")
    args = parser.parse_args()

    return_code = asyncio.run(run_stealth_graph_yahoo(
        url_string=args.url_string,
        deepness_max=args.deepness_max,
        pages_max=args.pages_max,
        retries_max=args.retries_max,
        html_folder_name=args.html_folder_name,
        rabbit_host=args.rabbit_host,
        rabbit_port=args.rabbit_port,
        redis_host=args.redis_host,
        redis_port=args.redis_port))
    logger.info(f"Agent finished with {return_code} error code.")

# Backlog:
# 1. Replace the hard-skip of oversized pages with proper handling (chunking / DOM truncation / salvage)
# NOTE: deepness_max == -1 ("unlimited") is supported now via the depth_ok guard;
#       the generic crawler (crawler_agent.py) still has the old len(urls) < deepness_max semantics.
