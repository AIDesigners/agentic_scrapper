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

# Make the parent 'src' directory importable so that modules moved there
# (rabbit_driver, redis_driver) can be imported as top-level modules.
_SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC_ROOT)
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, RemoveMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import ClientSession
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition


from web_tools import clean_dom

# Optional site-specific apriori traversal filters (boosters), selected by the
# entry URL. See crawler_site_filters.py (same dir) for the registry + boosters.
from crawler_site_filters import SiteFilter, select_site_filter

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


# --- DEBUG FLAG: Use mock implementations for testing/debugging ---
# Set the DEBUG environment variable to 'true', '1', or 'yes' (case-insensitive)
# to use mock implementations instead of real RabbitMQ and Redis clients.
# This is useful for unit testing and debugging without requiring actual
# RabbitMQ and Redis servers.
DEBUG = os.environ.get('DEBUG', '').lower() in ('true', '1', 'yes')

if DEBUG:
    logger.debug("INFO. Running in DEBUG mode.")

# Import the unified clients with mock support
# These modules were moved to the src/ directory root; they are importable
# thanks to the sys.path insertion above.
from rabbit_driver import RabbitMQClient
from redis_driver import RedisDBClient

# Define structural protocols to avoid importing from the raw 'mcp' package
class MCPContentBlock(Protocol):
    text: str
class MCPToolResult(Protocol):
    content: List[MCPContentBlock]
class MCPSession(Protocol):
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> MCPToolResult : ...

# The proxy browser class under controll of MCP
class BrowserProxy :
    def __init__(self, logger : logging.Logger, mcp_session : ClientSession, browser_instance_id : str) -> None :
        self.logger = logger
        self.mcp_session = mcp_session
        self.browser_instance_id = browser_instance_id

    # Implementing __call__ allows you to do: await browser("tool_name", {...})
    async def __call__(self, tool_name: str, args: Dict[str, Any] | None = None) -> Optional[MCPToolResult] :
        assert self.browser_instance_id is not None , "Managed browser is not initialized!"
        # 1. PRE-CALL HEALTH TEST
        try:
            result = await self.mcp_session.call_tool("execute_script",
                                                      {"instance_id": self.browser_instance_id, "script": "1"})
            assert not getattr(result, "isError", False)
        except Exception:
            # Captures library-level connection, read, or stream timeouts cleanly
            return None
        # 2. TIMEOUT PROTECTED TOOL EXECUTION
        full_args = args or {}
        full_args["instance_id"] = self.browser_instance_id
        return await self.mcp_session.call_tool(tool_name, full_args)

    # The routine to open page
    async def navigate(self, url_string : str) -> Optional[MCPToolResult] :
        mcp_response = await self("navigate", {"url" : url_string, "wait_until" : "networkidle"})
        return mcp_response if mcp_response and not getattr(mcp_response, "isError", False) else None

    # Accessing the page content
    async def get_content(self, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            mcp_response = await self("get_page_content")
            if mcp_response is None : return None
            # --- 1. IF JS FAILED / PAGE CRASHED ---
            if getattr(mcp_response, "isError", False) :
                self.logger.warning(f"JS/Page error on attempt {attempt + 1} of {max_retries}.")
                if attempt < max_retries - 1:
                    self.logger.warning("Triggering native page reload...")
                    mcp_response = await self("reload_page")
                    if not mcp_response : return None
                continue
            # --- 2. IF SUCCESSFUL ---
            if mcp_response and getattr(mcp_response, "content", None):
                raw_content = mcp_response.content[0].text.strip()
                if raw_content.startswith("{") and raw_content.endswith("}"):
                    try:
                        meta = json.loads(raw_content)
                        if isinstance(meta, dict) and "file_path" in meta:
                            file_path = meta["file_path"]
                            if os.path.exists(file_path):
                                with open(file_path, "r", encoding="utf-8") as f:
                                    return f.read()
                            else:
                                self.logger.error(f"Offloaded file not found at {file_path}")
                                return ""
                    except json.JSONDecodeError:
                        pass
                return raw_content
        # --- 3. IF ALL RETRIES FAIL ---
        self.logger.error(f"Failed to extract content after {max_retries} attempts. Returning empty page.")
        return ""

    # The routine to collect all liks on a page
    async def get_links(self) -> Optional[Set[str]] :
        js_script = """
        (() => {
            const links = Array.from(document.querySelectorAll('a[href]'));
            const webPageRegex = /\\.(png|jpe?g|gif|svg|webp|pdf|zip|tar|gz|mp3|mp4|css|js)$/i;
            return links
                .map(a => a.href)
                .filter(href => {
                    try {
                        const url = new URL(href);
                        // Ensure it's a web protocol and doesn't match static asset extensions
                        return (url.protocol === 'http:' || url.protocol === 'https:') 
                               && !webPageRegex.test(url.pathname);
                    } catch (e) {
                        return false;
                    }
                });
        })()
        """
        # JavaScript snippet to filter for actual web pages, ignoring typical asset files
        mcp_response = await self.mcp_session.call_tool("execute_script", {
            "instance_id": self.browser_instance_id,
            "script": js_script
        })
        if mcp_response and not getattr(mcp_response, "isError", False) :
            data = json.loads(mcp_response.content[0].text)
            return {item.get('value') for item in data.get('result', []) if item.get('value')}
        else                                     :
            return None


from contextlib import AsyncExitStack
class StealthMCPManager:
    """
    Manages the complete lifecycle of an MCP (Model Context Protocol) server and browser.
    This class provides a robust interface for connecting to, controlling, and disconnecting
    from a stealth browser MCP server. It handles all error cases gracefully by returning
    error codes instead of raising exceptions, making it suitable for production use.
    The manager follows a strict hierarchy:
        Client -> Session -> Browser Instance -> Tools
    Attributes:
        ERROR_MCP_CODES_WORDS (dict): Mapping of error codes to human-readable strings.
        ERROR_MCP_CODES_NUMS (dict): Reverse mapping for checking if an error code is valid.
        folder (str): Base directory for the stealth-browser-mcp installation.
        _server_config (dict): Configuration for MCP server connection.
        _mcp_client: The MCP client instance for managing server connections.
        _mcp_session: The active MCP session for tool invocation.
        managed_browser (str): Instance ID of the currently managed browser.
        _tools (list): List of loaded MCP tools available for browser automation.
        _tools_summary (str): Human-readable summary of available tools.
    Error Codes:
        0: ERROR_MCP_OK - Success / no error
        1: ERROR_MCP_ALREADY_CONNECTED - Client already exists, must close first
        2: ERROR_MCP_NOT_CONNECTED - Session is None, need to connect first
        3: ERROR_MCP_CLIENT_CREATE - Failed to instantiate MultiServerMCPClient
        4: ERROR_MCP_SESSION_START - Failed to start MCP session
        5: ERROR_MCP_LOAD_TOOLS - Failed to load tools from session
        6: ERROR_MCP_SESSION_STOP - Failed to stop session
        7: ERROR_MCP_CLIENT_STOP - Failed to stop client
        8: ERROR_MCP_BROWSER_STOP - Failed to close browser
        9: ERROR_MCP_BROWSER_SPAWN - Failed to spawn browser
        10: ERROR_MCP_TOOL_CALL - Failed to call an MCP tool
    Example:
        async with StealthMCPManager() as manager:
            await manager.connect()
            await manager.create_managed_browser(headless=True)
            # ... use browser ...
            await manager.disconnect()
    """
    ERROR_MCP_CODES_WORDS = { 0: "ERROR_MCP_OK",
                              1: "ERROR_MCP_ALREADY_CONNECTED",
                              2: "ERROR_MCP_NOT_CONNECTED",
                              3: "ERROR_MCP_CLIENT_CREATE",
                              4: "ERROR_MCP_SESSION_START",
                              5: "ERROR_MCP_LOAD_TOOLS",
                              6: "ERROR_MCP_SESSION_STOP",
                              7: "ERROR_MCP_CLIENT_STOP",
                              8: "ERROR_MCP_BROWSER_STOP",
                              9: "ERROR_MCP_BROWSER_SPAWN",
                             10: "ERROR_MCP_BROWSER_NOT_EXISTS",
                             11: "ERROR_MCP_TOOL_CALL",
    }
    ERROR_MCP_CODES_NUMS = {value: key for (key, value) in ERROR_MCP_CODES_WORDS.items()}

    def __init__(self, logger: Optional[logging.Logger], folder: str = "/home/ayakovenko/nila/stealth-browser-mcp/") -> None:
        """
        Initialize the StealthMCPManager with default configuration.
        Sets up the server configuration for connecting to the stealth-browser MCP server.
        The configuration includes environment variables for Chrome/Chromium paths and
        display settings for headless operation.
        Args:
            folder (str): Path to the stealth-browser-mcp installation directory.
                Defaults to "/home/ayakovenko/nila/stealth-browser-mcp/".
        Note:
            The server_config dictionary is hardcoded for the specific stealth-browser
            setup. It includes:
            - transport: "stdio" for standard input/output communication
            - command: Python interpreter path
            - args: Server script path
            - env: Environment variables for display and browser executable paths
        """
        self.folder = folder
        self.logger = logger
        self._server_config = {
            "stealth-browser": {
                "transport": "stdio",
                "command": os.path.join(folder, "./bin/python3"),
                "args": [os.path.join(folder, "./src/server.py")],
                "env": {
                    "DISPLAY": ":1",
                    "CHROME_PATH": "/home/ayakovenko/nila/flatpak-chromium.bash",
                    "BROWSER_EXECUTABLE_PATH": "/home/ayakovenko/nila/flatpak-chromium.bash",
                    "PUPPETEER_EXECUTABLE_PATH": "/home/ayakovenko/nila/flatpak-chromium.bash"
                }
            }
        }
        # Type hints assumed from external imports like MultiServerMCPClient, MCPSession, BrowserProxy
        self._mcp_client = None
        self._mcp_session = None
        self._managed_browser = None
        self._tools: List = []
        self._tools_summary: str = ""
        # AsyncExitStack manages async context managers safely without breaking AnyIO tasks
        self._exit_stack = None
    @property
    def session(self):
        return self._mcp_session

    @property
    def managed_browser(self):
        return self._managed_browser

    @property
    def tools(self):
        return self._tools

    @property
    def tools_summary(self):
        return self._tools_summary

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()

    async def _close_managed_browser(self):
        """Internal helper to close the browser instance safely."""
        if self._mcp_session and self._managed_browser:
            try:
                await asyncio.wait_for(self.mcp_session.call_tool("close_instance", {"instance_id" : self.managed_browser.browser_instance_id }), timeout=15.0)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error while closing managed browser: {e}")
            finally:
                self._managed_browser = None

    async def disconnect(self):
        """
        Gracefully disconnect and clean up all resources.
        This method reverses the connection process by closing resources in the correct order:
        1. Closes the managed browser if one exists
        2. Exits the MCP session and client seamlessly using AsyncExitStack
        3. Clears the tools list and summary
        Returns:
            None
        Side Effects:
            - Closes active browser instance if managed_browser is not empty
            - Sets self._mcp_session to None
            - Sets self._mcp_client to None
            - Clears self._tools and self._tools_summary
        """
        # 1. Manually close the browser via the active session first
        await self._close_managed_browser()
        # 2. Use the AsyncExitStack to properly close AnyIO scopes and contexts 
        # (This replaces the manual __aexit__ calls which caused RuntimeErrors)
        if hasattr(self, '_exit_stack') and self._exit_stack :
            await self._exit_stack.aclose()
            self._exit_stack = None
        # 3. Clean up internal state
        self._mcp_session = None
        self._mcp_client = None
        self._tools = []
        self._tools_summary = ""

    async def connect(self) -> int:
        """
        Establish connection to the MCP server and load available tools.
        This method performs the complete connection sequence:
        1. Creates the MultiServerMCPClient with the server configuration
        2. Establishes an asynchronous session with the stealth-browser server
        3. Loads all available tools for browser automation
        Returns:
            int: Error code from ERROR_MCP_CODES_NUMS:
                - ERROR_MCP_OK (0): Connection successful, tools loaded
                - ERROR_MCP_ALREADY_CONNECTED (1): Client already exists
                - ERROR_MCP_CLIENT_CREATE (3): Failed to create client
                - ERROR_MCP_SESSION_START (4): Failed to start session
                - ERROR_MCP_LOAD_TOOLS (5): Failed to load tools
        """
        if self._mcp_session is not None or self._exit_stack is not None :
            return self.ERROR_MCP_CODES_NUMS["ERROR_MCP_ALREADY_CONNECTED"]

        # Step 1: Create Client
        try:
            # Recreate the stack so the object can be connect()'d again
            self._exit_stack = AsyncExitStack()
            self._mcp_client = MultiServerMCPClient(self._server_config)
            # Note: If MultiServerMCPClient is itself an async context manager, you would enter it here:
            # self._mcp_client = await self._exit_stack.enter_async_context(self._mcp_client)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to instantiate MultiServerMCPClient(server_config): {e}")
            return self.ERROR_MCP_CODES_NUMS["ERROR_MCP_CLIENT_CREATE"]
        # Step 2: Start Session
        try:
            # Safely bind the session's context manager to our class lifecycle
            session_cm = self._mcp_client.session("stealth-browser")
            self._mcp_session = await self._exit_stack.enter_async_context(session_cm)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to start MCP session: {e}")
            await self.disconnect()
            return self.ERROR_MCP_CODES_NUMS["ERROR_MCP_SESSION_START"]
        # Step 3: Load Tools
        try:
            # Assuming load_mcp_tools is imported globally in your file
            self._tools = await load_mcp_tools(self._mcp_session)
            self._tools_summary = "\n".join([f"* Tool: {t.name}\nDescription: {t.description}" for t in self._tools]) + "\n"
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to load tools: {e}")
            await self.disconnect()
            return self.ERROR_MCP_CODES_NUMS["ERROR_MCP_LOAD_TOOLS"]
        return self.ERROR_MCP_CODES_NUMS["ERROR_MCP_OK"]

    async def restart(self) -> int:
        """
        Restart the MCP connection by disconnecting and reconnecting.
        This useful for recovering from errors or when wanting to refresh
        the browser instance. It performs a complete teardown and rebuild
        of the MCP connection stack.
        Returns:
            int: Error code from connect() method.
        """
        await self.disconnect()
        return await self.connect()

    async def create_managed_browser(self, headless=False) -> int:
        """
        Spawn a new managed browser instance through the MCP server.
        Creates a new stealth browser instance that can be controlled via
        MCP tools. The browser runs in a headless environment but can be
        configured for visualization during debugging.
        Args:
            headless (bool, default=False): If True, browser runs without GUI.
                Set to False for debugging/visualization, True for production.
        Returns:
            int: Error code from ERROR_MCP_CODES_NUMS:
                - ERROR_MCP_OK (0): Browser spawned successfully
                - ERROR_MCP_SESSION_STOP (2): No active session
                - ERROR_MCP_BROWSER_STOP (8): Browser already exists
                - ERROR_MCP_BROWSER_SPAWN (9): Failed to spawn browser
        """
        if self._mcp_session is None:
            return self.ERROR_MCP_CODES_NUMS["ERROR_MCP_SESSION_STOP"]
        if self.managed_browser is not None:
            return self.ERROR_MCP_CODES_NUMS["ERROR_MCP_BROWSER_STOP"]
        try:
            result = await self._mcp_session.call_tool("spawn_browser", {"headless": headless})
            data = json.loads(result.content[0].text if result.content and len(result.content) > 0 else str(result))
            self._managed_browser = BrowserProxy(logger=self.logger, mcp_session=self._mcp_session, browser_instance_id=data.get("instance_id"))
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to launch a managed browser: {e}")
            return self.ERROR_MCP_CODES_NUMS["ERROR_MCP_BROWSER_SPAWN"]
        return self.ERROR_MCP_CODES_NUMS["ERROR_MCP_OK"]

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> int:
        """
        Generic wrapper for calling MCP tools through the session.
        Provides a simplified interface for invoking MCP tools with proper
        error handling and session validation.
        Args:
            tool_name (str): Name of the MCP tool to invoke (e.g., "click", "type", "navigate").
            arguments (Dict[str, Any]): Arguments to pass to the tool. Structure depends on
                the specific tool being called.
        Returns:
            int: Error code from ERROR_MCP_CODES_NUMS:
                - ERROR_MCP_OK (0): Tool call successful
                - ERROR_MCP_SESSION_START (4): No active session
                - ERROR_MCP_TOOL_CALL (10): Failed to execute tool
        """
        if self._managed_browser is None:
            return self.ERROR_MCP_CODES_NUMS["ERROR_MCP_BROWSER_NOT_EXISTS"]
        try:
            # Assumes BrowserProxy is implemented dynamically with __call__ 
            result = await self._managed_browser(tool_name, arguments=arguments)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to call {tool_name} mcp tool. Error: {e}")
            return self.ERROR_MCP_CODES_NUMS["ERROR_MCP_TOOL_CALL"]
        return self.ERROR_MCP_CODES_NUMS["ERROR_MCP_OK"]
    

#   --------   A G E N T I C   P A R T   -------- 

# This routine extract amount of tokens 'locked' in tools
async def get_tools_tokens_length(tools_str: str) -> int :
    headers = {"Authorization": f"Bearer {BASE_API_KEY}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(base_url=BASE_URL, headers=headers) as client :
        response = await client.post( "/tokenize", json={ "model": MODEL_NAME, "prompt": tools_str, } )
        response.raise_for_status()
    return response.json().get("count", len(response.json().get("tokens", [])))
# This routine truncates the prompt to fit into a model
async def truncate_prompt(prompt: str, max_tokens_number: int) -> str :
    headers={ "Authorization": f"Bearer {BASE_API_KEY}", "Content-Type": "application/json" }
    async with httpx.AsyncClient(base_url=BASE_URL, headers=headers, timeout=30.0) as client:
        response = await client.post("/tokenize", json={ "model": MODEL_NAME, "prompt": prompt, })
        response.raise_for_status()
        tokens = response.json().get("tokens", [])
        if response.json().get("count", len(tokens)) > max_tokens_number :
            response = await client.post("/detokenize", json={"model": MODEL_NAME, "tokens": tokens[: max_tokens_number]})
            response.raise_for_status()
            return response.json().get("text", "")
        else :
            return prompt      


class AgenticState(TypedDict):
    """
    TypedDict defining the complete state structure for the agentic crawler.
    This state is used by LangGraph to track the crawler's progress, history,
    and resources. All fields are mutable and can be updated during exploration.
    Attributes:
        date_str (str): The date when the crawler started
        urls (List[List[str]]): Nested stack of URLs for traversal. Each inner list
            represents a level in the exploration tree. URLs are popped as visited.
        url_current (str): Current active url after it is popped from urls nested list
        deepness_max (int): Maximum depth allowed for tree exploration.
        pages_saved (int): Number of saved stories at current depth.
        pages_max (int): Maximum pages to visit (-1 means unlimited).
        retries_max (int): Maximum retry attempts for failed page access.
        retry_counter (int): Internal counter for tracking retry attempts on current page.
        browser_state (str): Current browser page content.
        page_history (List[str]): Stack of action plans executed on current page.
        html_folder_name (str): Folder to strore the downloaded html files.
        mcp_manager (StealthMCPManager): Handler for MCP/MCP tools integration.
        rabbit_manager (RabbitMQClient): Handler for message queue operations.
        redis_manager (RedisDBClient): Handler for visited URL tracking.
        site_filter (SiteFilter): Site-specific apriori traversal filter
            (booster) selected from the entry URL; the no-op default for
            unregistered sites. Rejects non-story branches before the browser
            enters them. The zero-level (entry) URL itself is exempt from the
            filter: only URLs popped from the recursion lists are checked.
    """
    date_str         : str
    urls             : List[List[str]]
    url_current      : List[str]
    deepness_max     : int 
    pages_saved      : List[int] 
    retries_max      : int 
    retry_counter    : List[int]
    browser_state    : List[str]
    page_history     : List[str] 
    html_folder_name : str
    mcp_manager      : StealthMCPManager  
    rabbit_manager   : RabbitMQClient     
    redis_manager    : RedisDBClient
    site_filter      : SiteFilter


# The agentic system
async def run_agentic_crawler(url_string : str, mcp_manager : StealthMCPManager,
                              llm_base_context_len : int, llm_base : ChatOpenAI, llm_base_with_tools : ChatOpenAI,
                              deepness_max : int, pages_max : int, retries_max : int,
                              html_folder_name : str, rabbit_manager : RabbitMQClient, redis_manager : RedisDBClient) -> int :
    """
    Main entry point for the agentic web crawler system.
    This function orchestrates a tree-search exploration of web pages using LLM-based agents
    and MCP-controlled browsers. It sets up and executes the LangGraph state machine that
    coordinates multiple specialized agents for crawling, classification, and data extraction.
    Args:
        url_string (str): The entry point URL for the crawler to begin exploration from.
        mcp_manager (StealthMCPManager): Manager for MCP server and browser lifecycle.
            Handles browser spawning, tool loading, and connection management.
        llm_base_context_len (int): Token limit for LLM context window (used for truncation).
        llm_base (ChatOpenAI): Base LLM instance for general-purpose tasks.
        llm_base_with_tools (ChatOpenAI): LLM instance bound with available MCP tools for
            browser automation (navigation, clicks, form filling, etc.).
        rabbit_manager (RabbitMQClient): Message queue client for publishing crawled data.
            Used to send saved stories and metadata to downstream processors.
        redis_manager (RedisDBClient): Database client for tracking visited URLs.
            Uses a set to prevent revisiting the same URL during traversal.
        deepness_max (int, default=2): Maximum depth for tree exploration. Controls how many
            "hops" away from the entry URL the crawler will explore. -1 means unlimited.
        pages_max (int, default=-1): Maximum number of pages to visit. -1 means unlimited.
    Returns:
        int: Error code (0 = success, non-zero = error occurred)
    Internal Components:
        - access_agent: LLM planning agent for determining page access actions
        - page_classify_agent: Planned agent for page categorization (currently unimplemented)
        - LangGraph state machine: Coordinates agent execution flow
    Raises:
        AssertionError: If state becomes desynchronized or browser management fails.
    """

    # Select the site-specific booster (apriori filter) based on the name of
    # the entry URL. Registered sites get a concrete filter that rejects
    # non-story branches (e.g. Yahoo Finance /quote/<TICKER> aggregators)
    # BEFORE the browser enters them, which speeds up the traversal.
    # Unregistered sites get the no-op default filter.
    site_filter = select_site_filter(url_string)
    logger.info(f"SITE FILTER SELECTED. Entry URL={url_string} -> {type(site_filter).__name__}")

    # ----------------------------   A G E N T S   ----------------------------

    # This agent attempt to access web page content using tools
    # A page may require filling some form or pressing a button or whatever else to get access to its content
    async def page_access_agent(state : AgenticState) -> None :
        """
        Access agent responsible for navigating web pages and extracting full content using MCP tools.
        This agent uses an LLM to investigate web pages and determine the actions needed to extract
        the complete content. It uses MCP tools for browser automation (navigation, clicks, form filling, etc.)
        to gain full access to page content. If full content cannot be extracted, it tracks attempts
        and retries with different strategies.
        Args:
            state (AgenticState): The current state of the agentic crawler containing:
                - urls: Stack of URLs to visit (nested list, last element is current level)
                - url_current: Current active URL being processed
                - page_history: List of action plans executed on the current page
                - mcp_manager: Manager for MCP browser sessions
                - retries_max: Maximum retry attempts for failed page access
                - deepness_max: Maximum depth for tree exploration
        Workflow:
            1. If page_history is empty (new page):
               - Pop a URL from the URL stack
               - Navigate to the URL using MCP browser
               - Fetch current browser state via BrowserProxy.get_content()
            2. If page_history has items (retry mode):
               - Fetch latest browser state
               - Use LLM to analyze state and plan next action
            3. If full content is extracted:
               - Save content to browser_state
               - Clear page_history to signal completion
               - Return to proceed to content classifier agent
            4. If the prompt exceeds the LLM context budget (oversized page):
                - Log a warning, discard the page (clear browser_state/history/retries)
                  and continue to the next URL
            5. If content extraction fails and max retries reached:
                - Log the failure and continue to next URL
            6. Retry counter prevents infinite loops on stuck pages
        Returns:
            None (modifies state in-place)
        """
        
        # --- Handle URL stack and check if URL is already visited ---
        if not state["page_history"] :    # New page - need to get URL from stack
            state["retry_counter"][0] = 0 # Reset retry counter for the new page
            while True :
                while not state["urls"][-1] :
                    state["urls"].pop()  # Remove empty levels
                if not state["urls"] :
                    return None    
                else              :
                    url = state["urls"][-1].pop()
                # Omit already visited pages. READ-ONLY check (check): the visited DB
                # is NOT updated here - it is updated only after classification
                # in page_classify_agent. Returns 1 if already visited.
                if await state["redis_manager"].check_only(url) != 1 :
                    # Apriori site-specific pruning: reject known non-story branches
                    # (e.g. Yahoo Finance /quote/<TICKER> aggregators) BEFORE entering.
                    # Only URLs popped from the recursion lists (level > 0) are
                    # filtered; the zero-level (entry) URL is always visited as-is.
                    if len(state["urls"]) != 1 and not state["site_filter"].is_story_candidate(url) :
                        logger.info(f"SKIPPED URL (apriori site filter). url={url}")
                    else                                                :
                        state["url_current"][0] = url
                        logger.info(f"NEW URL. Navigating to: {url}")
                        await state["mcp_manager"].managed_browser.navigate(state["url_current"][0])
                        break
                else              :
                    logger.info(f"SKIPPED URL. Already visited: {url}")
        
        # Fetch current browser state via BrowserProxy.get_content()
        state["browser_state"][0] = await state["mcp_manager"].managed_browser.get_content()
        state["browser_state"][0] = clean_dom(state["browser_state"][0]) if len(state["browser_state"][0]) else ""
        if not len(state["browser_state"][0]) or state["retry_counter"][0] == state["retries_max"] :
            state["page_history"].clear()
            state["retry_counter"][0] = 0
            return None # Exhausted browser or retries

        # Enhanced prompt for investigating web pages and extracting content
        access_agent_prompt = (
            f"# ROLE & CONTEXT\n"
            f"You are an expert web scraping specialist focused on extracting financial content.\n"
            f"Your task is to analyze the current browser state and determine how to gain full access to the page content.\n"
            f"This is a systematic exploration process for tree-search crawling of financial websites.\n\n"
            f"# KEY INFORMATION\n"
            f"* **Current URL:** {state['url_current'][0]}\n"
            f"* **Maximum Depth:** {state['deepness_max']}\n"
            f"* **Current Depth:** {len(state['urls'])}\n"
            f"* **Maximum Retries:** {state['retries_max']}\n"
            f"* **Current Retry Attempt:** {state['retry_counter'][0]}\n"
            f"* **Instance ID:** {state['mcp_manager'].managed_browser.browser_instance_id}\n\n"
            f"# OBJECTIVE\n"
            f"Investigate the CURRENT BROWSER STATE and ACTIONS HISTORY to determine what actions are needed to extract MORE CONTENT from this web page.\n\n"
            f"## Content Analysis Requirements:\n"
            f"1. **Financial Content Focus:** Look for stock tickers (AAPL, MSFT, TSLA, GOOGL, etc.), news articles, market data, earnings reports, financial analysis\n"
            f"2. **Access Barriers:** Identify forms, login buttons, captcha, cookie banners, popups, or paywalls blocking content\n"
            f"3. **Content Depth:** Determine if the page shows full content or if pagination/ajax-loaded content is missing\n"
            f"4. **Interactive Elements:** Identify buttons, links, or form fields that need to be activated\n\n"
            f"## Strategy Priority:\n"
            f"1. If SIGN IN page/form detected → Prioritize authentication with Google account\n"
            f"2. If POP-UP or MODAL detected → Close it to access underlying content\n"
            f"3. If COOKIE BANNER detected → Dismiss it\n"
            f"4. If PAGINATION exists → Navigate to next pages\n"
            f"5. If LOAD MORE button exists → Click it\n"
            f"6. If content is truncated → Look for expandable elements\n\n"
            f"# TOOLKIT\n"
            f"You can operate with the following MCP browser tools:\n"
            f"{state['mcp_manager'].tools_summary}\n\n"
            f"## CRITICAL INSTRUCTIONS:\n"
            f"- If FULL CONTENT is extracted (all articles, tables, data visible) and no more browser actions are needed → Reply EXACTLY: 'ACCESS_OBTAINED' and DO NOT call any tools.\n"
            f"- Otherwise, you MUST CALL A TOOL to execute your chosen action (click, type, navigate, execute_script, etc.).\n"
            f"- DO NOT write your action plan as plain text. Instead, directly invoke the tool with the exact element descriptions, selectors, or values required.\n\n"
            f"## ACTIONS HISTORY (Previous Attempts)\n"
            f"```text\n"
            f"{'\n'.join(f'{i+1}. {s}' for i, s in enumerate(state["page_history"]))}\n"
            f"```\n\n" 
            f"# DATA \n"
            f"## CURRENT BROWSER STATE (HTML)\n"
            f"```html\n"
            f"{state['browser_state'][0]}\n"
            f"```\n\n"
        )

        # BACKLOG: Oversized pages are hard-skipped for now (page is discarded and not saved).
        # Later we might fix this behavior by salvaging such pages instead of dropping them,
        # e.g. via aggressive DOM truncation, chunked summarization, or a two-pass
        # links-only extraction, and possibly persist the skip in Redis to avoid re-fetching.
        prompt_tokens = await get_tools_tokens_length(access_agent_prompt)
        if prompt_tokens > llm_base_context_len :
            logger.warning(f"OVERSIZED PAGE SKIPPED. url={state['url_current'][0]}, prompt_tokens={prompt_tokens} > context_budget={llm_base_context_len}")
            state["browser_state"][0] = ""
            state["page_history"].clear()
            state["retry_counter"][0] = 0
            return None

        # Query vLLM's generation
        input_messages = [ SystemMessage(content=access_agent_prompt),
                           HumanMessage(content="Analyze the browser state and provide action plan for content extraction. Use the output format specified.") ]
        response = await llm_base_with_tools.ainvoke( input_messages, 
                                                      config={"configurable": {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "high"}}} )
        if response.content is None :
            state["page_history"].append("LLM returned no response")
            state["retry_counter"][0] += 1
        else                        :
            # Analyse the response    
            if "ACCESS_OBTAINED" in response.content :
                state["page_history"].clear()
                state["retry_counter"][0] = 0
            elif hasattr(response, 'tool_calls') and response.tool_calls : # Check for tool_calls in response and execute them using managed_browser.__call__
                tool_calls_log = []
                for tc in response.tool_calls :
                    tname = tc.value.get('name') if hasattr(tc, 'value') else tc.get('name')
                    if tname != "get_page_content" : 
                        targs = tc.value.get('arguments', {}) if hasattr(tc, 'value') else tc.get('arguments', {})
                        result = await state["mcp_manager"].managed_browser(tname, targs)
                        txt = result.content[0].text[:200] if result and hasattr(result, 'content') else ""
                        tool_calls_log.append(f"CALLED TOOL: {tname} -> {txt}\n")
                if len(tool_calls_log) : # Update the history with tools results
                    state["page_history"].append(f"{response.content.lstrip('\n').rstrip('\n')}\n{''.join(tool_calls_log)}\n")
                    state["retry_counter"][0] += 1
                else                   : # It is actually "ACCESS_OBTAINED" exit
                    state["page_history"].clear()
                    state["retry_counter"][0] = 0
            else : # Store LLM response text for replay
                state["page_history"].append(response.content.strip())
                state["retry_counter"][0] += 1
            
        return None 

    async def page_classify_agent(state: AgenticState) -> None :
        """
        Page classification agent that categorizes visited web pages.

        This agent analyzes the current browser state to determine the page type and appropriate action.
        It classifies pages into three categories:
        Categories:
            - IRRELEVANT: Pages with no stock-trading, economy, or financial content
                Action: Discard the page, no further processing
            -   RELEVANT:        
                - STORY: A news article containing solid high perplexity piece of financial content
                    Action: Save raw HTML to disk, publish metadata to RabbitMQ for downstream processing
                - AGGREGATOR: Relevant page but may not contain only links collection without a story
                    Action: Extract links, push to url stack for further traversal
        Args:
            state (AgenticState): Current crawler state with browser_state content

        Returns:
            None (modifies state in-place by updating url stacks, page_history, etc.)
        """

        # Use managed_browser proxy for cleaner and more robust browser interactions
        # The proxy automatically handles instance_id and connection health checks
        if len(state["browser_state"][0]) :
            # Ensure managed_browser is available before proceeding
            assert state["mcp_manager"].managed_browser is not None , "Managed browser is not available!"
            # Extract links from the page using the managed_browser proxy's get_links() method
            # This is cleaner than direct session.call_tool() and uses the existing BrowserProxy functionality
            # get_links() returns Optional[Set[str]] directly, not an MCP result object
            links_result = await state["mcp_manager"].managed_browser.get_links()
            extracted_links = list(links_result) if links_result else []

            # =====================================================================
            # IMPROVED PROMPT for Page Classification Agent
            # =====================================================================
            # Key improvements:
            # 1. Relevance-first decision structure (RELEVANT vs IRRELEVANT)
            # 2. Strict requirement that AGGREGATOR must aggregate RELEVANT financial content
            # 3. Explicit criteria differentiating financial vs non-financial links
            # 4. Step-by-step evaluation framework
            # =====================================================================

            classifier_agent_prompt = (
                f"# ROLE & EXPERTISE\n"
                f"You are a senior financial market researcher and web content analyst.\n"
                f"Your task is to classify web pages based on HTML content and link analysis.\n"
                f"Extracted Links Count: {len(extracted_links)} links found on this page.\n\n"
                f"# HIERARCHICAL CLASSIFICATION RULE\n"
                f"Evaluate relevance FIRST:\n"
                f"1. Is the primary content or link set financial/market-related? -> RELEVANT\n"
                f"   - If RELEVANT and primary content is a single article/report -> STORY\n"
                f"   - If RELEVANT and primary content is a collection of financial links -> AGGREGATOR\n"
                f"2. Is the content non-financial OR an index of non-financial links? -> IRRELEVANT\n\n"
                f"# CATEGORY DEFINITIONS\n\n"
                f"## 1. STORY (Relevant Financial Article)\n"
                f"**Definition:** A page primarily displaying a single news story, report, analysis, or article focused on finance, stock markets, or the economy.\n"
                f"**HTML Patterns:**\n"
                f"  - <article> tags, main content blocks, or <h1>-<h3> title headers\n"
                f"  - Financial tickers (AAPL, TSLA, MSFT) or financial terms (earnings, revenue, yield, dividend)\n"
                f"  - Byline, date/timestamp, and long-form narrative text\n"
                f"**Examples:**\n"
                f"  - 'Tesla Q3 Earnings Beat Expectations'\n"
                f"  - 'Fed Signals Rate Adjustments for Next Quarter'\n\n"
                f"## 2. AGGREGATOR (Relevant Financial Directory / Link Hub)\n"
                f"**Definition:** A directory, index, or listing page whose primary function is providing links to RELEVANT FINANCIAL content.\n"
                f"**HTML Patterns:**\n"
                f"  - High link count ({len(extracted_links)} links extracted) pointing to financial stories or tickers\n"
                f"  - Lists (<ul>, <ol>), tables, or grids containing news headlines, stock screeners, or market summaries\n"
                f"  - Category hubs like 'Top Market Movers', 'Financial News Today', 'Sector Watch'\n"
                f"**Examples:**\n"
                f"  - A stock market news index listing dozens of financial article headlines\n"
                f"  - A stock screener result list linking to individual ticker pages\n"
                f"  - A market calendar page with links to company earnings reports\n\n"
                f"## 3. IRRELEVANT (Non-Financial OR Irrelevant Directory)\n"
                f"**Definition:** Pages lacking market/financial substance, OR pages that aggregate NON-FINANCIAL / general site links.\n"
                f"**HTML Patterns:**\n"
                f"  - Non-financial topics: Careers, About Us, Terms of Service, Privacy Policy, Help/FAQ\n"
                f"  - General site navigation or site maps devoid of financial market context\n"
                f"  - Directories listing non-financial content (e.g., general entertainment, non-financial tech specs)\n"
                f"  - Login, error pages (404/500), or empty boilerplate pages\n"
                f"**Examples:**\n"
                f"  - A website's 'Careers' or 'Contact Us' page\n"
                f"  - An index page listing generic site help articles\n"
                f"  - A general user login or subscription paywall landing page\n\n"
                f"# DECISION FRAMEWORK\n"
                f"Step 1 [RELEVANCE FILTER]: Does this page contain meaningful financial/market content or links to financial topics? If NO -> IRRELEVANT.\n"
                f"Step 2 [FORMAT DETERMINATION]: If YES (it is RELEVANT):\n"
                f"       - Is it mostly structured text / a single news article? -> STORY\n"
                f"       - Is it primarily a collection / list of links to financial articles/data? -> AGGREGATOR\n\n"
                f"# OUTPUT FORMAT - STRICT REQUIREMENTS\n"
                f"RELEVANCE: <RELEVANT|IRRELEVANT>\n"
                f"TYPE: <STORY|AGGREGATOR|NONE> (If RELEVANT, specify STORY or AGGREGATOR. If IRRELEVANT, specify NONE)\n"
                f"Analysis: [2-3 sentences explaining your decision, confirming relevance before structure]\n\n"
                f"# CURRENT HTML\n"
                f"```html\n{state['browser_state'][0]}\n```\n"
            )

            input_messages = [ SystemMessage(content=classifier_agent_prompt),
                               HumanMessage(content="Analyze the browser state and classify the web page content.") ]
            response = await llm_base.ainvoke( input_messages, 
                                               config={"configurable": {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "high"}}} )
            
            # Parse the response - Hierarchical parsing
            if response.content:
                relevance_match = re.search(r'RELEVANCE:\s*(RELEVANT|IRRELEVANT)', response.content, re.IGNORECASE)
                if not relevance_match or relevance_match.group(1).upper() != "RELEVANT":
                    # IRRELEVANT  --  Discard page, but still mark as visited in RedisDB
                    redis_res = await state["redis_manager"].check_and_add(state["url_current"][0])
                    logger.info(f"IRRELEVANT page. Marked as visited in Redis (add_status={redis_res}): {state['url_current'][0]}")
                else                                                                    :  
                    # RELEVANT
                    type_match = re.search(r'TYPE:\s*(STORY|AGGREGATOR)', response.content, re.IGNORECASE)
                    if type_match and type_match.group(1).upper() == "STORY" :  # STORY  --  save it at hdd
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
                        rabbit_res = await state["rabbit_manager"].publish_json({"url": state["url_current"][0],
                                                                                 "file_path": file_path,
                                                                                 "timestamp": state["date_str"],})
                        logger.info(f"STORY page. Published to RabbitMQ (publish_status={rabbit_res}): url={state['url_current'][0]}, file={file_path}")
                        state["pages_saved"][0] += 1
                        # Store visited URL in RedisDB
                        redis_res = await state["redis_manager"].check_and_add(state["url_current"][0])
                        logger.info(f"STORY page. Marked as visited in Redis (add_status={redis_res}): {state['url_current'][0]}")
                    else                                                     :  # Aggregator page - DON'T store in RedisDB                        
                        logger.debug(f"AGGREGATOR page. URL NOT marked as visited in Redis (eligible for revisit): {state['url_current'][0]}")
                    # Extract links if we have some hops
                    if  len(state["urls"]) < state["deepness_max"] and extracted_links :
                        # Apriori site-specific pruning: drop known non-story branches
                        # so they never enter the traversal stack.
                        links_to_push = state["site_filter"].filter_urls(extracted_links)
                        if links_to_push :
                            state["urls"].append(links_to_push)
            state["browser_state"][0] = ""
        return None

    # --- Graph Assembly ---
    # This section constructs the LangGraph state machine that orchestrates the crawler agents.
    # The graph defines a workflow for processing a SINGLE URL cycle:
    #
    #    EXTERNAL WHILE URLs LOOP :
    #           .----------------------.
    #           V                      |
    #    START -.-> page_access_agent -.-> page_classify_agent -.-> END
    #
    # The workflow:
    # 1. START triggers page_access_agent to handle page content access.
    # 2. page_access_agent retries if page_history is non-empty, otherwise moves to page_classify_agent.
    # 3. page_classify_agent categorizes the page and extracts new URLs into state.urls.
    # 4. Graph exits to END after each page cycle. The external Python loop handles multi-page traversal.
    builder = StateGraph(AgenticState)
    builder.add_node("page_access_agent",   page_access_agent)
    builder.add_node("page_classify_agent", page_classify_agent)
    builder.add_edge(START, "page_access_agent")
    builder.add_conditional_edges("page_access_agent", lambda state : str(len(state["page_history"]) == 0), 
                                  { "True" : "page_classify_agent", "False" : "page_access_agent" })
    
    # Route classification directly to END so each graph run handles 1 page cycle
    builder.add_edge("page_classify_agent", END)
    
    # Compile ONCE outside the loop
    graph = builder.compile()

    state = AgenticState({ "date_str"         : datetime.datetime.now().strftime('%Y-%m-%d'),
                           "urls"             : [[url_string,],],
                           "url_current"      : ["",],
                           "deepness_max"     : deepness_max,
                           "pages_saved"      : [0,],
                           "retries_max"      : retries_max,
                           "retry_counter"    : [0,],
                           "browser_state"    : ["",],
                           "page_history"     : [],
                           "html_folder_name" : html_folder_name,
                           "mcp_manager"      : mcp_manager,
                           "rabbit_manager"   : rabbit_manager,
                           "redis_manager"    : redis_manager,
                           "site_filter"      : site_filter,
                        })
    
    # External driver loop: reuses the compiled 'graph' across all URL iterations
    while state["urls"] and sum(map(len, state["urls"])) > 0 :
        if ( pages_max > 0 and state["pages_saved"][0] >= pages_max ) : break  # break on pages limit
        state = await graph.ainvoke(state)
        assert not len(state["browser_state"][0]) and not len(state["page_history"]) and not state["retry_counter"][0] , "Failure in agentic clean-up."
        state["url_current"][0] = ""

    logger.info(f"Saved pages:    {state['pages_saved'][0]}")
    logger.info("======================================================")

    return 0


async def run_stealth_graph(url_string : str = "https://ca.finance.yahoo.com/", deepness_max : int = 2, pages_max :int = -1, retries_max : int = 3,
                            html_folder_name : str = "./html/", rabbit_host : str = "localhost", rabbit_port : int = 15672, redis_host : str = "localhost", redis_port : int = 6379) :
    """
    Main orchestration function that initializes all components and runs the crawler.
    
    This function serves as the entry point that wires together all subsystems:
    MCP browser management, LLM interaction, RabbitMQ messaging, and Redis storage.
    It demonstrates the complete lifecycle from initialization to cleanup.
    
    Args:
        url_string (str, default="https://ca.finance.yahoo.com/"): Entry URL for crawling.
        deepness_max (int, default=2): Maximum tree depth for exploration.
        pages_max (int, default=-1): Maximum pages to visit (-1 = unlimited).
        retries_max (int, default=3): Maximum retry attempts for failed page access.
        rabbit_host (str, default="localhost"): RabbitMQ server hostname.
        rabbit_port (int, default=15672): RabbitMQ management port.
        redis_host (str, default="localhost"): Redis server hostname.
        redis_port (int, default=6379): Redis server port.
        
    DEBUG MODE:
        Set the DEBUG environment variable to 'true', '1', or 'yes' (case-insensitive)
        to use mock implementations of RabbitMQ and Redis clients instead of real ones.
        This is useful for unit testing and debugging without requiring actual
        RabbitMQ and Redis servers.
        
        Example:
            DEBUG=true python run_agent.py --url_string "https://example.com"
        
    Note:
        When DEBUG is true, the mock implementations store data in memory and
        do not persist data across runs. This is ideal for isolated testing.
        
    Returns:
        int: Return code (0 = success, 1 = MCP/RabbitMQ/Redis connection failure, -1 = LLM failure)
        
    Workflow:
        1. Creates StealthMCPManager context for browser lifecycle
        2. Connects to stealth-browser MCP server
        3. Spawns a managed browser instance (headless=False for debugging)
        4. Initializes OpenAI-compatible LLM client with specific model
        5. Binds LLM tools for browser automation
        6. Tests LLM connectivity with ping
        7. Connects to RabbitMQ for message publishing
        8. Connects to Redis for visited URL tracking
        9. Invokes run_agentic_crawler with all components
        10. Returns any error codes from the agent execution
        
    Raises:
        ConnectionError: From RabbitMQClient.__aenter__ if connection fails.
        
    Note:
        The function uses async context managers for automatic cleanup of resources.
        The MCP session is created within the manager context, and all connections
        are properly closed on exit or error.
    """
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

        # Plug-in LLM
        llm_base = ChatOpenAI( base_url=f"{BASE_URL}/v1",
                               api_key=BASE_API_KEY,
                               model_name=MODEL_NAME,
                               max_tokens=LLM_MAX_OUTPUT_TOKENS,
                               temperature=0.1,)
        llm_base_with_tools = llm_base.bind_tools(mcp_manager.tools, tool_choice="auto")
        # Test LLM connectivity with a simple ping
        try : # Request a minimal token completion with a strict timeout
            response = await asyncio.wait_for(
                llm_base.ainvoke("ping", config={"max_tokens": 1}), 
                timeout=30.0  # Define timeout value
            )
            assert bool(response and response.content) , "Failed to connect LLM!"
        except Exception as e :
            logger.error(f"LLM Failed with error code {e}")
            return -1
        # Get tools tokens number
        llm_tooks_tokens_num = await get_tools_tokens_length(json.dumps(llm_base_with_tools.kwargs.get("tools", [])))

        # Link RabbitMQ 
        async with RabbitMQClient(host=rabbit_host, port=rabbit_port,
                                  publish_queue_name="crawler_json_queue", receive_queue_name=None,
                                  username="crawler", password="crawler") as rabbit_manager:
            rabbit_err_code = await rabbit_manager.connect()
            if rabbit_err_code:
                logger.error(f"Failed to connect RabbitMQ at host {rabbit_manager.host} port {rabbit_manager.port}")
                return 1

            # Connect to RedisDB
            async with RedisDBClient(host=redis_host, port=redis_port, db=0, readonly=False) as redis_manager:
                redis_err_code = await redis_manager.connect()
                if redis_err_code:
                    logger.error(f"Failed to connect RedisDB at host {redis_manager.host} port {redis_manager.port}")
                    return 1

                # Launch agents 
                agentic_error_code = await run_agentic_crawler(url_string=url_string, mcp_manager=mcp_manager,
                                                               llm_base_context_len=(LLM_BASE_CONTEXT_LEN - llm_tooks_tokens_num - LLM_MAX_OUTPUT_TOKENS - 1024),
                                                               llm_base=llm_base, llm_base_with_tools=llm_base_with_tools, 
                                                               deepness_max=deepness_max, pages_max=pages_max, retries_max=retries_max,
                                                               html_folder_name=html_folder_name, rabbit_manager=rabbit_manager, redis_manager=redis_manager)

    return agentic_error_code


# python3 ./src/crawler/crawler_agent.py -u "https://ca.finance.yahoo.com/"  -d 2 -f "./html" -n -1 -m 1 
if __name__ == "__main__":
    """
    Main entry point for command-line execution.
    
    Parses command-line arguments and launches the web crawler with user-specified
    configuration for URL, depth, page limits, and connection parameters.
    """
    import argparse
    parser = argparse.ArgumentParser(description="The web crawler.")

    parser.add_argument("--url_string",       "-u", type=str, default="https://ca.finance.yahoo.com/", dest="url_string",        help="Entry point url.")
    parser.add_argument("--deepness_max",     "-d", type=int, default=2,                               dest="deepness_max",      help="Deepness (in hops) of exploration.")
    parser.add_argument("--html_folder_name", "-f", type=str, default="./html/",                       dest="html_folder_name",  help="Name of the folder for storing downloaded html files.")
    parser.add_argument("--pages_max",        "-n", type=int, default=-1,                              dest="pages_max",         help="Maximal number of pages to extract.")
    parser.add_argument("--retries_max",      "-m", type=int, default=1,                               dest="retries_max",       help="Maximal number of pages to extract.")
    parser.add_argument("--rabbit_host",     "-bh", type=str, default="localhost",                     dest="rabbit_host",       help="RabbitMQ host.")
    parser.add_argument("--rabbit_port",     "-bp", type=int, default=5672,                            dest="rabbit_port",       help="RabbitMQ port.")
    parser.add_argument("--redis_host",      "-rh", type=str, default="localhost",                     dest="redis_host",        help="RedisDB host.")
    parser.add_argument("--redis_port",      "-rp", type=int, default=6379,                            dest="redis_port",        help="RedisDB port.")
    args = parser.parse_args()

    return_code = asyncio.run(run_stealth_graph(
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
# Replace the hard-skip of oversized pages with proper handling (chunking / DOM truncation / salvage links)

