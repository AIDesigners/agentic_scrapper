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
import asyncio
from contextlib import asynccontextmanager
import re
import json
import datetime
import uuid
import gzip
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, RemoveMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import ClientSession
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition


from web_tools import clean_dom


# --- DEBUG FLAG: Use mock implementations for testing/debugging ---
# Set the DEBUG environment variable to 'true', '1', or 'yes' (case-insensitive)
# to use mock implementations instead of real RabbitMQ and Redis clients.
# This is useful for unit testing and debugging without requiring actual
# RabbitMQ and Redis servers.
DEBUG = os.environ.get('DEBUG', '').lower() in ('true', '1', 'yes')

if DEBUG:
    logger.debug("INFO. Running in DEBUG mode.")

# Import the unified clients with mock support
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
                    mcp_response = await self("reload_page", {"wait_until": "networkidle"})
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
                # Assuming managed_browser holds the ID or can be stringified as expected by the tool
                await self._mcp_session.call_tool("close_browser", {"instance_id": self.managed_browser})
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
        pages_number (int): Number of visited pages at current depth.
        pages_saved (int): Number of saved stories at current depth.
        pages_max (int): Maximum pages to visit (-1 means unlimited).
        retries_max (int): Maximum retry attempts for failed page access.
        retry_counter (int): Internal counter for tracking retry attempts on current page.
        browser_state (str): Current browser page content.
        page_history (List[str]): Stack of action plans executed on current page.
        mcp_manager (StealthMCPManager): Handler for MCP/MCP tools integration.
        rabbit_manager (RabbitMQClient): Handler for message queue operations.
        redis_manager (RedisDBClient): Handler for visited URL tracking.
    """
    date_str         : str
    urls             : List[List[str]]
    url_current      : str
    deepness_max     : int 
    pages_number     : int 
    pages_saved      : int 
    retries_max      : int 
    retry_counter    : int
    browser_state    : str
    page_history     : List[str] 

    mcp_manager      : StealthMCPManager  
    rabbit_manager   : RabbitMQClient     
    redis_manager    : RedisDBClient


# The agentic system
async def run_agentic_crawler(url_string : str, mcp_manager : StealthMCPManager,
                              llm_base_context_len : int, llm_base : ChatOpenAI, llm_base_with_tools : ChatOpenAI,
                              rabbit_manager : RabbitMQClient, redis_manager : RedisDBClient,
                              deepness_max : int = 2, pages_max : int = -1, retries_max : int = 3) -> int :
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
            4. If content extraction fails and max retries reached:
               - Log the failure and continue to next URL
            5. Retry counter prevents infinite loops on stuck pages
            
        Returns:
            None (modifies state in-place)
        """
        
        # --- Handle URL stack and check if URL is already visited ---
        if not state.page_history:  # New page - need to get URL from stack
            while True :
                while not state.urls[-1] :
                    state.urls.pop()  # Remove empty levels
                if not state.urls :
                    return None    
                else              :
                    url = state.urls[-1].pop()
                # Check if URL is already visited via RedisDB
                visited = await state.redis_manager.check_and_add(url)
                if visited != 0:  # Already existed if 0
                    state.url_current = url
                    await state.mcp_manager.managed_browser.navigate(state.url_current)
                    break
        
        # Fetch current browser state via BrowserProxy.get_content()
        state.browser_state = await state.mcp_manager.managed_browser.get_content()
        state.browser_state = clean_dom(state.browser_state) if state.browser_state else ""

        # --- Only proceed to LLM analysis if browser_state is non-empty ---
        if not state.browser_state:
            return None

        # Enhanced prompt for investigating web pages and extracting content
        access_agent_prompt = (
            f"# ROLE & CONTEXT\n"
            f"You are an expert web scraping specialist focused on extracting financial content.\n"
            f"Your task is to analyze the current browser state and determine how to gain full access to the page content.\n"
            f"This is a systematic exploration process for tree-search crawling of financial websites.\n\n"
            
            f"# KEY INFORMATION\n"
            f"* **Browser Instance ID:** {state.mcp_manager.managed_browser}\n"
            f"* **Current URL:** {state.url_current}\n"
            f"* **Current Depth:** {state.deepness_max}\n"
            f"* **Maximum Retries:** {state.retries_max}\n"
            f"* **Current Retry Attempt:** {state.retry_counter}\n\n"

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
            f"{state.mcp_manager.tools_summary}"

            f"## Available Actions for LLM:\n"
            f"- Use 'click' tool to click on buttons/links\n"
            f"- Use 'type' tool to fill form fields\n"
            f"- Use 'navigate' tool to go to specific URLs\n"
            f"- Use 'execute_script' tool to run JavaScript for complex interactions\n"
            f"- Use 'get_page_content' to re-fetch content after changes\n\n"

            f"## CRITICAL INSTRUCTIONS:\n"
            f"- If FULL CONTENT is extracted (all articles, tables, data visible) → Reply EXACTLY: 'ACCESS_OBTAINED'\n"
            f"- If partial content extracted but more available → Provide DETAILED step-by-step instruction\n"
            f"- Include specific element selectors (CSS/XPath) for the LLM to execute\n"
            f"- For forms: specify 'name', 'value', or CSS selector for input fields\n"
            f"- For buttons: specify exact text or CSS selector\n\n"

            f"# DATA \n"
            f"## CURRENT BROWSER STATE (HTML)\n"
            f"```html\n"
            f"{state.browser_state}\n"
            f"```\n\n"

            f"## ACTIONS HISTORY (Previous Attempts)\n"
            f"```text\n"
            f"{'\n'.join(f'{i+1}. {s}' for i, s in enumerate(state.page_history))}\n"
            f"```\n\n"

            f"# OUTPUT FORMAT\n"
            f"Respond with a clear action plan. If you need to click a button, type in a field, or run any browser action, provide:\n"
            f"1. Action type (click/type/navigate/execute_script)\n"
            f"2. Target element description or selector\n"
            f"3. Any required values (for typing)\n"
            f"4. Expected outcome\n\n"
            f"Example format:\n"
            f"ACTION: click\n"
            f"TARGET: Button with text 'Accept Cookies'\n"
            f"SELECTOR: //button[contains(text(), 'Accept')]\n"
            f"EXPECTED: Cookie banner dismissed, main content visible\n"
        )

        # Use LLM to analyze the page and generate action plan
        response = await llm_base_with_tools.ainvoke(
            [SystemMessage(content=access_agent_prompt),
             HumanMessage(content="Analyze the browser state and provide action plan for content extraction. Use the output format specified.")],
            config={"configurable": {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "high"}}}
        )
        
        # Check for tool_calls in response and execute them using managed_browser.__call__
        if hasattr(response, 'tool_calls') and response.tool_calls :
            for tc in response.tool_calls:
                tname = tc.value.get('name') if hasattr(tc, 'value') else tc.get('name')
                targs = tc.value.get('arguments', {}) if hasattr(tc, 'value') else tc.get('arguments', {})
                result = await state.mcp_manager.managed_browser(tname, targs)
                txt = result.content[0].text[:200] if result and hasattr(result, 'content') else ""
                state.page_history.append(f"TOOL: {tname} -> {txt}")
            state["retry_counter"] += 1
        elif response.content is None :
            state.page_history.append("LLM returned no response")
            state["retry_counter"] += 1
        elif response.content.strip() == 'ACCESS_OBTAINED' :
            state.page_history.clear()
            state["retry_counter"] = 0
        else : # Store LLM response text for replay
            state.page_history.append(response.content.strip())
            state["retry_counter"] += 1
        
        return None 

    async def page_classify_agent(state: AgenticState) -> None :
        """
        Page classification agent that categorizes visited web pages.

        This agent analyzes the current browser state to determine the page type and appropriate action.
        It classifies pages into three categories:

        Categories:
            - STORY: A news article containing actionable financial content
                Action: Save raw HTML to disk, publish metadata to RabbitMQ for downstream processing
            - AGGREGATOR: A directory/list page with multiple links to explore
                Action: Extract links, push to url stack for further traversal
            - IRRELEVANT: Pages with no stock-trading, economy, or financial content
                Action: Discard the page, no further processing

        Args:
            state (AgenticState): Current crawler state with browser_state content

        Returns:
            None (modifies state in-place by updating url stacks, page_history, etc.)
        """

        # Use managed_browser proxy for cleaner and more robust browser interactions
        # The proxy automatically handles instance_id and connection health checks
        if len(state.browser_state) :
            # Ensure managed_browser is available before proceeding
            assert state.mcp_manager.managed_browser is not None , "Managed browser is not available!"
        
            # Extract links from the page using the managed_browser proxy's get_links() method
            # This is cleaner than direct session.call_tool() and uses the existing BrowserProxy functionality
            # get_links() returns Optional[Set[str]] directly, not an MCP result object
            links_result = await state.mcp_manager.managed_browser.get_links()
            extracted_links = list(links_result) if links_result else []

            # =====================================================================
            # IMPROVED PROMPT for Page Classification Agent
            # =====================================================================
            # Key improvements:
            # 1. Includes link count from extracted links as a key factor
            # 2. Detailed classification criteria with specific patterns to look for in HTML
            # 3. Concrete examples for each category
            # 4. Step-by-step analysis framework for the LLM
            # 5. Clear output format requirements
            # =====================================================================

            classifier_agent_prompt = (
                f"# ROLE & EXPERTISE\n"
                f"You are a senior financial market researcher and web content analyst.\n"
                f"Your task is to classify web pages based on HTML content analysis.\n"
                f"You have extracted links from this page - use the count as a key factor:\n"
                f"**Extracted Links Count:** {len(extracted_links)} links found on this page\n\n"

                f"# CLASSIFICATION CRITERIA - DETAILED\n\n"

                f"## STORY - A financial news article or report\n"
                f"**Definition:** A page that primarily displays news, analysis, or informational content\n"
                f"about financial markets, stocks, economy, or related topics.\n\n"
                f"**IDENTIFY STORY by checking for these HTML patterns:**\n"
                f"  1. <article> tags or <h1>-<h3> headers indicating story titles\n"
                f"  2. News keywords: 'reports', 'says', 'announces', 'confirms', 'earnings', 'forecast'\n"
                f"  3. Stock tickers: AAPL, MSFT, TSLA, AMZN, FB, NFLX, GOOGL, etc.\n"
                f"  4. Financial terms: profit, loss, dividend, portfolio, market, trading, ratio\n"
                f"  5. 'By [Author]' or publishing organization mentioned\n"
                f"  6. Date stamps or timestamps indicating recent news\n"
                f"  7. Long paragraphs of financial/news text content\n"
                f"**Examples of STORY pages:**\n"
                f"  - 'Tesla Q3 2024 Earnings Beat Expectations' (Yahoo Finance article)\n"
                f"  - 'Federal Reserve Holds Rates, Signals Future Cuts' (news headline)\n"
                f"  - 'Apple Reports Strong Quarterly Results' (earnings report)\n"
                f"  - Any page with primary content being financial news/text\n\n"

                f"## AGGREGATOR - A directory/list page with multiple links\n"
                f"**Definition:** A page that primarily contains links to other pages.\n"
                f"Think of it as a 'table of contents', index, or directory.\n\n"
                f"**IDENTIFY AGGREGATOR by checking for these HTML patterns:**\n"
                f"  1. Many <a href> elements pointing to different URLs\n"
                f"  2. <ul>, <ol>, <table> layouts with multiple links\n"
                f"  3. Navigation menus, header/footer links\n"
                f"  4. Category pages like 'Top Stories', 'Most Read', 'Market Movers'\n"
                f"  5. Pagination: 'Next', 'Previous', page numbers\n"
                f"  6. Directory format: company listings, sector indices\n"
                f"  7. **Link Count:** {len(extracted_links)} links found on this page\n\n"
                f"**Examples of AGGREGATOR pages:**\n"
                f"  - Page with links to 'Top 10 Stocks Today', 'Best Performers'\n"
                f"  - 'All News' page with list of headlines linking to articles\n"
                f"  - 'Stock Screener' results page with many stock links\n"
                f"  - 'Market Calendar' with dates linking to summaries\n"
                f"  - Pages where most of the content is links/navigation\n\n"

                f"## IRRELEVANT - No financial content of interest\n"
                f"**Definition:** A page with little to no content about stocks, finance, or markets.\n\n"
                f"**IDENTIFY IRRELEVANT by checking for:**\n"
                f"  1. Non-financial topics: About, contact, careers, help pages\n"
                f"  2. Pure media: images, videos, files without financial context\n"
                f"  3. Login/registration forms with no content\n"
                f"  4. Error pages: 404, 500, maintenance\n"
                f"  5. Empty or minimal content\n\n"
                f"**Examples of IRRELEVANT pages:**\n"
                f"  - Career Opportunities page at a company website\n"
                f"  - Contact Us form page\n"
                f"  - Image gallery or photo slideshow\n"
                f"  - PDF download page (unless financial report)\n"
                f"  - Social media links or icons page\n\n"

                f"# DECISION FRAMEWORK\n"
                f"Follow these steps to analyze the HTML:\n\n"
                f"Step 1: Analyze the EXTRACTED LINKS COUNT as an important indicator:\n"
                f"        - Many links found: Likely AGGREGATOR (directory, index, or listing page)\n"
                f"        - Few or no links found: Likely STORY or IRRELEVANT -> Check content type\n\n"

                f"Step 2: Scan for financial keywords and stock tickers:\n"
                f"        Keywords: stock, share, market, trading, earnings, dividend, ratio,\n"
                f"        bond, portfolio, investment, economy, finance, Bloomberg, etc.\n"
                f"        Tickers: AAPL, MSFT, GOOGL, TSLA, AMZN, FB, NFLX, etc.\n\n"

                f"Step 3: Examine the page STRUCTURE and CONTENT:\n"
                f"        - Mostly financial text + article structure? -> STORY\n"
                f"        - Mostly links for navigation/discovery? -> AGGREGATOR\n"
                f"        - Non-financial or minimal content? -> IRRELEVANT\n\n"

                f"Step 4: Make your final decision based on the HTML patterns AND link count.\n\n"

                f"# OUTPUT FORMAT - STRICT REQUIREMENTS\n"
                f"Response format:\n"
                f"\n"
                f"  CLASSIFICATION: <STORY|AGGREGATOR|IRRELEVANT>\n"
                f"  Analysis: [2-3 sentences explaining your decision, including link count factor]\n\n"

                f"# CURRENT HTML\n"
                f"```html\n{state.browser_state}\n```\n"
            )
            # Clean the browser state
            state.browser_state = ""

            # Call LLM
            response = await llm_base.ainvoke(
                [SystemMessage(content=classifier_agent_prompt),
                HumanMessage(content="Please classify this page. Follow the decision framework and output format exactly.")],
                            config={"configurable": {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "high"}}}
            )

            # Parse the response - look for the classification keyword
            if response.content is None:
                classification = "IRRELEVANT"
            else:
                classification_match = re.search(r'\b(STORY|AGGREGATOR|IRRELEVANT)\b', response.content, re.IGNORECASE)
                classification = "IRRELEVANT" if not classification_match else classification_match.group(0).upper()

            # Execute the appropriate action based on classification
            if classification == "STORY":
                save_dir = f"./html/{date_str}"
                os.makedirs(save_dir, exist_ok=True)
                # Generate a unique filename - keep trying until file creation succeeds
                # This ensures uniqueness even in concurrent scenarios
                compressed_browser_state = gzip.compress(state.browser_state.encode("utf-8"))
                while True:
                    unique_fname = str(uuid.uuid4())
                    file_path = os.path.join(save_dir, unique_fname + ".html.gz")
                    # Check if file already exists, if so try another name
                    if os.path.exists(file_path):
                        continue  # File exists, try another name
                    try:
                        # Attempt to create the file exclusively
                        with open(file_path, 'x') as f:
                            f.write(compressed_browser_state)
                        break  # File created successfully!
                    except FileExistsError :
                        continue  # Another process created it, try another name
                await state.rabbit_manager.publish_json({"url": state.url_current,
                                                        "file_path": file_path,
                                                        "timestamp": state.date_str,})
                state.pages_saved += 1
                # Store visited URL in RedisDB
                await state.redis_manager.check_and_add(state.url_current)
            elif classification == "IRRELEVANT":
                # Discard page, but still mark as visited in RedisDB
                await state.redis_manager.check_and_add(state.url_current)
            elif classification == "AGGREGATOR":
                # Aggregator page - extract links but DON'T store in RedisDB
                if extracted_links and len(state.urls) < state.deepness_max :
                    state.urls.append(extracted_links)
            state.browser_state = ""
        return None

    # --- Graph Assembly ---
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
    builder.add_conditional_edges("page_access_agent", lambda state : str(len(state.page_history) == 0), 
                                  { "True" : "page_classify_agent", "False" : "page_access_agent" })
    
    # Route classification directly to END so each graph run handles 1 page cycle
    builder.add_edge("page_classify_agent", END)
    
    # Compile ONCE outside the loop
    graph = builder.compile()

    state = AgenticState({ "date_str"       : datetime.datetime.now().strftime('%Y-%m-%d'),
                           "urls"           : [[url_string,],],
                           "url_current"    : "",
                           "deepness_max"   : deepness_max,
                           "pages_saved"    : 0,
                           "pages_number"   : 0,
                           "retries_max"    : retries_max,
                           "retry_counter"  : 0,
                           "browser_state"  : "",
                           "page_history"   : [],
                           "mcp_manager"    : mcp_manager,
                           "rabbit_manager" : rabbit_manager,
                           "redis_manager"  : redis_manager,
                 })
    
    # External driver loop: reuses the compiled 'graph' across all URL iterations
    while state["urls"] and sum(map(len, state["urls"])) > 0 and (pages_max < 0 or state["pages_number"] < pages_max) :
        state = await graph.ainvoke(state)
        assert not len(state.browser_state) and not len(state.page_history) and not state.retry_counter , "Failure in agentic clean-up."
        state.url_current = ""

    logger.info(f"Visited pages:    {state['pages_number']}")
    logger.info(f"Saved pages:    {state['pages_saved']}")
    logger.info("======================================================")

    return 0


async def run_stealth_graph(url_string : str = "https://ca.finance.yahoo.com/", deepness_max : int = 2, pages_max :int = -1, retries_max : int =3,
                            rabbit_host : str = "localhost", rabbit_port : int = 15672, redis_host : str = "localhost", redis_port : int = 6379) :
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
        llm_base_context_len = 262144
        llm_base = ChatOpenAI( base_url="http://ifo4:8000/v1",
                               api_key="alex_llm_qwen",
                               model_name="poolside/Laguna-XS-2.1-NVFP4",
                               # model_name="QuantTrio/Qwen3.6-27B-AWQ",
                               temperature=0.1, )
        llm_base_with_tools = llm_base.bind_tools(mcp_manager.tools)
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

        # Link RabbitMQ 
        async with RabbitMQClient(host=rabbit_host, port=rabbit_port,
                                  publish_queue_name="crawler_json_queue", receive_queue_name=None,
                                  username="crawler", password="ceawler") as rabbit_manager:
            rabbit_err_code = await rabbit_manager.connect()
            if rabbit_err_code:
                logger.error(f"Failed to connect RabbitMQ at host {rabbit_manager.get_host()} port {rabbit_manager.get_port()}")
                return 1

            # Connect to RedisDB
            async with RedisDBClient(host=redis_host, port=redis_port, db=0, readonly=False) as redis_manager:
                redis_err_code = await redis_manager.connect()
                if redis_err_code:
                    logger.error(f"Failed to connect RedisDB at host {redis_manager.get_host()} port {redis_manager.get_port()}")
                    return 1

                # Launch agents 
                agentic_error_code = await run_agentic_crawler(url_string=url_string, mcp_manager=mcp_manager,
                                                               llm_base_context_len=llm_base_context_len, llm_base=llm_base, llm_base_with_tools=llm_base_with_tools,
                                                               rabbit_manager=rabbit_manager, redis_manager=redis_manager,
                                                               deepness_max=deepness_max, pages_max=pages_max, retries_max=retries_max)

    return agentic_error_code


if __name__ == "__main__":
    """
    Main entry point for command-line execution.
    
    Parses command-line arguments and launches the web crawler with user-specified
    configuration for URL, depth, page limits, and connection parameters.
    """
    import argparse
    parser = argparse.ArgumentParser(description="The web crawler.")

    parser.add_argument("--url_string",   "-u", type=str, default="https://ca.finance.yahoo.com/", dest="url_string",   help="Entry point url.")
    parser.add_argument("--deepness_max", "-d", type=int, default=2,                               dest="deepness_max", help="Deepness (in hops) of exploration.")
    parser.add_argument("--pages_max",    "-n", type=int, default=-1,                              dest="pages_max",    help="Maximal number of pages to extract.")
    parser.add_argument("--retries_max",  "-m", type=int, default=3,                               dest="retries_max",  help="Maximal number of pages to extract.")
    parser.add_argument("--rabbit_host", "-bh", type=str, default="localhost",                     dest="rabbit_host",  help="RabbitMQ host.")
    parser.add_argument("--rabbit_port", "-bp", type=int, default=15672,                           dest="rabbit_port",  help="RabbitMQ port.")
    parser.add_argument("--redis_host",  "-rh", type=str, default="localhost",                     dest="redis_host",   help="RedisDB host.")
    parser.add_argument("--redis_port",  "-rp", type=int, default=6379,                            dest="redis_port",   help="RedisDB port.")
    args = parser.parse_args()

    return_code = asyncio.run(run_stealth_graph(
        url_string=args.url_string,
        deepness_max=args.deepness_max,
        pages_max=args.pages_max,
        retries_max=args.retries_max,
        rabbit_host=args.rabbit_host,
        rabbit_port=args.rabbit_port,
        redis_host=args.redis_host,
        redis_port=args.redis_port
    ))
    logger.info(f"Agent finished with {return_code} error code.")
    