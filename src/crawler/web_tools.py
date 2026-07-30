import os, logging
import json
import inspect
from typing import Callable, Dict, Set, Any, AsyncIterator, Coroutine, Protocol, List, Optional
from contextlib import asynccontextmanager
from langchain_mcp_adapters.client import MultiServerMCPClient



# Define structural protocols to avoid importing from the raw 'mcp' package
class MCPContentBlock(Protocol):
    text: str
class MCPToolResult(Protocol):
    content: List[MCPContentBlock]
class MCPSession(Protocol):
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> MCPToolResult : ...

# The proxy browser class under controll of MCP
class BrowserProxy :
    def __init__(self, logger : logging.Logger, mcp_session : Optional[MCPSession] = None, browser_instance_id : Optional[str] = None) -> None :
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


# The Browser runner wrapper
class MCPBrowserRunner :
    """
    Manages the lifecycle of an MCP MultiServer client and its managed stealth browser.
    Decouples environment setup and connection management from agent workflows.
    """
    def __init__(self, logger : logging.Logger, server_config : Optional[Dict[str, Any]] = None) -> None:
        # ToDO: replace it with some config file of whatever
        self.logger = logger
        self.server_config : Dict[str, Any] = server_config or {
            "stealth-browser": {
                "transport": "stdio",
                "command": "/home/ayakovenko/nila/stealth-browser-mcp/bin/python3",
                "args": ["/home/ayakovenko/nila/stealth-browser-mcp/src/server.py"],
                "env": {
                    "DISPLAY": ":1",
                    "CHROME_PATH": "/home/ayakovenko/nila/flatpak-chromium.bash",
                    "BROWSER_EXECUTABLE_PATH": "/home/ayakovenko/nila/flatpak-chromium.bash",
                    "PUPPETEER_EXECUTABLE_PATH": "/home/ayakovenko/nila/flatpak-chromium.bash"
                }
            }
        }
        self.client : MultiServerMCPClient = MultiServerMCPClient(self.server_config)
        self.mcp_session : Optional[MCPSession] = None
        self.managed_browser : BrowserProxy = BrowserProxy(self.logger)

    async def get_managed_browser(self) -> str :
        self.logger.info("Spawning new managed browser instance...")
        assert self.managed_browser.browser_instance_id is None , "An existing managed browser should be closed first"
        result : MCPToolResult = await self.mcp_session.call_tool("spawn_browser", {"headless": False})
        content_text: str = result.content[0].text if getattr(result, "content", None) else str(result)
        try:
            data : Dict[str, Any] = json.loads(content_text)
            instance_id : str = data.get("instance_id", content_text.strip())
        except (json.JSONDecodeError, TypeError):
            instance_id = content_text.strip()
        self.logger.info(f"New managed browser is ready. Instance ID: {instance_id}")
        self.managed_browser = BrowserProxy(self.logger, mcp_session=self.mcp_session, browser_instance_id=instance_id)
        return instance_id

    async def close_managed_browser(self) -> None:
        if self.managed_browser.browser_instance_id :
            self.logger.info(f"Closing browser context for Instance ID: {self.managed_browser.browser_instance_id }")
            try:
                await self.mcp_session.call_tool("close_browser", {"instance_id" : self.managed_browser.browser_instance_id })
                self.managed_browser.active_instance_id = None
            except Exception:
                pass

    async def run(self, agent_callback: Callable[..., Coroutine[Any, Any, Any] | Any], *args: Any, **kwargs: Any) -> Any:
        """
        Launches the MCP server environment, provisions the browser, and runs the passed agent workflow.
        """
        async with self.client.session("stealth-browser") as mcp_session :
            self.mcp_session = mcp_session
            self.managed_browser.mcp_session = mcp_session
            self.managed_browser.browser_instance_id = await self.get_managed_browser()
            self.logger.info("INFO. Executing agent workflow with web access.")
            try     :
                if inspect.iscoroutinefunction(agent_callback) :
                    return await agent_callback(self, *args, **kwargs)
                else                                           :
                    return       agent_callback(self, *args, **kwargs)
            finally :
                self.close_managed_browser()


#  ------------   U T I L I T Y   F U N C T I O N S   ------------ #

from bs4 import BeautifulSoup
import re

# The routine from gemini, IDK how smart is it but it shrinks some content
def clean_dom(raw_html : str) -> str:
    if not raw_html : return "" # Empty DOM
    soup = BeautifulSoup(raw_html, "html.parser")
    # --- STEP 1: Remove "Invisible" Elements ---
    # Removes elements hidden via inline styles
    hidden_styles = ["display: none", "visibility: hidden", "display:none", "visibility:hidden"]
    for tag in soup.find_all(True, style=True):
        if any(style in tag['style'].lower() for style in hidden_styles):
            tag.extract()
    # --- STEP 2: Semantic Pruning (Keep only functional/readable tags) ---
    # Only keep tags that contain actionable data for the agent
    allowed_tags = ["h1", "h2", "h3", "h4", "p", "a", "button", "input", "select", "li", "table", "tr", "td", "th"]
    # Create a new, minimal soup to hold only the pruned structure
    pruned_soup = BeautifulSoup("<html><body></body></html>", "html.parser")
    # Filter and append only allowed tags
    for tag in soup.find_all(allowed_tags):
        # We perform a small check to ensure we don't nest tags incorrectly
        # (e.g., if we extract a <td>, we should ensure the parent structure is kept
        # or simplified. For extreme shrinking, we keep only the leaf nodes)
        pruned_soup.body.append(tag)
    # --- Cleanup Attributes on remaining tags ---
    allowed_attrs = ["id", "href", "name", "type", "value", "placeholder"]
    for tag in pruned_soup.find_all(True):
        attrs = list(tag.attrs.keys())
        for attr in attrs:
            if attr not in allowed_attrs and not attr.startswith("data-"):
                del tag[attr]
    # --- Final Formatting ---
    cleaned_text = pruned_soup.prettify()
    cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text)
    # Return
    return cleaned_text


import tldextract

def is_same_registered_domain(base_url: str, target_url: str) -> bool:
    base = tldextract.extract(base_url)
    target = tldextract.extract(target_url)

    # Compare the domain and suffix (e.g., 'yahoo' and 'com')
    return (base.domain, base.suffix) == (target.domain, target.suffix)

