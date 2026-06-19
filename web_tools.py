import os
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
    def __init__(self, mcp_session : MCPSession, browser_instance_id: str) -> None :
        self.mcp_session = mcp_session
        self.browser_instance_id = browser_instance_id
    # Implementing __call__ allows you to do: await browser("tool_name", {...})
    async def __call__(self, tool_name: str, args: Dict[str, Any] | None = None) -> MCPToolResult:
        full_args = args or {}
        full_args["instance_id"] = self.browser_instance_id
        return await self.mcp_session.call_tool(tool_name, full_args)

    # The routine to open page
    async def navigate(self, url_string : str) -> Optional[MCPToolResult] :
        mcp_response = await self("navigate", {"url" : url_string, "wait_until" : "networkidle"})
        return mcp_response if not getattr(mcp_response, "isError", False) else None
    # The routine to get page content
    async def get_content(self) -> Optional[str] :
        mcp_response = await self("get_page_content")
        if not getattr(mcp_response, "isError", False) and mcp_response.content :
            raw_content = mcp_response.content[0].text.strip()
            if raw_content.startswith("{") and raw_content.endswith("}") :
                try :
                    meta = json.loads(raw_content)
                    if isinstance(meta, dict) and "file_path" in meta :
                        file_path = meta["file_path"]
                        if os.path.exists(file_path) :
                            with open(file_path, "r", encoding="utf-8") as f :
                                browser_state_raw = f.read()
                        else                         :
                            return f"Error: Offloaded file not found at {file_path}"
                    else                                              :
                        browser_state_raw = raw_content  # It was JSON but not our metadata
                except json.JSONDecodeError :
                    browser_state_raw = raw_content  # Not valid JSON
            else :
                browser_state_raw = raw_content  # Not JSON
            return browser_state_raw
        else                                                                    :
            return None

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
        tools_result = await self.mcp_session.call_tool("execute_script", {
            "instance_id": self.browser_instance_id,
            "script": js_script
        })
        if not getattr(tools_result, "isError", False) :
            data = json.loads(tools_result.content[0].text)
            return {item.get('value') for item in data.get('result', []) if item.get('value')}
        else                                     :
            return None


# The Browser runner wrapper
class MCPBrowserRunner :
    """
    Manages the lifecycle of an MCP MultiServer client and its managed stealth browser.
    Decouples environment setup and connection management from agent workflows.
    """
    def __init__(self, server_config: Dict[str, Any] | None = None) -> None:
        # ToDO: replace it with some config file of whatever
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

    @asynccontextmanager
    async def _managed_browser(self, mcp_session: MCPSession) -> AsyncIterator[str] :
        """Internal helper to provision the browser instance and yield its ID."""
        print("Spawning managed browser instance...")
        # Enforces the protocol contract safely
        result: MCPToolResult = await mcp_session.call_tool("spawn_browser", {"headless": False})
        content_text : str = result.content[0].text if getattr(result, "content", None) else str(result)
        try :
            data : Dict[str, Any] = json.loads(content_text)
            instance_id: str = data.get("instance_id", content_text.strip())
        except (json.JSONDecodeError, TypeError) :
            instance_id = content_text.strip()
        print(f"Browser ready. Instance ID: {instance_id}")
        try :
            yield instance_id
        finally :
            print(f"Tearing down browser context for Instance ID: {instance_id}")

    async def run(self, agent_callback: Callable[..., Coroutine[Any, Any, Any] | Any], *args: Any, **kwargs: Any) -> Any:
        """
        Launches the MCP server environment, provisions the browser, and runs the passed agent workflow.
        """
        async with self.client.session("stealth-browser") as mcp_session :
            async with self._managed_browser(mcp_session) as browser_instance_id:
                print("INFO. Executing agent workflow with web access.")
                browser = BrowserProxy(mcp_session, browser_instance_id)
                if inspect.iscoroutinefunction(agent_callback) :
                    return await agent_callback(browser, *args, **kwargs)
                else                                           :
                    return       agent_callback(browser, *args, **kwargs)


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

