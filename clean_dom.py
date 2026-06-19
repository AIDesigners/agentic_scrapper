
from bs4 import BeautifulSoup
import re


def clean_dom(raw_html: str) -> str:
    if not raw_html :
        return "Empty DOM"

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

    return cleaned_text




from langchain_mcp_adapters.sessions import ClientSession
import json
import os

async def get_browser_state(mcp_session : ClientSession, instance_id : str) -> str :
    assert mcp_session is not None, "ERROR: MCP session is None!"
    try:
        mcp_response = await mcp_session.call_tool("get_page_content", arguments={"instance_id" : instance_id})
        if not (mcp_response and hasattr(mcp_response, "content") and mcp_response.content):
            return "Error: No content received from MCP server."
        raw_content = mcp_response.content[0].text.strip()
        # Robust Detection Strategy:
        # 1. Attempt to parse as JSON.
        # 2. Verify it contains the 'file_path' key specifically used by the MCP tool.
        # 3. If parsing fails, treat it as plain text (HTML).
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
        return clean_dom(browser_state_raw)
    except Exception as e :
        return f"Failed to retrieve live DOM: {str(e)}"
