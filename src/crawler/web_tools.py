import os, logging
import json
import asyncio
from typing import Dict, Set, Any, Protocol, List, Optional
from contextlib import AsyncExitStack
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import ClientSession
from langchain_mcp_adapters.tools import load_mcp_tools



# Define structural protocols to avoid importing from the raw 'mcp' package
class MCPContentBlock(Protocol):
    text: str
class MCPToolResult(Protocol):
    content: List[MCPContentBlock]

# The proxy browser class under controll of MCP
class BrowserProxy :
    def __init__(self, logger : Optional[logging.Logger], _mcp_session : ClientSession, browser_instance_id : str) -> None :
        # The manager may be constructed with logger=None; fall back to a module
        # logger so the error paths below never crash on self.logger.*
        self.logger = logger if logger is not None else logging.getLogger("BrowserProxy")
        self._mcp_session = _mcp_session
        self.browser_instance_id = browser_instance_id

    # Generic entry point (LLM tool loop, StealthMCPManager.call_tool): pre-flight
    # probe + instance_id injection. The domain methods below (navigate,
    # get_content, get_sanitized_content, get_links) bypass this on purpose -
    # they call the session directly so every call is exactly one MCP round-trip
    # (the probe would double each of them).

    async def __call__(self, tool_name: str, args: Dict[str, Any] | None = None) -> Optional[MCPToolResult] :
        assert self.browser_instance_id is not None , "Managed browser is not initialized!"
        # 1. PRE-CALL HEALTH TEST
        try:
            result = await self._mcp_session.call_tool("execute_script",
                                                      {"instance_id": self.browser_instance_id, "script": "1"})
            assert not getattr(result, "isError", False)
        except Exception:
            # Captures library-level connection, read, or stream timeouts cleanly
            return None
        # 2. TIMEOUT PROTECTED TOOL EXECUTION
        full_args = args or {}
        full_args["instance_id"] = self.browser_instance_id
        return await self._mcp_session.call_tool(tool_name, full_args)

    # The routine to open page. Direct single-round-trip session call (bypasses
    # __call__ on purpose: no pre-flight probe); instance_id injected explicitly.
    # Transport errors are tolerated as None - callers re-navigate / retry.
    async def navigate(self, url_string : str) -> Optional[MCPToolResult] :
        try :
            mcp_response = await self._mcp_session.call_tool("navigate", {
                "url" : url_string, "wait_until" : "networkidle", "instance_id" : self.browser_instance_id
            })
        except Exception as e :
            self.logger.warning(f"navigate: MCP transport error: {e}")
            return None
        return mcp_response if mcp_response and not getattr(mcp_response, "isError", False) else None

    # Accessing the page content. Direct single-round-trip session calls
    # (bypass __call__ on purpose: no pre-flight probe); instance_id injected
    # explicitly. Returns None only on a transport-level failure; "" after the
    # retries are exhausted.
    async def get_content(self, max_retries: int = 3) -> Optional[str]:
        for attempt in range(max_retries):
            try : # Wrap it in try/except because it sometimes throw exceptions on MCP side
                mcp_response = await self._mcp_session.call_tool("get_page_content", {"instance_id" : self.browser_instance_id})
            except Exception as e :
                self.logger.warning(f"MCP transport error on attempt {attempt + 1}: {e}")
                mcp_response = None
            if mcp_response is None : return None
            # --- 1. IF JS FAILED / PAGE CRASHED ---
            if getattr(mcp_response, "isError", False) :
                self.logger.warning(f"JS/Page error on attempt {attempt + 1} of {max_retries}.")
                if attempt < max_retries - 1:
                    self.logger.warning("Triggering native page reload...")
                    # Direct session call like the fetch above. NOTE: deliberately
                    # unguarded for now (deferred): a transport dying mid-retry still
                    # propagates to the caller instead of resolving to None.
                    mcp_response = await self._mcp_session.call_tool("reload_page", {"instance_id" : self.browser_instance_id})
                    if not mcp_response : return None
                continue
            # --- 2. IF SUCCESSFUL ---
            if mcp_response and getattr(mcp_response, "content", None):
                raw_content = mcp_response.content[0].text.strip()
                # The get_page_content tool serializes a content dict
                # {"html", "text", "url", "title"}: inline when small, or as a
                # file_path envelope pointing to a file wrapping the same dict
                # in {"metadata": ..., "data": ...} when too large. In both
                # cases extract the real HTML instead of the JSON wrapper.
                if raw_content.startswith("{") and raw_content.endswith("}"):
                    try:
                        meta = json.loads(raw_content)
                        if isinstance(meta, dict):
                            if "html" in meta:
                                return str(meta.get("html", ""))
                            if "file_path" in meta:
                                file_path = meta["file_path"]
                                if os.path.exists(file_path):
                                    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                                        file_text = f.read()
                                    try:
                                        envelope = json.loads(file_text)
                                    except json.JSONDecodeError:
                                        return file_text
                                    data = envelope.get("data", envelope) if isinstance(envelope, dict) else envelope
                                    if isinstance(data, dict):
                                        return str(data.get("html", ""))
                                    return str(data)
                                else:
                                    self.logger.error(f"Offloaded file not found at {file_path}")
                                    return ""
                    except json.JSONDecodeError:
                        pass
                return raw_content
        # --- 3. IF ALL RETRIES FAIL ---
        self.logger.error(f"Failed to extract content after {max_retries} attempts. Returning empty page.")
        return ""

    # In-browser read-only JS DOM sanitizer. Direct single-round-trip
    # execute_script call (bypasses __call__ pre-flight probe, same retry /
    # reload contract as get_content).
    #
    # Why this runs inside the browser instead of BeautifulSoup on raw HTML:
    #   1. Live visibility: uses window.getComputedStyle(el) and
    #      getBoundingClientRect() to drop elements hidden via CSS classes or
    #      zero-sized containers, which static HTML parsing cannot see.
    #   2. Coherent sentences: serializes inline children (<a>, <span>, <b>,
    #      <em>) IN PLACE inside their enclosing <p>, <h1>-<h4>, <li>, <th>,
    #      <td> block so links/tickers are never detached from sentences.
    #   3. Intact tables: preserves <table><tr><th>/<td> hierarchy.
    #   4. Zero DOM footprint: 100% read-only (never mutates attributes, nodes,
    #      or window globals).
    # Returns None only on a transport-level failure; "" after retries fail.
    async def get_sanitized_content(self, max_retries: int = 3) -> Optional[str]:
        js_script = r"""
        (() => {
            if (!document.body) return "";

            // 1. Non-content / media / script tags to skip completely
            const DROP_TAGS = new Set([
                'SCRIPT', 'STYLE', 'NOSCRIPT', 'SVG', 'PATH', 'IMG', 'PICTURE',
                'SOURCE', 'VIDEO', 'AUDIO', 'CANVAS', 'IFRAME', 'OBJECT',
                'EMBED', 'META', 'LINK', 'HEAD', 'TEMPLATE', 'BR', 'HR'
            ]);

            // 2. Site chrome ARIA roles & ad/promo class markers
            const CHROME_ROLES = new Set([
                'navigation', 'banner', 'contentinfo', 'complementary', 'search', 'menubar'
            ]);
            const NOISE_RE = /\b(ad|ads|advert|advertisement|social-share|share-bar|newsletter|promo|outbrain|taboola)\b/i;

            const HEADING_TAGS = new Set(['H1', 'H2', 'H3', 'H4']);
            const BLOCK_PROSE_TAGS = new Set(['P', 'BLOCKQUOTE', 'PRE', 'FIGCAPTION']);
            const CONTROL_TAGS = new Set(['BUTTON', 'INPUT', 'SELECT', 'TEXTAREA']);
            const ASSET_RE = /\.(png|jpe?g|gif|svg|webp|ico|css|js|mp3|mp4|woff2?)$/i;

            function escapeHtml(str) {
                return String(str)
                    .replace(/&/g, '&amp;')
                    .replace(/</g, '&lt;')
                    .replace(/>/g, '&gt;');
            }

            function escapeAttr(str) {
                return escapeHtml(str).replace(/"/g, '"');
            }

            // 3. Live rendered visibility check via ComputedStyle + bounding box
            function isHidden(el) {
                if (el.hidden || el.getAttribute('aria-hidden') === 'true') return true;
                const style = window.getComputedStyle(el);
                if (
                    style.display === 'none' ||
                    style.visibility === 'hidden' ||
                    style.visibility === 'collapse' ||
                    parseFloat(style.opacity) === 0 ||
                    style.clip === 'rect(0px, 0px, 0px, 0px)' ||
                    style.clipPath === 'inset(100%)'
                ) {
                    return true;
                }
                if (style.display !== 'contents' && style.overflow !== 'visible') {
                    const rect = el.getBoundingClientRect();
                    if (rect.width === 0 && rect.height === 0) return true;
                }
                return false;
            }

            // 4. Skip site chrome (<nav>, <footer>, <aside>, outer <header>),
            //    while NEVER skipping an ancestor of <main>, <article>, or <h1>.
            function isChromeContainer(el) {
                if (el.querySelector && el.querySelector('article, main, [role="main"], h1')) {
                    return false;
                }
                const tag = el.tagName;
                if (tag === 'NAV' || tag === 'FOOTER' || tag === 'ASIDE') return true;
                if (tag === 'HEADER' && !el.closest('article, main, [role="main"]')) return true;

                const role = (el.getAttribute('role') || '').toLowerCase();
                if (CHROME_ROLES.has(role)) return true;

                const idAndClass = `${el.id || ''} ${typeof el.className === 'string' ? el.className : ''}`;
                if (NOISE_RE.test(idAndClass) && !el.closest('article')) return true;

                return false;
            }

            // 5. Normalize <a href> (drop #fragments, javascript:, mailto:, static assets)
            function cleanHref(el) {
                const raw = (el.getAttribute('href') || '').trim();
                if (
                    !raw ||
                    raw.startsWith('#') ||
                    raw.toLowerCase().startsWith('javascript:') ||
                    raw.toLowerCase().startsWith('mailto:')
                ) {
                    return null;
                }
                try {
                    const u = new URL(el.href, location.href);
                    if (u.protocol !== 'http:' && u.protocol !== 'https:') return null;
                    if (ASSET_RE.test(u.pathname)) return null;
                    u.hash = '';
                    return u.toString();
                } catch (e) {
                    return null;
                }
            }

            // 6. Serialize inline content inside a block (<p>, <h1>-<h4>, <li>, <td>, <th>)
            //    so inline <a href="..."> links remain in place within the sentence.
            function serializeInline(node) {
                if (!node) return '';
                if (node.nodeType === Node.TEXT_NODE) {
                    return escapeHtml(node.textContent.replace(/\s+/g, ' '));
                }
                if (node.nodeType !== Node.ELEMENT_NODE) return '';
                if (DROP_TAGS.has(node.tagName) || isHidden(node)) return '';

                if (node.tagName === 'A') {
                    const label = (node.innerText || node.textContent || '').replace(/\s+/g, ' ').trim();
                    if (!label) return '';
                    const href = cleanHref(node);
                    return href
                        ? `<a href="${escapeAttr(href)}">${escapeHtml(label)}</a>`
                        : escapeHtml(label);
                }

                let out = '';
                for (const child of node.childNodes) {
                    out += serializeInline(child);
                }
                return out;
            }

            function normalizeInlineWhitespace(htmlFragment) {
                return htmlFragment
                    .replace(/\s+/g, ' ')
                    .replace(/\s+([,.;:!?])/g, '$1')
                    .trim();
            }

            // 7. Serialize form/button controls with allowed attributes
            function serializeControl(el) {
                const tag = el.tagName.toLowerCase();
                const attrs = [];
                if (el.id) attrs.push(`id="${escapeAttr(el.id)}"`);
                if (el.getAttribute('name')) attrs.push(`name="${escapeAttr(el.getAttribute('name'))}"`);
                if (el.getAttribute('type')) attrs.push(`type="${escapeAttr(el.getAttribute('type'))}"`);
                if (el.getAttribute('placeholder')) attrs.push(`placeholder="${escapeAttr(el.getAttribute('placeholder'))}"`);

                if (el.tagName === 'INPUT') {
                    const val = (el.value || el.getAttribute('value') || '').trim();
                    if (val) attrs.push(`value="${escapeAttr(val.slice(0, 120))}"`);
                    return `<input${attrs.length ? ' ' + attrs.join(' ') : ''}>`;
                }
                if (el.tagName === 'SELECT') {
                    const selected = el.selectedOptions && el.selectedOptions[0]
                        ? el.selectedOptions[0].text.trim()
                        : '';
                    if (selected) attrs.push(`value="${escapeAttr(selected)}"`);
                    return `<select${attrs.length ? ' ' + attrs.join(' ') : ''}></select>`;
                }
                const label = (el.innerText || el.getAttribute('aria-label') || el.value || '')
                    .replace(/\s+/g, ' ').trim();
                if (!label) return '';
                return `<${tag}${attrs.length ? ' ' + attrs.join(' ') : ''}>${escapeHtml(label.slice(0, 150))}</${tag}>`;
            }

            // 8. Serialize <table> keeping <tr>, <th>, <td> hierarchy intact
            function serializeTable(tableEl) {
                const rows = [];
                for (const tr of tableEl.querySelectorAll('tr')) {
                    if (isHidden(tr)) continue;
                    const cells = [];
                    for (const cell of tr.querySelectorAll(':scope > th, :scope > td')) {
                        if (isHidden(cell)) continue;
                        const cTag = cell.tagName.toLowerCase();
                        const content = normalizeInlineWhitespace(serializeInline(cell));
                        if (content) cells.push(`<${cTag}>${content}</${cTag}>`);
                    }
                    if (cells.length > 0) {
                        rows.push(`<tr>${cells.join('')}</tr>`);
                    }
                }
                return rows.length > 0 ? `<table>\n${rows.join('\n')}\n</table>` : '';
            }

            // 9. Top-down DOM traversal
            const outLines = [];
            const seenBlocks = new Set();

            function emit(line) {
                if (!line) return;
                if (line.length > 30 && seenBlocks.has(line)) return;
                if (line.length > 30) seenBlocks.add(line);
                outLines.push(line);
            }

            function walk(el) {
                if (!el || el.nodeType !== Node.ELEMENT_NODE) return;
                const tag = el.tagName;
                if (DROP_TAGS.has(tag) || isHidden(el) || isChromeContainer(el)) return;

                if (tag === 'TABLE') {
                    emit(serializeTable(el));
                    return;
                }

                if (HEADING_TAGS.has(tag)) {
                    const content = normalizeInlineWhitespace(serializeInline(el));
                    if (content) {
                        const hTag = tag.toLowerCase();
                        emit(`<${hTag}>${content}</${hTag}>`);
                    }
                    return;
                }

                // Standalone <a href> (handle links wrapping headings or cards cleanly)
                if (tag === 'A') {
                    const innerHeading = el.querySelector('h1, h2, h3, h4');
                    const href = cleanHref(el);
                    if (innerHeading && !isHidden(innerHeading)) {
                        const hTag = innerHeading.tagName.toLowerCase();
                        const text = (innerHeading.innerText || '').replace(/\s+/g, ' ').trim();
                        if (text) {
                            emit(href
                                ? `<${hTag}><a href="${escapeAttr(href)}">${escapeHtml(text)}</a></${hTag}>`
                                : `<${hTag}>${escapeHtml(text)}</${hTag}>`);
                            return;
                        }
                    }
                    if (el.querySelector('p, table, ul, ol')) {
                        for (const child of el.children) walk(child);
                        return;
                    }
                    const label = (el.innerText || el.getAttribute('aria-label') || '').replace(/\s+/g, ' ').trim();
                    if (label && href) {
                        emit(`<a href="${escapeAttr(href)}">${escapeHtml(label)}</a>`);
                    }
                    return;
                }

                if (BLOCK_PROSE_TAGS.has(tag)) {
                    if (el.querySelector('p, h1, h2, h3, h4, table')) {
                        for (const child of el.children) walk(child);
                        return;
                    }
                    const content = normalizeInlineWhitespace(serializeInline(el));
                    if (content) {
                        emit(`<p>${content}</p>`);
                    }
                    return;
                }

                if (tag === 'LI') {
                    if (el.querySelector('h1, h2, h3, h4, p, table, ul, ol')) {
                        for (const child of el.children) walk(child);
                        return;
                    }
                    const content = normalizeInlineWhitespace(serializeInline(el));
                    if (content) {
                        emit(`<li>${content}</li>`);
                    }
                    return;
                }

                if (CONTROL_TAGS.has(tag)) {
                    emit(serializeControl(el));
                    return;
                }

                // Leaf <div>/<span> holding direct prose when a site avoids <p> tags
                if ((tag === 'DIV' || tag === 'SPAN' || tag === 'SECTION' || tag === 'ARTICLE') && el.children.length === 0) {
                    const txt = (el.innerText || '').replace(/\s+/g, ' ').trim();
                    if (txt.length >= 40) {
                        emit(`<p>${escapeHtml(txt)}</p>`);
                    }
                    return;
                }

                const children = (el.shadowRoot && el.children.length === 0)
                    ? el.shadowRoot.children
                    : el.children;
                for (const child of children) {
                    walk(child);
                }
            }

            walk(document.body);
            if (outLines.length === 0) return "";
            return "<html>\n<body>\n" + outLines.join("\n") + "\n</body>\n</html>";
        })()
        """

        for attempt in range(max_retries):
            try :
                mcp_response = await self._mcp_session.call_tool("execute_script", {
                    "instance_id" : self.browser_instance_id,
                    "script" : js_script
                })
            except Exception as e :
                self.logger.warning(f"get_sanitized_content: MCP transport error on attempt {attempt + 1}: {e}")
                mcp_response = None
            if mcp_response is None : return None
            # --- 1. IF JS FAILED / PAGE CRASHED ---
            if getattr(mcp_response, "isError", False) :
                self.logger.warning(f"get_sanitized_content: JS/Page error on attempt {attempt + 1} of {max_retries}.")
                if attempt < max_retries - 1:
                    self.logger.warning("Triggering native page reload...")
                    mcp_response = await self._mcp_session.call_tool("reload_page", {"instance_id" : self.browser_instance_id})
                    if not mcp_response : return None
                continue
            # --- 2. IF SUCCESSFUL ---
            if mcp_response and getattr(mcp_response, "content", None):
                raw_content = mcp_response.content[0].text.strip()
                if raw_content.startswith("{") and raw_content.endswith("}"):
                    try:
                        meta = json.loads(raw_content)
                        if isinstance(meta, dict):
                            if "file_path" in meta:
                                file_path = meta["file_path"]
                                if os.path.exists(file_path):
                                    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                                        file_text = f.read()
                                    try:
                                        envelope = json.loads(file_text)
                                    except json.JSONDecodeError:
                                        return file_text
                                    meta = envelope.get("data", envelope) if isinstance(envelope, dict) else envelope
                                else:
                                    self.logger.error(f"Offloaded file not found at {file_path}")
                                    return ""
                            res = meta.get("result", meta) if isinstance(meta, dict) else meta
                            if isinstance(res, dict) and "value" in res:
                                res = res["value"]
                            if isinstance(res, str):
                                return res
                            return ""
                    except json.JSONDecodeError:
                        pass
                return raw_content
        # --- 3. IF ALL RETRIES FAIL ---
        self.logger.error(f"Failed to extract sanitized content after {max_retries} attempts. Returning empty page.")
        return ""

    # The routine to collect all links on a page
    async def get_links(self) -> Optional[Set[str]] :
        js_script = """
        (() => {
            const links = Array.from(document.querySelectorAll('a[href]'));
            const webPageRegex = /\\.(png|jpe?g|gif|svg|webp|pdf|zip|tar|gz|mp3|mp4|css|js)$/i;
            // LINK NORMALIZATION - why in-page anchors and fragments are dropped:
            // the browser resolves a.href against the current page, so an in-page
            // anchor (<a href="#navigation-container">) becomes <current page>#fragment
            // - the SAME document, not a new one. Left unnormalized such links
            // pollute the link set with near-duplicates (page.html#frag vs
            // page.html) and the visited set (exact-string dedupe) fails, so the
            // same page gets crawled and saved twice. Rules applied below:
            //   1. drop in-page anchors (raw href starting with '#');
            //   2. drop empty hrefs (they resolve to the current page URL);
            //   3. strip the fragment from the rest (page#frag == page).
            // CAVEAT / FUTURE-PROOFING: this is correct for classic server-rendered
            // pages, where a fragment never changes the document. If some site ever
            // uses fragments/tags to lead to different pages (SPA hash-routing,
            // page#view-a vs page#view-b rendering different content), that host's
            // fragments would need to be PRESERVED here instead of stripped.
            return links
                .filter(a => {
                    const rawHref = a.getAttribute('href') || '';
                    // Rule 1+2: not a new document (in-page anchor / empty href)
                    return rawHref !== '' && !rawHref.startsWith('#');
                })
                .map(a => {
                    try {
                        const url = new URL(a.href);
                        // Ensure it's a web protocol and doesn't match static asset extensions
                        if (url.protocol !== 'http:' && url.protocol !== 'https:') return null;
                        if (webPageRegex.test(url.pathname)) return null;
                        // Rule 3: canonical same-document form (query string kept -
                        // it can change the document; the fragment does not)
                        url.hash = '';
                        return url.toString();
                    } catch (e) {
                        return null;
                    }
                })
                .filter(u => u !== null);
        })()
        """
        # JavaScript snippet to filter for actual web pages, ignoring typical asset
        # files. It also NORMALIZES links (see the in-script comment): in-page
        # anchors (#fragment) and empty hrefs are dropped, and fragments are
        # stripped from the rest, because page#frag is the same document as page -
        # this is what dedupes the link/visited sets (the duplicate-save bug:
        # ...043254178.html#navigation-container vs ...043254178.html). The
        # decision assumes fragments never change the document; if a host ever
        # uses them to select different pages, preserve fragments for that host.
        # Direct single-round-trip session call - bypasses __call__ on purpose (no
        # pre-flight probe: that is the speedup, this runs inside a poll loop and
        # the probe would double every round-trip). instance_id is injected
        # explicitly; any transport failure is tolerated and reported as None -
        # the caller treats that as "no links yet" and polls again.
        try :
            mcp_response = await self._mcp_session.call_tool("execute_script", {
                "instance_id" : self.browser_instance_id,
                "script" : js_script
            })
        except Exception as e :
            self.logger.warning(f"get_links: MCP transport error: {e}")
            return None
        if mcp_response is None or getattr(mcp_response, "isError", False) :
            return None
        try :
            data = json.loads(mcp_response.content[0].text)
        except (json.JSONDecodeError, AttributeError, IndexError, TypeError) as e :
            self.logger.warning(f"get_links: unexpected result payload: {e}")
            return None
        # Tolerate both result shapes: the stealth-browser-mcp envelope
        # {"result": [{"value": <url>}, ...]} and a plain list of URL strings.
        items = data.get("result", []) if isinstance(data, dict) else data
        if not isinstance(items, list) :
            return None
        links = set()
        for item in items :
            value = item.get("value") if isinstance(item, dict) else item
            if isinstance(value, str) and value :
                links.add(value)
        return links or None


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
        10: ERROR_MCP_BROWSER_NOT_EXISTS - No managed browser, call create_managed_browser first
        11: ERROR_MCP_TOOL_CALL - Failed to call an MCP tool
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
        # Type hints assumed from external imports like MultiServerMCPClient, BrowserProxy
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
                await asyncio.wait_for(self._mcp_session.call_tool("close_instance", {"instance_id" : self.managed_browser.browser_instance_id }), timeout=15.0)
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
                - ERROR_MCP_SESSION_STOP (6): No active session
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
            # Validate the spawn response: a missing instance_id would create a
            # broken proxy (assert crash / instance_id: None sent to the server).
            instance_id = data.get("instance_id") if isinstance(data, dict) else None
            if not instance_id:
                raise ValueError(f"spawn_browser response has no instance_id: {data}")
            self._managed_browser = BrowserProxy(logger=self.logger, _mcp_session=self._mcp_session, browser_instance_id=instance_id)
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
                - ERROR_MCP_BROWSER_NOT_EXISTS (10): No managed browser
                - ERROR_MCP_TOOL_CALL (11): Failed to execute tool
        """
        if self._managed_browser is None:
            return self.ERROR_MCP_CODES_NUMS["ERROR_MCP_BROWSER_NOT_EXISTS"]
        try:
            # Assumes BrowserProxy is implemented dynamically with __call__
            result = await self._managed_browser(tool_name, args=arguments)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to call {tool_name} mcp tool. Error: {e}")
            return self.ERROR_MCP_CODES_NUMS["ERROR_MCP_TOOL_CALL"]
        return self.ERROR_MCP_CODES_NUMS["ERROR_MCP_OK"]


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

