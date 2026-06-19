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

import re
import os, uuid
from datetime import datetime
import asyncio

from typing import TypedDict, Optional, List, Set, Dict, Any, Annotated
from dataclasses import dataclass, field, replace

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, RemoveMessage
from langgraph.graph import StateGraph, START, END

from lmdb_driver import LMDB
from web_tools import BrowserProxy, MCPBrowserRunner, clean_dom, is_same_registered_domain
MAX_DEEP = 3  # Maximal diving deepness


# Now launch llm and start exploring
llm_base = ChatOpenAI(
    #b ase_url="http://ifo4:8000/v1",
    # api_key="alex_llm_qwen",
    # model_name="QuantTrio/Qwen3.6-27B-AWQ",
    # temperature=0.1,
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    model_name="neuralmagic/DeepSeek-R1-Distill-Qwen-14B-quantized.w4a16",
    temperature=0.1,
    # base_url="http://ifo4:8000/v1",
    # api_key="alex_llm_qwen",
    # model_name="neuralmagic/DeepSeek-R1-Distill-Qwen-32B-quantized.w4a16",
    # temperature=0.1,
)


@dataclass
class AgenticStateDataclass :  # The state structure
    # ---------- GRAPH TOPOLOGY ---------- #
    deep          : int = 1  # deepness of this node from root of the tree
    url_new       : Optional[str] = None # Next url to visit (pending)
    url           : List[str] = field(default_factory=list)  # The URL associated with this node
    context       : List[str] = field(default_factory=list)  # Messages from the upper levels
    # ---------- MOMENTUM/REFLECTION DATA ---------- #
    browser_state : List[str] = field(default_factory=list)  # Browser state
    web_links     : List[str] = field(default_factory=list)  # Stack of unique links on the current page
    # ----------  UTILITY  ---------- #
    browser       : Optional['BrowserProxy'] = field(init=True, default=None) # The proxy browser
    db            : Optional['LMDB']         = field(init=True, default=None) # The LMDB
    # ---------- STATISTICS AT THIS LAYER ---------- #
    visited       : int = 0  # Amount of visited urls in this stream
    saved         : int = 0  # Amount of saved documents downstream

    def __len__(self) -> int :
        state_len = { len(self.browser_state), len(self.web_links), len(self.url), len(self.context) }
        assert len(state_len) == 1 , "ERROR: Agentic state is desynchronized!"
        return state_len.pop()

    def pop_state(self) -> int :
        assert len(self) == self.deep , "ERROR: Agentic state is desynchronized!"
        self.browser_state.pop()
        self.web_links.pop()
        self.url.pop()
        self.context.pop()
        self.deep -= 1
        return self.deep

    def __repr__(self) -> str:
        return (f"AgenticStateDataclass(deep={self.deep}, url=\'{self.url}\', saved={self.saved}, visited={self.visited}")

    def pretty_print(self) -> None:
        print(f"\n--- ExplorationHistory Node: Depth {self.deep} ---")
        print(f"URL: {self.url}")
        print(f"Thought Data Items: {len(self.context)}")
        print(f"Stats: Saved={self.saved}, Visited={self.visited}")
        print("---------------------------")

class AgenticUpdate(TypedDict):  # The updates structure
    cmd   : str  # Command type
    agent : str  # Agent who issued the update
    data  : AgenticStateDataclass  # Update itself

# Define custom reducer for web-tree traversal
# Note. Langgraph expects that multiple updates can come together, thus it delivers updates list. Not our case though.
def agentic_state_manager(agentic_state : AgenticStateDataclass, update_instruction: Optional[AgenticUpdate]) -> AgenticStateDataclass :
    # Formal block
    cmd   = update_instruction.get("cmd", "END")
    agent = update_instruction.get("agent", None)
    data  = update_instruction.get("data", "")
    if agent == "START" :
        return data
    else                :
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[{current_time}] INFO: progress: {agentic_state.visited} URL: {agentic_state.url[-1]}")
        agentic_state.visited += 1
        return agentic_state

class AgenticState(TypedDict):
    # Use our intelligent stack manager
    agentic_state : Annotated[AgenticStateDataclass, agentic_state_manager]


# This routine reads the url and extract the lists of links in there
async def DIG_routine(agentic_state : AgenticState) -> Optional[Dict[str, AgenticUpdate]] :
    state      = agentic_state["agentic_state"]
    assert  state.browser is not None , "CRITICAL: The browser is not initialized yet!"
    browser    = state.browser
    assert state.url_new is not None , "CRITICAL: The url for exploration is missing!"
    # Go to the root URL
    logger.debug(f"visiting url {state.url_new}")
    (url_string, state.url_new) = (state.url_new, None)
    state.url.append(url_string)
    state.browser_state.append(None)
    state.web_links.append(None)
    state.context.append(None)
    state.deep += 1
    browser_location = await browser.navigate(url_string)
    if browser_location is not None :
        logger.debug(f"browser_locations is not None")
        state.context[-1] = f"Arrived at url {url_string}.\n"
        state.visited += 1
        browser_content = await browser.get_content()
        if browser_content is not None :
            state.browser_state[-1] = clean_dom(browser_content)
            if state.deep <= MAX_DEEP and is_same_registered_domain(state.url[0], url_string) :
                # --- Extract all webpage links ---
                web_links = await browser.get_links()
                if web_links is not None :
                    state.web_links[-1] = state.db.check_keys(web_links) # Filter visited links
                    state.context[-1] += f"Extracted {len(state.web_links[-1] )} uniquie web liks.\n"
                    logger.info(f"extracted {len(state.web_links[-1])} links from page {state.url[-1]}.")
                else                     :
                    state.context[-1] += f"Failed to extract web links at {len(state.web_links[-1] )}.\n"
    return None # This agent update its state himself - no branching is there


# This routine saves data to hdd in format ./data/YYYY-MM-DD/unique_uuid.md
DATA_PREFIX="./data/"
def save_story(story : str) -> str :
    date_str = datetime.now().strftime("%Y-%m-%d")
    save_dir = os.path.join(f"{DATA_PREFIX}", date_str)
    os.makedirs(save_dir, exist_ok=True)
    while True :
        unique_fname = f"{uuid.uuid4()}.md"
        file_path = os.path.join(save_dir, unique_fname)
        if not os.path.exists(file_path):
            break
    logger.debug(f"saving STORY into {file_path}")
    with open(file_path, "w", encoding="utf-8") as f :
        f.write(story)
    return file_path

# This agent iteratively rewrites story for saving
async def story_rewritting_agent(deep : int, base_url : str, curr_url : str, context_text : str, browser_state : str) -> str :
    # =========================================================================
    # STEP 1: THE DRAFTER (Exhaustive Markdown Generation)
    # =========================================================================
    draft_agent_prompt = (
        f"# ROLE & CONTEXT\n"
        f"You are a meticulous financial reviewer working on summarization and filtering news for future sentiment analysis.\n"
        f" * **Diving Depth:** {deep}\n"
        f" * **Base URL:** {base_url}\n"
        f" * **Current Active Node (URL):** {curr_url}\n"
        f" * **Exploration Context:** {context_text}\n\n"

        f"# OBJECTIVE\n"
        f"Rewrite the provided raw scraped \"STORY\" text into a highly detailed, comprehensive Markdown document.\n"
        f"Your mandate is NO DATA LOSS. Retain every mentioned entity, ticker symbol and the news event.\n"
        f"Remove all html formatting, tegs, etc it is not html any more, everything should be Markdown only.\n"
        f"Do not worry about making it concise. Focus on structuring the raw text logically using Markdown headers, lists, and bold text.\n"

        f"# \"STORY\": \n"
        f"```html\n"
        f"{browser_state}\n"
        f"```\n\n"
    )
    # Invoke the model
    drafter_response = await llm_base.ainvoke([SystemMessage(content=draft_agent_prompt),
                                               HumanMessage(
                                                   content="Please convert the raw html text into a structured Markdown document.  Use your thinking process to analyze the page to confirm that no details are lost during the converson.")],
                                                   config={"configurable": {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "high"}}})
    draft_content = re.sub(r'.*?</think>', '', drafter_response.content, flags=re.DOTALL).strip()
    logger.info(f"extracting agent processed the STORY")
    # =========================================================================
    # STEP 2: THE MASTER EDITOR (Clean & Polish)
    # =========================================================================
    editor_agent_prompt = (
        f"# ROLE & CONTEXT\n"
        f"You are a Lead Financial Editor finalizing a comprehensive market intelligence brief.\n\n"

        f"# OBJECTIVE\n"
        f"Take the provided detailed Markdown draft and enforce strict, professional formatting.\n"
        f"You must preserve the deep granularity of the data, but improve the narrative flow, merge duplicate stories and ensure a pristine visual hierarchy.\n\n"

        f"# TARGET STRUCTURE\n"
        f"1. **Executive Market Summary**\n"
        f"2. **Detailed Sector & Macro Movements**\n"
        f"3. **Exhaustive Ticker/Asset Breakdown** (List every asset mentioned with its corresponding news/data)\n"
        f"4. **Actionable Traversal Vectors** (What concepts/links the crawler should target next)\n\n"

        f"Output ONLY the final, polished Markdown after thinking."

        f"# DETAILED DRAFT:\n"
        f"```html\n"
        f"{draft_content}\n"
        f"```\n\n"
    )
    editor_response = await llm_base.ainvoke([
                         SystemMessage(content=editor_agent_prompt),
                         HumanMessage(content="Please polish the raw.  Use your thinking process to analyze the page to confirm that no details are lost during the converson.")
                         ], config={"configurable": {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "high"}}})
    markdown_story = re.sub(r'.*?</think>', '', editor_response.content, flags=re.DOTALL).strip()
    logger.info(f"editor agent processed the STORY")
    return markdown_story

# The analysis agent
async def ANL_llmgent(agentic_state : AgenticState) -> Optional[Dict[str, AgenticUpdate]] :
    state = agentic_state["agentic_state"]
    assert state.browser_state is not None , "There should be a browser state to handle with LLM!"
    # 1. Safely format the history for the prompt
    context_text = ""
    context_text += "\n".join(f"{i}. {s}" for i, s in enumerate(state.context, start=0))
    # 2. The Analyser Prompt
    access_agent_prompt = (
        f"# ROLE & CONTEXT\n"
        f"You are the web scrapper handling tree-search exploration of STOCKS-related content.\n"
        f"You are looking for news pages to assist following agent in making trading decisions, but the route typically goes via aggregator pages.\n"
        f"* **Diving Depth:** {state.deep}\n"
        f"* **Current Active Node (URL):** {state.url[-1]}\n"
        f"* **Exploration Context:** {context_text}\n\n"

        f"## OBJECTIVE\n"
        f"Analyze the provided browser state and action history to classify the page for the navigation controller.\n"
        f"Your goal is to *CLASSIFY* if the page contains actionable news or if it acts as a directory for further traversal.\n\n"

        f"## CLASSIFICATION CRITERIA\n"
        f"Put \"STORY\" if the page is a single, primary financial or corporate news article.\n"
        f"Put \"AGGREGATOR\"  if the page is a list/index of multiple stories or links (e.g., news feeds, homepages).\n"
        f"Put \"IRRELEVANT\" if the page contains (next to) no information on stock trading, economy, finances etc.\n"
        f"Note that IRRELEVANT STORY or IRRELEVANT AGGREGATOR are simply \"IRRELEVANT\" web pages.\n"
        f"If the page containt irrelevant to stock training information then put both \"STORY\" and \"AGGREGATOR\" to \"FALSE\".\n"
        f"CRITICAL. PLEASE REPLY WITH ONLY ONE WORD AFTER THINKING TOKENS: \"STORY\" | \"AGGREGATOR\" | \"IRRELEVANT\".\n"
        f"Depending on your classification the following qwery will either focus on story extraction or brief summarization and algorithmic traversal of the present links.\n\n"

        f"# DATA \n"
        f"## CURRENT BROWSER STATE\n"
        f"```html\n"
        f"{state.browser_state}\n"
        f"```\n\n"
    )
    # Do several classification attempts
    classification_attempts = 0
    classification_logs     = []
    response = await llm_base.ainvoke([SystemMessage(content=access_agent_prompt),
                                       HumanMessage(content=f"Please execute the requested task. Use your thinking process to analyze the page.\n")],
                                      config={"configurable": {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "high"}}})
    raw_content = response.content
    clean_response = re.sub(r'.*?</think>', '', raw_content, flags=re.DOTALL).strip()
    while (match := re.search(r'\b(STORY|AGGREGATOR|IRRELEVANT)\b', clean_response, re.IGNORECASE) ) is None and classification_attempts < 3 :
        # Invoke the model
        classification_attempts += 1
        classification_logs.append(f" - ATTEMPT: {classification_attempts} RESPONSE: {clean_response}\n")
        response = await llm_base.ainvoke([SystemMessage(content=access_agent_prompt),
                                           HumanMessage(
                                               content=(f"Please execute the requested task. Use your thinking process to analyze the page.\n"
                                                        f"Please better follow the requested output format! {classification_attempts} previous attempts had filed: {"\n".join(classification_logs)}")) ],
                                          config={"configurable": {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "high"}}})
        raw_content = response.content
        clean_response = re.sub(r'.*?</think>', '', raw_content, flags=re.DOTALL).strip()
    # Trigger the behaviour
    classification = match.group(0).upper()
    match classification  :
        case "STORY"      :
            logger.debug(f"AN agent classified the {state.url[-1]} page as STORY")
            story   = await story_rewritting_agent(state.deep, state.url[0], state.url[-1], context_text, state.browser_state[-1])
            if story is not None :
                state.context[-1] += f"The paage (url={state.url[-1]}) is a \"STORY\". I am  extracting it and keep traversing the links.\n"
                file_name = save_story(story)
                state.saved += 1
                state.db.put(state.url[-1], file_name[7:]) # Ommit ./data/ prefix
            else                 :
                state.context[-1] = None
                state.web_links[-1] = None
        case "AGGREGATOR" :
            logger.debug(f"AN agent classified the {state.url[-1]} page as AGGREGATOR")
            if state.deep == 2 : # Entry point aggregator
                state.context[-1] += f"The paage (url={state.url[-1]}) is an aggregator. I am  traversing it down for the individual news.\n"
            else               : # Discard external branches
                state.context[-1] = None
                state.web_links[-1] = None
            state.db.put(state.url[-1], "")
        case "IRRELEVANT" : # Discard irrelevant pages
            logger.debug(f"AN agent classified the {state.url[-1]} page as IRRELEVANT")
            state.context[-1] = None
            state.web_links[-1] = None
            state.db.put(state.url[-1], "")
        case _            : # Distcard failed pages
            logger.warning(f"ANL agent failed to classify the {state.url[-1]} page")
            state.context[-1] = None
            state.web_links[-1] = None
            state.db.put(state.url[-1], "")
    # Formalize updates
    state.browser_state[-1] = None # Free some memory
    return { "agentic_state": AgenticUpdate({"cmd"   : "END", "agent" : "ANL", "data"  : state.url[-1]}) } # Return traceble report


# The return agent, it sets url_new
async def RET_routine(agentic_state : AgenticState) -> Optional[Dict[str, AgenticUpdate]] :
    state = agentic_state["agentic_state"]
    assert state.deep == len(state) , "ERROR: Agentic state is desynchronized!"
    assert state.url_new is None , "ERROR: some url is still pending in RETurn routine!"
    if state.deep == MAX_DEEP : # Limit exploration reached
        logger.debug(f"RETurn agent reached max deep {state.deep}")
        state.pop_state()
    while state.deep :
        if len(state.web_links[-1]) :
            state.url_new = state.web_links[-1].pop()
            logger.debug(f"RETurn agent picked the new line {state.url_new}")
            break
        else                    :
            logger.debug(f"RETurn agent run out of links!")
            state.pop_state()
    return None  # This agent update its state himself - no branching is there


# Your isolated agent workflow. It only cares about using an active session/browser.
async def run_web_analysis_agent(browser : BrowserProxy, url_string: str, db : LMDB) -> None :
    print(f"Agent starting analysis on: url={url_string} using browser instance context: {browser.browser_instance_id}\n")

    # --- Graph Assembly ---

    #             .----------------------------.
    #             V                            |
    # START --> DIGin -+-> ANaLyse --> RETurn -+-> END
    #                  |                 ^^    |
    #                  '-----------------''----'

    builder = StateGraph(AgenticState)
    # Define nodes
    builder.add_node("DIG", DIG_routine)
    builder.add_node("ANL", ANL_llmgent)
    builder.add_node("RET", RET_routine)
    # Define edges
    builder.add_edge(START, "DIG")
    builder.add_conditional_edges("DIG", lambda state : str(state["agentic_state"].browser_state is None),{"False": "ANL", "True": "RET"})
    builder.add_edge("ANL", "RET")
    builder.add_conditional_edges("RET", lambda state : str(state["agentic_state"].deep == 0),            {"False": "DIG", "True":  END })
    # Finalize
    graph = builder.compile()

    # Launch agentic scrapper
    print("Launching agent execution...")
    final_state = await graph.ainvoke({
        "agentic_state": AgenticUpdate({
            "cmd": "EXPLORE",
            "agent": "START",
            "data": AgenticStateDataclass(
                deep=1,
                url_new=url_string,
                url=[url_string], # Set base url
                context=[f"Rooting exploration at url {url_string}"],
                browser_state=[None],
                web_links=[None],
                browser=browser,
                db=db,
                saved=0,
                visited=0,
            ),
        }),
    })
    print("\n==================== AGENT EXECUTION COMPLETE ====================")
    final_state.pretty_print()
    print("======================================================\n")



# The main routine
async def main() -> None:
    # Show the legend
    print(
        f"Running web analysis agent.\n"
        f"This agent assumes whole access to web page.\n"
        f"Its goal is to prepare documents for RAG + index search.\n"
    )
    # Wrap a database
    with LMDB(db_path="./data/stocks.db") as db :
        # Launch browser
        runner : MCPBrowserRunner = MCPBrowserRunner()
        # Launch agent
        await runner.run(run_web_analysis_agent, url_string="https://ca.finance.yahoo.com/", db=db)

if __name__ == "__main__":
    asyncio.run(main())
