import asyncio
from contextlib import asynccontextmanager
import re
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, RemoveMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.sessions import ClientSession
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition


@asynccontextmanager
async def managed_browser(session):
    print("Spawning managed browser instance...")
    result = await session.call_tool("spawn_browser", {"headless": False})

    # 1. Get the raw content
    content_text = result.content[0].text if result.content else str(result)

    # 2. Extract ONLY the ID string
    try:
        # Assuming the output is JSON
        data = json.loads(content_text)
        instance_id = data.get("instance_id")
    except json.JSONDecodeError:
        # Fallback if it's just a raw string
        instance_id = content_text.strip()

    print(f"Browser ready. Instance ID: {instance_id}")
    yield instance_id


# The genetic agentic scrapper
async def run_stealth_graph(url_string : str = "https://ca.finance.yahoo.com/") : #
    server_config = {
        "stealth-browser": {
            "transport": "stdio",
            "command": "/home/ayakovenko/nila/stealth-browser-mcp/bin/python3",
            "args": ["/home/ayakovenko/nila/stealth-browser-mcp/src/server.py"],
            "env": {
                # 1. ToDo implement automatic display detection
                "DISPLAY": ":1",

                # 2. Point the underlying automation library to our new wrapper
                "CHROME_PATH": "/home/ayakovenko/nila/flatpak-chromium.bash",
                "BROWSER_EXECUTABLE_PATH": "/home/ayakovenko/nila/flatpak-chromium.bash",
                "PUPPETEER_EXECUTABLE_PATH": "/home/ayakovenko/nila/flatpak-chromium.bash"
            }
        }
    }

    mcp_client = MultiServerMCPClient(server_config)
    async with mcp_client.session("stealth-browser") as mcp_session :
        tools = await load_mcp_tools(mcp_session)
        tools_summary = "\n".join([f"* Tool: {t.name}\nDescription: {t.description}" for t in tools]) + "\n" # To expose tool for the thinking modle

        # The lifecycle is now fully managed here
        async with managed_browser(mcp_session) as browser_instance_id :

            # Now launch llm and start exploring
            llm_base = ChatOpenAI(
                base_url="http://ifo4:8000/v1",
                api_key="alex_llm_qwen",
                model_name="QuantTrio/Qwen3.6-27B-AWQ",
                temperature=0.1,
                #base_url="http://localhost:8000/v1",
                #api_key="not-needed",
                #model_name="neuralmagic/DeepSeek-R1-Distill-Qwen-14B-quantized.w4a16",
                #temperature=0.1,
                #base_url="http://ifo4:8000/v1",
                #api_key="alex_llm_qwen",
                #model_name="neuralmagic/DeepSeek-R1-Distill-Qwen-32B-quantized.w4a16",
                #temperature=0.1,
            )
            #llm_tool = ChatOpenAI(
            #    base_url="http://ifo4:8000/v1",
            #    api_key="alex_llm_qwen",
            #    #model_name="neuralmagic/DeepSeek-R1-Distill-Qwen-32B-quantized.w4a16",
            #    model_name="QuantTrio/Qwen3.6-35B-A3B-AWQ",
            #    temperature=0.1,
            #)
            llm_tool = llm_base
            llm_tool_with_tools = llm_tool.bind_tools(tools)

            # ----------------------------   A G E N T S   ----------------------------

            import json
            import re
            from typing import TypedDict, Optional, List, Set, Dict, Any, Annotated
            from dataclasses import dataclass, field, replace
            from clean_dom import get_browser_state

            @dataclass
            class ExplorationHistory :                                       # The state structure
                # ---------- GRAPH TOPOLOGY ---------- #
                deep          : int = 1                                      # deepness of this node from root of the tree
                url           : str = "127.0.0.1"                            # The URL associated with this node
                context       : List[str] = field(default_factory=list)      # Messages from the upper levels
                # ---------- MOMENTUM/REFLECTION DATA ---------- #
                instance_id   : str = ""                                     # Browser instance id
                browser_state : str = ""                                     # Browser state
                links         : List[Set[str]] = field(default_factory=list) # Stack of unique links on the current page
                history       : List[str]      = field(default_factory=list) # History of actions at the current level to get the full content
                recovery_logs : str = ""                                     # Log of the current error (if there is any)
                recovery_nums : int = 0
                # UTILITY
                mcp_session   : Optional[ClientSession] = None
                tools_summary : str = ""
                # ---------- STATISTICS AT THIS LAYER ---------- #
                visited_links : Set[str]       = field(default_factory=set ) # Visited links during exploration
                saved         : int = 0                                      # Amount of saved documents downstream

                def __repr__(self) -> str :
                    return (f"ExplorationHistory(deep={self.deep}, url='{self.url}', "
                            f"saved={self.saved}, visited_links={self.visited_links})")

                def pretty_print(self) -> None :
                    print(f"\n--- ExplorationHistory Node: Depth {self.deep} ---")
                    print(f"URL: {self.url}")
                    print(f"Stats: Saved={self.saved}, Visited={len(self.visited_links)}")
                    print(f"Instance ID: {self.instance_id}")
                    print(f"Thought Data Items: {len(self.context)}")
                    if self.recovery_logs:
                        print(f"Recovery Log: {self.recovery_logs}")
                    print("---------------------------")

            class AgenticUpdate(TypedDict) :                           # The updates structure
                cmd   : str                                            # Command type
                agent : str                                            # Agent who issued the update
                data  : ExplorationHistory                             # Update itself

            # Define custom reducer for web-tree traversal
            # Note. Langgraph expects that multiple updates can come together, thus it delivers updates list. Not our case though.
            def exploration_stack_manager(current_history_stack : ExplorationHistory, update_instruction : Optional[AgenticUpdate]) -> ExplorationHistory :
                # Formal block
                cmd   = update_instruction.get("cmd",   "END")
                agent = update_instruction.get("agent",  None)
                data  = update_instruction.get("data",     "")
                # Switch based on actor type
                if agent == "START" : # Initialization call
                    updated_history_stack = data  # Initialization
                else                 :
                    updated_history_stack = replace(current_history_stack)  # Framework 'safety' policy
                    if   agent ==   "access_agent" : # Planner agent logic
                        # 0. END: move upstream updating statistics
                        if   command == "END"     : # It's nothing left to do at this level, exiting
                            old_page = updated_history_stack.pop()
                            if len(updated_history_stack) :
                                curr_page = updated_history_stack[-1]
                                curr_page["saved"] += old_page["saved"]
                                curr_page["visited"] += old_page["visited"]
                                curr_page["data"].append(f"Exploration of the downstream url {old_page["url"]} is completed."
                                                         f"It resulted in {old_page["saved"]} saved documents and {old_page["visited"]} visited pages.")
                        elif command == "EXPLORE" :
                            updated_history_stack[-1]["data"].append(data)
                        else : raise NotImplementedError(f"Unknown command {command} of the planning agent is spotted. Failing...")
                    elif agent ==  "executor_agent" : # Execution agent logic
                        raise NotImplementedError(f"Unknown command {command} of the executor agent is spotted. Failing...")
                    elif agent == "validator_agent" : # Validation agent logic
                        raise NotImplementedError(f"Unknown command {command} of the validator agent is spotted. Failing...")
                    else : raise NotImplementedError(f"Unknown agent {agent} is spotted. Failing...")

                return updated_history_stack

            class AgenticState(TypedDict):
                # Use our intelligent stack manager
                exploration_history : Annotated[ExplorationHistory, exploration_stack_manager]

            # The agent to get furll access to a web page content
            async def access_agent(state : AgenticState) -> Optional[Dict[str, AgenticUpdate]] :
                exploration_history = state["exploration_history"]

                # 1. Get browser state
                if not len(exploration_history.browser_state) :
                    exploration_history.browser_state = await get_browser_state(exploration_history.mcp_session, exploration_history.instance_id)

                # 1. Safely format the history for the prompt
                context_text = ""
                context_text += "\n".join(f"{i}. {s}" for i, s in enumerate(exploration_history.history, start=0))
                history_text = f"0. Opened the page {exploration_history.url}"
                history_text += "\n".join(f"{i}. {s}" for i, s in enumerate(exploration_history.history, start=1))

                # 2. The Smart Driver Prompt
                access_agent_prompt = (
                    f"# ROLE & CONTEXT\n"
                    f"You are the planning component of an automated stock-trading document scraper.\n"
                    f"Executing tree-search exploration.\n"
                    f"* **Browser instace id:** {exploration_history.instance_id}\n"
                    f"* **Current Active Node (URL):** {exploration_history.url}\n"
                    f"* **Diving Depth:** {exploration_history.deep}\n"
                    f"* **Exploration Context:** {context_text}\n\n"
    
                    f"# OBJECTIVE\n"
                    f"Investigate the `CURRENT BROWSER STATE` and `ACTIONS HISTORY` below. Your goal is to determine the actions required (e.g., logging in, closing pop-ups, bypassing cookie banners) to obtain more content of the current web page.\n"
                    f"Please pick one principal action (like sign in, solve captcha, close a popup etc) and write a detailed instruction on how to execute it using the available tools.\n"
                    f"The execution agent will execute your instructions and you'll review new state of browser again to pick another action if there is someting left. Just be systematic and pick whatever you think can be helpful go get more content access.\n"
                    f"Don\'t hesitate to spend tokens to thinking and generative page analysis is highly supported - please do your best to end up with good actionable plan.\n"
                    # f"DO NOT ATTEPMT TO LOGIN AS YOU HAVE NO NAME / PASSWORD CREDENTIALS FOR THE GIVEN URL!\n"
                    # f"Spend as many tokens as needed on thinking about the page layout and JavaScript behaviors. Arrive at a precise step-by-step execution plan.\n\n"
                    f"CRITICAL! If there is SIGN IN form or button on the page then you YOUR PRIMARY REQUIRED ACTION IS TO SIGN IN unless you already signed. You need to sign in with google using account \'alex.ifowonco@gmail.com\'\n\n"
    
                    f"# TOOLKIT\n"
                    f"You ahn operate with the following tools to carry out actions in the browser and change its state toward more content presented.\n"
                    f"{exploration_history.tools_summary}"
     
                    f"## CRITICAL STRUCTURAL INSTRUCTIONS:\n"
                    f"- If an action is required for a detailed step-by-step instruction on how and which tool to use in order to accomplish the action.\n"
                    f"- If the content full access is already obtained and no actions are needed simply reply with the exact text after thinking: 'ACCESS_OBTAINED'\n\n"
    
                    f"# DATA \n"
                    f"## CURRENT BROWSER STATE\n"
                    f"```html\n"
                    f"{exploration_history.browser_state}\n"
                    f"```\n\n"
    
                    f"## ACTIONS HISTORY\n"
                    f"```text\n"
                    f"{history_text}\n"
                    f"```\n\n"
                )
                # Invoke the model
                response = await llm_base.ainvoke([SystemMessage(content=access_agent_prompt),
                                                         HumanMessage(content="Please execute the requested task. Use your thinking process to analyze the page.")],
                                                  config={"configurable": { "chat_template_kwargs": {"enable_thinking" : True, "reasoning_effort" : "high"} }})
                raw_content = response.content
                # 3. Postprocessing
                match = re.search(r'</think>\s*(.*)$', raw_content, re.DOTALL)
                next_actions_text = match.group(1) if match else raw_content
                if next_actions_text ==  'ACCESS_OBTAINED' :
                    # No tools call
                    exploration_history.recovery_logs = []
                else                   :
                    # Call the tools
                    exploration_history.recovery_logs = [next_actions_text,]
                # Formalize updates
                return None


            async def execute_agent(state : AgenticState) -> Optional[Dict[str, AgenticUpdate]] :
                exploration_history = state["exploration_history"]
                assert len(exploration_history.browser_state) , "Browser state should be known here!"
                assert len(exploration_history.recovery_logs) , "The recovery_nums can not be empty!"
                #  The Smart Driver Prompt
                execute_agent_prompt = (
                    f"# ROLE\n"
                    f"You are the Tactical Executor for an automated web-scraping agent. Your sole purpose is to execute the immediate next tool call based on the provided PLAN.\n\n"
                    f"# YOUR TASKS\n"
                    f"  1. Inspect the 'Immediate Next Step' defined in the PLAN.\n"
                    f"  2. Cross-reference the selector with the CURRENT BROWSER STATE. If the selector is stale, DYNAMICALLY CORRECT IT to match the current HTML structure.\n"
                    f"  3. Execute exactly ONE tool call that advances the plan.\n\n"
                    f"# CONSTRAINTS\n"
                    f"  - Output ONLY the tool call. NO conversational filler.\n"
                    f"  - If you cannot perform the step, explain why briefly in the tool output.\n\n"
                    f"# PLAN\n{chr(10).join(exploration_history.recovery_logs)}\n\n"
                    f"# CURRENT BROWSER STATE\n```html\n{exploration_history.browser_state}\n```\n"
                )
                # Invoke the model
                response = await llm_tool_with_tools.ainvoke([SystemMessage(content=execute_agent_prompt),
                                                              HumanMessage(content="Please invoke the best tool to execute the next action.")],
                                                             config={"configurable": { "chat_template_kwargs": {"enable_thinking": True} }})
                # Executing the command
                if response.tool_calls :
                    # Extract the first tool call (since we constrained it to exactly ONE)
                    tool_call = response.tool_calls[0]
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    try :
                        tool_result = await exploration_history.mcp_session.call_tool(tool_name, arguments=tool_args)
                        execution_report = f"EXECUTED: {tool_name} | ARGS: {tool_args} | RESULT: {tool_result}"
                    except Exception as e:
                        # Package the crash for the verifier so it can issue a recovery hint
                        execution_report = f"FAILED TO EXECUTE: {tool_name} |  ARGS: {tool_args} | ERROR: {str(e)}"
                else:
                    # The model decided no tools were needed (or hallucinated)
                    execution_report = f"NO ACTION TAKEN. Model Output: {response.content}"
                # Formalize updates
                return [{ "exploration_history": AgenticUpdate({"cmd" : "VERIFY", "agent" : "execution_agent", "data" : execution_report }) }]


            async def validate_agent(state : AgenticState) -> Optional[Dict[str, AgenticUpdate]] :
                last_msg = state["messages_current"][-1].content.lower()

                # Simple validation logic
                if "error" in last_msg or "failed" in last_msg:
                    new_attempts = state.get("recovery_attempts", 0) + 1
                    return {"recovery_attempts": new_attempts}

                # If successful, we might update the browser state
                return {"browser_state_current": "Successfully executed action. Page loaded."}

            async def analyse_agent(state: AgenticState) -> Dict[str, Any]:
                # The executor only looks at the current thread and the tools
                executor_prompt = f"Execute the next step for URL: {state['user_query']}. Current browser state: {state['browser_state_current']}"

                # In a real setup, this LLM is bound to your stealth-browser-mcp tools
                messages_for_executor = state.get("messages_current", []) + [HumanMessage(content=executor_prompt)]
                response = await llm_with_tools.ainvoke(messages_for_executor)

                return {
                    "messages_current" : [response]
                }

            # The node extracts one link from current page
            async def return_node(state : AgenticState) -> AgenticUpdate :
                current_history_stack = state["exploration_history"]
                if not len(current_history_stack.links[-1]) :
                    # Return to the upper level
                    updated_history_stack = replace(current_history_stack)
                    updated_history_stack.deep -= 1
                    updated_history_stack.url = ""
                    updated_history_stack.context.pop()
                    updated_history_stack.links.pop()
                    return AgenticUpdate(command="END", agent="RETURN", data=updated_history_stack)
                else                    :
                    # Explore a new link
                    return AgenticUpdate(command="EXPLORE", agent="RETURN", data=next(iter(current_history_stack.links[-1])))

            # --- Graph Assembly ---

            #             .-----------------------------------.
            #             V                                   |                          |
            # START -> access_a -+-> executor --> validator --|                          |
            #                    |       ^                    |     |  n_runs == 0 or n_runs
            #                    |       '--------------------'     |
            #                    '---------------------------------> analyse_a --> RETURN -+-> End                                       | Yes       |

            builder = StateGraph(AgenticState)
            # Define nodes
            builder.add_node(  "access_agent",   access_agent)
            builder.add_node( "execute_agent",  execute_agent)
            builder.add_node("validate_agent", validate_agent)
            builder.add_node( "analyse_agent",  analyse_agent)
            builder.add_node(        "RETURN",   return_node )
            # Define edges
            builder.add_edge(START, "access_agent")
            builder.add_conditional_edges("access_agent", lambda state : str(len(state["exploration_history"].recovery_logs) == 0), { "False" : "execute_agent", "True" : "analyse_agent" } )
            builder.add_edge("execute_agent", "validate_agent")
            builder.add_conditional_edges("validate_agent", lambda state : str(state["exploration_history"].recovery_nums  == 0), { "False" : "execute_agent", "True" : "access_agent" } )
            builder.add_conditional_edges("analyse_agent",  lambda state : str(len(state["exploration_history"].links[-1]) == 0), { "False" : "access_agent", "True" : "RETURN" })
            builder.add_conditional_edges("RETURN",         lambda state : "ACCESS" if len(state["exploration_history"].links[-1]) != 0 else (
                                                 "RETURN" if state["exploration_history"].deep else "END" ) , { "ACCESS" : "access_agent", "RETURN" : "RETURN", "END" : END })
            # Finalize
            graph = builder.compile()


            # Go to the root URL
            result = await mcp_session.call_tool("navigate", arguments={
                    "instance_id" : browser_instance_id,
                    "url"         : url_string,
                    "wait_until"  : "networkidle"  # Ensures the page is fully loaded before continuing
                    }
            )
            assert not getattr(result, "isError", False), f"Failed to open url {url_string} due to {result}"
            print(f"\n[Target Mission Dispatched]: {url_string}\n")

            # Launch agentic scrapper
            async for event in graph.astream({"exploration_history" : AgenticUpdate({
                                               "cmd"   : "EXPLORE",
                                               "agent" : "START",
                                               "data"  : ExplorationHistory(
                                                   deep=1,
                                                   url=url_string,
                                                   context=[f"Starting explorationof url {url_string}"],
                                                   instance_id=browser_instance_id,
                                                   browser_state="",
                                                   links=[],
                                                   history=[],
                                                   recovery_logs="",
                                                   mcp_session=mcp_session,
                                                   tools_summary=tools_summary,
                                                   visited_links=set(),
                                                   saved=0,
                                                   ),
                                               },),
                                             },
                                             stream_mode="values" ) :
                print("\n==================== STREAM EVENT ====================")

                # 1. Safely print the strategy/planning history if it just updated
                if "exploration_history" in event and event["exploration_history"]:
                    print("[Global Planner State Changed]")
                    event["exploration_history"].pretty_print()

                # 2. Safely print the executor thread if it just updated
                if "messages_current" in event and event["messages_current"]:
                    print("[Executor Tool State Changed]")
                    event["messages_current"].pretty_print()

                # 3. Print the tracking flags so you can see your tree navigation state
                print(f"-> Current Status: {event.get('current_status')}")
                print(f"-> Current URL:    {event.get('current_url')}")
                print("======================================================\n")

# The main entry point
if __name__ == "__main__" :
    asyncio.run(run_stealth_graph())
