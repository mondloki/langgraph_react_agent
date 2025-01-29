from dotenv import load_dotenv

load_dotenv()

from langchain_core.agents import AgentAction, AgentFinish
from state import AgentState
from langgraph.graph import StateGraph, END
from nodes import run_agent_reasoning_engine, execute_tools

AGENT_REASON = "agent_reason"
ACT = "act"

def should_continue(state: AgentState) -> str:
    if isinstance(state["agent_outcome"], AgentAction):
        return ACT
    elif isinstance(state["agent_outcome"], AgentFinish):
        return END
    
builder = StateGraph(state_schema=AgentState)
builder.add_node(AGENT_REASON, run_agent_reasoning_engine)
builder.add_node(ACT, execute_tools)
builder.add_edge(ACT, AGENT_REASON)
builder.add_conditional_edges(AGENT_REASON, should_continue)
builder.set_entry_point(AGENT_REASON)
graph = builder.compile()

graph.get_graph().draw_mermaid_png(output_file_path="graph.png")

if __name__ == "__main__":
    res = graph.invoke({
        "input": "What is the weather in Bangalore ? Write it and then Triple it"
    })

    print(res["agent_outcome"].return_values["output"])