# app/agent/edges.py
from typing import Literal
from app.agent.state import AgentState

def route_after_supervisor(state: AgentState) -> list[str]:
    """
    Returns a list of nodes to run in parallel.
    LangGraph supports fan-out: returning multiple node names
    runs them concurrently, then waits for all to complete.
    """
    agents = state.get("agents_to_invoke", ["safety", "rag"])
    node_map = {
        "rag":    "rag_agent",
        "data":   "data_agent",
        "safety": "safety_agent",
    }
    return [node_map[a] for a in agents if a in node_map]


def route_after_safety(state: AgentState) -> Literal["synthesizer", "blocked"]:
    """
    If safety agent classified the query as blocked, skip to a
    blocked terminal node instead of running other agents.
    """
    if state.get("safety_classification") == "blocked":
        return "blocked"
    return "synthesizer"


def all_agents_complete(state: AgentState) -> bool:
    """
    Used as a wait condition — all invoked agents must finish
    before the synthesizer runs.
    """
    invoked   = set(state.get("agents_to_invoke", []))
    completed = set(state.get("agents_completed", []))
    return invoked.issubset(completed)