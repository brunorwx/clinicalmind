# app/agent/state.py
"""Unified agent state combining all contexts."""
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


def merge_agent_outputs(existing: dict, new: dict) -> dict:
    """Reducer for merging agent outputs."""
    return {**existing, **new}


class AgentState(TypedDict):
    """Complete state for agent orchestration.
    
    Composed of multiple contexts to keep state concerns separated.
    """
    # Core query context
    messages: Annotated[list, add_messages]
    trial_id: str
    user_id: str
    question: str
    
    # RAG context
    retrieved_chunks: list[dict]
    rag_answer: str | None
    
    # Data context
    sql_results: str | None
    analysis_result: str | None
    data_answer: str | None
    
    # Safety context
    safety_classification: Literal["safe", "needs_review", "blocked"] | None
    safety_reason: str | None
    flag_id: str | None
    agents_completed: list[str]
    agent_outputs: Annotated[dict, merge_agent_outputs]
    
    # Orchestration
    agents_to_invoke: list[str]
    iteration_count: int
    tools_used: list[str]
    
    # Output
    final_answer: str | None
    sources: list[dict]
    error: str | None