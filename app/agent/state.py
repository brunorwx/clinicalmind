# app/agent/state.py
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

# Reducer for agent_outputs — merges dicts from multiple agents
def merge_agent_outputs(existing: dict, new: dict) -> dict:
    return {**existing, **new}

class AgentState(TypedDict):
    # ── Core conversation ───────────────────────────────────────────
    messages: Annotated[list, add_messages]

    # ── Request context ─────────────────────────────────────────────
    trial_id: str
    user_id: str
    question: str                    # original question, never mutated

    # ── Supervisor routing ──────────────────────────────────────────
    # Which agents should handle this question
    agents_to_invoke: list[str]      # ["rag", "data", "safety"]
    # Results keyed by agent name
    agent_outputs: Annotated[dict, merge_agent_outputs]

    # ── RAG agent state ─────────────────────────────────────────────
    retrieved_chunks: list[dict]
    rag_answer: str | None

    # ── Data agent state ────────────────────────────────────────────
    sql_results: str | None
    analysis_result: str | None
    data_answer: str | None

    # ── Safety agent state ──────────────────────────────────────────
    safety_classification: Literal["safe", "needs_review", "blocked"] | None
    safety_reason: str | None
    flag_id: str | None

    # ── Orchestration ───────────────────────────────────────────────
    iteration_count: int
    tools_used: list[str]
    agents_completed: list[str]

    # ── Output ──────────────────────────────────────────────────────
    final_answer: str | None
    sources: list[dict]
    error: str | None