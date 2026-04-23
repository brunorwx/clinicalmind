# app/agent/nodes.py
from app.agent.state import AgentState

async def input_node(state: AgentState) -> dict:
    """
    Entry node. Validates and normalizes the incoming state.
    Extracts the human question from the last message.
    """
    question = ""
    for msg in reversed(state["messages"]):
        if getattr(msg, "type", None) == "human" or isinstance(msg, dict) and msg.get("role") == "human":
            question = getattr(msg, "content", msg.get("content", ""))
            break

    if not question:
        return {"error": "No question found in messages.", "final_answer": "Please provide a question."}

    if len(question) > 2000:
        return {"error": "Question too long.", "final_answer": "Please shorten your question."}

    return {
        "question": question.strip(),
        "iteration_count": 0,
        "tools_used": [],
        "agents_completed": [],
        "agent_outputs": {},
        "retrieved_chunks": [],
        "error": None,
    }


async def blocked_node(state: AgentState) -> dict:
    """Terminal node for blocked queries."""
    reason = state.get("safety_reason", "This query cannot be processed.")
    return {
        "final_answer": f"This query has been blocked by the safety system. {reason}",
        "error": "blocked",
    }


async def audit_node(state: AgentState) -> dict:
    """
    Runs after synthesis. Writes a full audit record to the database.
    Non-blocking — errors here don't fail the request.
    """
    from sqlalchemy import text
    from app.database import async_session
    import uuid

    try:
        async with async_session() as db:
            await db.execute(text("""
                INSERT INTO audit_logs
                    (id, user_id, trial_id, question, agents_invoked, tools_used, chunk_ids, created_at)
                VALUES (:id, :uid, :tid, :q, :agents, :tools, :chunks, NOW())
            """), {
                "id":      str(uuid.uuid4()),
                "uid":     state.get("user_id", "unknown"),
                "tid":     state.get("trial_id", "unknown"),
                "q":       state.get("question", ""),
                "agents":  state.get("agents_completed", []),
                "tools":   state.get("tools_used", []),
                "chunks":  [c["id"] for c in state.get("retrieved_chunks", [])],
            })
            await db.commit()
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Audit log failed: {e}")

    return {}   # No state change — purely a side-effect node