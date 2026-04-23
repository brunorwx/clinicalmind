# app/agent/tools/safety_tools.py
import uuid
from langchain_core.tools import tool
from sqlalchemy import text
from app.database import async_session

@tool
async def flag_for_review(
    question: str, reason: str, priority: str, trial_id: str, user_id: str
) -> str:
    """
    Flag this query for human clinical review.
    Use when: patient safety is at risk, data is ambiguous,
    the query is outside safe automated scope, or confidence is low.

    Args:
        question: The original user question.
        reason:   Why human review is needed.
        priority: 'low' | 'medium' | 'high'.
        trial_id: Trial identifier.
        user_id:  Requesting user.
    """
    flag_id = str(uuid.uuid4())
    async with async_session() as db:
        await db.execute(text("""
            INSERT INTO review_flags (id, trial_id, user_id, question, reason, priority, resolved, created_at)
            VALUES (:id, :tid, :uid, :q, :r, :p, false, NOW())
        """), {"id": flag_id, "tid": trial_id, "uid": user_id, "q": question,
               "r": reason, "p": priority if priority in ("low","medium","high") else "medium"})
        await db.commit()
    return f"Flagged for {priority} priority review. Flag ID: {flag_id}"

@tool
async def classify_safety_risk(question: str, trial_id: str) -> str:
    """
    Classify whether this query poses a patient safety risk.
    Returns 'safe', 'needs_review', or 'blocked' with reasoning.

    Use at the start of every query to determine handling.

    Args:
        question: The user question to classify.
        trial_id: Trial context.
    """
    from langchain_openai import ChatOpenAI
    from app.config import settings

    llm = ChatOpenAI(model=settings.openai_model, temperature=0)
    prompt = f"""Classify this clinical research question for safety risk.

Question: "{question}"
Trial: {trial_id}

Rules:
- 'blocked': requests to modify data, bypass controls, or share PII
- 'needs_review': patient safety decisions, treatment recommendations,
  questions about specific patient identifiers in a clinical context
- 'safe': aggregate statistics, protocol questions, document retrieval

Respond with ONLY a JSON object: {{"classification": "safe|needs_review|blocked", "reason": "..."}}"""

    resp = await llm.ainvoke(prompt)
    return resp.content

SAFETY_TOOLS = [flag_for_review, classify_safety_risk]