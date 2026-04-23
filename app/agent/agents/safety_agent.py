# app/agent/agents/safety_agent.py
"""Safety Agent - classifies safety risks and flags queries for review.

Dependency Injection:
  - LLM provider instantiated per call (no global state)
  - Tools bound dynamically at runtime
"""
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.prebuilt import ToolNode
from app.agent.state import AgentState
from app.agent.tools.safety_tools import SAFETY_TOOLS
from app.config import settings

SAFETY_SYSTEM = """You are a clinical safety officer AI.
Your job: classify the safety risk of a query and flag if needed.

Always call classify_safety_risk first.
If classification is 'needs_review' or 'blocked', call flag_for_review.
If 'blocked', your output must state the query cannot be processed.

Trial: {trial_id}
User: {user_id}
"""

def _get_safety_llm():
    """Get safety agent LLM with tools bound."""
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0
    )
    return llm.bind_tools(SAFETY_TOOLS)

async def safety_agent_node(state: AgentState) -> dict:
    """Safety agent: classifies safety risk and flags if needed.
    
    Runs a tool-calling loop to classify risk and optionally flag for review.
    Terminates when classification is complete.
    """
    llm = _get_safety_llm()
    tool_node = ToolNode(SAFETY_TOOLS)
    messages = [
        SystemMessage(SAFETY_SYSTEM.format(
            trial_id=state["trial_id"],
            user_id=state["user_id"]
        )),
        {"role": "user", "content": state["question"]},
    ]
    tools_used: list[str] = []
    classification: str | None = None
    safety_reason: str | None = None
    flag_id: str | None = None

    for _ in range(3):
        response = await llm.ainvoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        tool_results = await tool_node.ainvoke({"messages": messages})
        new_msgs = tool_results["messages"]
        messages.extend(new_msgs)
        tools_used.extend(tc.name for tc in response.tool_calls)

        # Parse classification from classify_safety_risk tool result
        for msg in new_msgs:
            if isinstance(msg, AIMessage):
                continue
            content = getattr(msg, "content", "")
            if isinstance(content, str):
                try:
                    parsed = json.loads(content)
                    if "classification" in parsed:
                        classification = parsed["classification"]
                        safety_reason  = parsed.get("reason", "")
                except Exception:
                    # Tool returned plain text (flag_for_review result)
                    if "Flag ID:" in content:
                        flag_id = content.split("Flag ID:")[-1].strip()

    safety_summary = f"Classification: {classification or 'unknown'}. {safety_reason or ''}"

    return {
        "safety_classification": classification or "safe",
        "safety_reason": safety_reason,
        "flag_id": flag_id,
        "agent_outputs": {"safety": safety_summary},
        "tools_used": state.get("tools_used", []) + tools_used,
        "agents_completed": state.get("agents_completed", []) + ["safety"],
    }