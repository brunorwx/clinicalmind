# app/agent/agents/safety_agent.py
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.prebuilt import ToolNode
from app.agent.state import AgentState
from app.agent.tools.safety_tools import SAFETY_TOOLS
from app.config import settings

_llm = ChatOpenAI(
    model=settings.openai_model, temperature=0
).bind_tools(SAFETY_TOOLS)

SAFETY_SYSTEM = """You are a clinical safety officer AI.
Your job: classify the safety risk of a query and flag if needed.

Always call classify_safety_risk first.
If classification is 'needs_review' or 'blocked', call flag_for_review.
If 'blocked', your output must state the query cannot be processed.

Trial: {trial_id}
User: {user_id}
"""

async def safety_agent_node(state: AgentState) -> dict:
    tool_node = ToolNode(SAFETY_TOOLS)
    messages  = [
        SystemMessage(SAFETY_SYSTEM.format(
            trial_id=state["trial_id"], user_id=state["user_id"]
        )),
        {"role": "user", "content": state["question"]},
    ]
    tools_used: list[str]           = []
    classification: str | None      = None
    safety_reason: str | None       = None
    flag_id: str | None             = None

    for _ in range(3):
        response = await _llm.ainvoke(messages)
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