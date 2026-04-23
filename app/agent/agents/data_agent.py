# app/agent/agents/data_agent.py
"""Data Agent - queries database and performs statistical analysis.

Dependency Injection:
  - LLM provider instantiated per call (no global state)
  - Tools bound dynamically at runtime
"""
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.prebuilt import ToolNode
from app.agent.state import AgentState
from app.agent.tools.data_tools import DATA_TOOLS
from app.config import settings

DATA_SYSTEM = """You are a clinical data analyst.
Your job: query the trial database and run statistical analysis to answer the question.

Database schema:
  patients(id, trial_id, external_id, arm, enrolled_date, status)
  adverse_events(id, patient_id, grade, description, onset_day, resolved)
  lab_results(id, patient_id, test_name, value, unit, collected_at)

Always filter by trial_id. Use run_python_analysis for statistics on query results.
Return precise numbers with context (denominator, timeframe).
Trial: {trial_id}
"""

def _get_data_llm():
    """Get data agent LLM with tools bound."""
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0
    )
    return llm.bind_tools(DATA_TOOLS)

async def data_agent_node(state: AgentState) -> dict:
    """Data agent: queries database and analyzes trial statistics.
    
    Runs a tool-calling loop against SQL and analysis tools.
    Terminates when LLM has enough information to answer.
    """
    llm = _get_data_llm()
    tool_node = ToolNode(DATA_TOOLS)
    messages = [
        SystemMessage(DATA_SYSTEM.format(trial_id=state["trial_id"])),
        {"role": "user", "content": state["question"]},
    ]
    tools_used: list[str] = []

    for _ in range(5):
        response = await llm.ainvoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        tool_results = await tool_node.ainvoke({"messages": messages})
        messages.extend(tool_results["messages"])
        tools_used.extend(tc.name for tc in response.tool_calls)

    final = next(
        (m.content for m in reversed(messages)
         if isinstance(m, AIMessage) and not m.tool_calls),
        "Data agent could not retrieve relevant statistics."
    )

    return {
        "data_answer": final,
        "agent_outputs": {"data": final},
        "tools_used": state.get("tools_used", []) + tools_used,
        "agents_completed": state.get("agents_completed", []) + ["data"],
    }