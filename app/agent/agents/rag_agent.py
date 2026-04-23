# app/agent/agents/rag_agent.py
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from app.agent.state import AgentState
from app.agent.tools.rag_tools import RAG_TOOLS
from app.config import settings

_llm = ChatOpenAI(
    model=settings.openai_model, temperature=0.1
).bind_tools(RAG_TOOLS)

RAG_SYSTEM = """You are a clinical document specialist.
Your job: search trial documents to answer the question.
Use search_documents with targeted queries.
If documents are insufficient, say so clearly.
Always note which documents you found information in.
Trial: {trial_id}
"""

async def rag_agent_node(state: AgentState) -> dict:
    """
    RAG agent: runs a tool-calling loop against document search tools.
    Terminates when LLM stops calling tools (has enough context).
    """
    from langchain_core.messages import AIMessage, ToolMessage
    from langgraph.prebuilt import ToolNode

    tool_node = ToolNode(RAG_TOOLS)
    messages  = [
        SystemMessage(RAG_SYSTEM.format(trial_id=state["trial_id"])),
        {"role": "user", "content": state["question"]},
    ]
    all_chunks: list[dict] = []
    tools_used: list[str]  = []

    # Internal mini-loop: up to 4 tool call rounds
    for _ in range(4):
        response = await _llm.ainvoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        # Execute tools
        tool_results = await tool_node.ainvoke({"messages": messages})
        new_messages  = tool_results["messages"]
        messages.extend(new_messages)
        tools_used.extend(tc.name for tc in response.tool_calls)

    # Final answer is the last non-tool-calling AI message
    final = next(
        (m.content for m in reversed(messages)
         if isinstance(m, AIMessage) and not m.tool_calls),
        "RAG agent could not find relevant information."
    )

    return {
        "rag_answer": final,
        "agent_outputs": {"rag": final},
        "tools_used": state.get("tools_used", []) + tools_used,
        "agents_completed": state.get("agents_completed", []) + ["rag"],
    }