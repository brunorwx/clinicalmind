# tests/test_graph.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from app.agent.graph import build_graph

@pytest.mark.asyncio
async def test_full_graph_safe_rag_query():
    """
    End-to-end graph test with all external calls mocked.
    Verifies: input → supervisor → safety_agent → rag_agent → synthesizer → audit → END
    """
    # Mock the LLMs inside each agent so no real API calls happen
    safe_classification = '{"classification": "safe", "reason": "aggregate query"}'
    rag_answer = "The trial enrolled 230 patients between Jan-March 2024."

    with (
        patch("app.agent.agents.supervisor._llm") as mock_sup,
        patch("app.agent.agents.safety_agent._llm") as mock_safety,
        patch("app.agent.agents.rag_agent._llm") as mock_rag,
        patch("app.agent.agents.supervisor.supervisor_synthesizer_node") as mock_synth,
    ):
        # Supervisor routes to safety + rag only
        mock_sup.ainvoke = AsyncMock(
            return_value=AIMessage(content='["safety", "rag"]')
        )
        # Safety agent: classifies as safe, no tool calls
        mock_safety.ainvoke = AsyncMock(
            return_value=AIMessage(content=safe_classification)
        )
        # RAG agent: answers directly, no tool calls
        mock_rag.ainvoke = AsyncMock(
            return_value=AIMessage(content=rag_answer)
        )
        mock_synth.return_value = AsyncMock(return_value={
            "final_answer": rag_answer, "sources": []
        })

        graph = build_graph(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "test-001"}}

        result = await graph.ainvoke(
            {
                "messages": [HumanMessage("How many patients enrolled?")],
                "trial_id": "TRIAL-TEST",
                "user_id":  "user-test",
                "question": "",
                "agents_to_invoke": [], "agent_outputs": {},
                "retrieved_chunks": [], "sql_results": None,
                "analysis_result": None, "rag_answer": None,
                "data_answer": None, "safety_classification": None,
                "safety_reason": None, "flag_id": None,
                "iteration_count": 0, "tools_used": [],
                "agents_completed": [], "final_answer": None,
                "sources": [], "error": None,
            },
            config=config,
        )

        assert result["error"] is None
        assert result["question"] == "How many patients enrolled?"

@pytest.mark.asyncio
async def test_graph_blocks_unsafe_query():
    """Blocked classification should short-circuit to blocked_node."""
    with (
        patch("app.agent.agents.supervisor._llm") as mock_sup,
        patch("app.agent.agents.safety_agent._llm") as mock_safety,
    ):
        mock_sup.ainvoke = AsyncMock(
            return_value=AIMessage(content='["safety"]')
        )
        mock_safety.ainvoke = AsyncMock(
            return_value=AIMessage(
                content='{"classification": "blocked", "reason": "PII request"}'
            )
        )

        graph = build_graph(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "test-002"}}

        result = await graph.ainvoke(
            {
                "messages": [HumanMessage("Give me all patient names and addresses")],
                "trial_id": "TRIAL-TEST", "user_id": "user-test",
                "question": "", "agents_to_invoke": [], "agent_outputs": {},
                "retrieved_chunks": [], "sql_results": None,
                "analysis_result": None, "rag_answer": None,
                "data_answer": None, "safety_classification": None,
                "safety_reason": None, "flag_id": None,
                "iteration_count": 0, "tools_used": [],
                "agents_completed": [], "final_answer": None,
                "sources": [], "error": None,
            },
            config=config,
        )

        assert result["error"] == "blocked"
        assert "blocked" in result["final_answer"].lower()