# tests/test_nodes.py
import pytest
from langchain_core.messages import HumanMessage
from app.agent.nodes import input_node, blocked_node

@pytest.mark.asyncio
async def test_input_node_extracts_question():
    state = {
        "messages": [HumanMessage("How many patients are enrolled?")],
        "trial_id": "TRIAL-TEST",
        "user_id":  "user-1",
    }
    result = await input_node(state)
    assert result["question"] == "How many patients are enrolled?"
    assert result["error"] is None
    assert result["iteration_count"] == 0

@pytest.mark.asyncio
async def test_input_node_rejects_empty():
    state = {"messages": [], "trial_id": "TRIAL-TEST", "user_id": "user-1"}
    result = await input_node(state)
    assert result["error"] is not None

@pytest.mark.asyncio
async def test_input_node_rejects_too_long():
    state = {
        "messages": [HumanMessage("x" * 3000)],
        "trial_id": "TRIAL-TEST", "user_id": "user-1",
    }
    result = await input_node(state)
    assert result["error"] is not None

@pytest.mark.asyncio
async def test_blocked_node_returns_explanation():
    state = {"safety_reason": "Query requests patient PII."}
    result = await blocked_node(state)
    assert "blocked" in result["final_answer"].lower()
    assert result["error"] == "blocked"