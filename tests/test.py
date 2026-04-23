# tests/test_edges.py
import pytest
from app.agent.edges import route_after_safety, route_after_supervisor

def test_route_safety_blocked():
    state = {"safety_classification": "blocked", "agents_to_invoke": ["safety"]}
    assert route_after_safety(state) == "blocked"

def test_route_safety_needs_review():
    state = {"safety_classification": "needs_review", "agents_to_invoke": ["safety"]}
    assert route_after_safety(state) == "synthesizer"

def test_route_safety_safe():
    state = {"safety_classification": "safe", "agents_to_invoke": ["safety"]}
    assert route_after_safety(state) == "synthesizer"

def test_supervisor_routes_to_correct_nodes():
    state = {"agents_to_invoke": ["safety", "rag", "data"]}
    result = route_after_supervisor(state)
    assert "safety_agent" in result
    assert "rag_agent" in result
    assert "data_agent" in result

def test_supervisor_handles_unknown_agent():
    state = {"agents_to_invoke": ["safety", "unknown_agent"]}
    result = route_after_supervisor(state)
    assert "safety_agent" in result
    assert len(result) == 1   # unknown_agent filtered out