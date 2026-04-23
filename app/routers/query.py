# app/routers/query.py
"""Query routing - HTTP layer only.
Uses ProcessQueryUseCase for business logic.
"""
import json
from uuid import uuid4
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from app.auth import get_current_user
from app.agent.graph import get_graph
from app.container import Container
from app.domain.query.value_objects import TrialId, Question

router = APIRouter(prefix="/api/v1", tags=["query"])


class QueryRequest(BaseModel):
    """HTTP request model for queries."""
    question: str = Field(..., min_length=3, max_length=2000)
    trial_id: str = Field(..., pattern=r"^TRIAL-[A-Z0-9]+$")
    thread_id: str | None = None


@router.post("/query")
async def query(body: QueryRequest, user=Depends(get_current_user)):
    """Process a query through the agent system.
    
    Returns server-sent events with:
    - token: LLM streaming chunks
    - tool_start/tool_end: Tool execution events
    - node_done: Agent completion events
    - final: Final answer with sources and metadata
    """
    # Get use case from DI container
    use_case = Container.get_process_query_use_case()
    
    # Parse domain objects
    try:
        trial_id = TrialId(body.trial_id)
        question = Question(body.question)
    except Exception as e:
        return _error_response(str(e))
    
    # Get graph and prepare execution config
    graph = await get_graph()
    thread_id = body.thread_id or f"{user.id}:{body.trial_id}:{uuid4()}"
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 30
    }

    # Initialize minimal state
    initial = {
        "messages": [HumanMessage(content=body.question)],
        "trial_id": body.trial_id,
        "user_id": user.id,
        "question": body.question,
        "agents_to_invoke": [],
        "agent_outputs": {},
        "retrieved_chunks": [],
        "sql_results": None,
        "analysis_result": None,
        "rag_answer": None,
        "data_answer": None,
        "safety_classification": None,
        "safety_reason": None,
        "flag_id": None,
        "iteration_count": 0,
        "tools_used": [],
        "agents_completed": [],
        "final_answer": None,
        "sources": [],
        "error": None,
    }

    return StreamingResponse(
        _event_stream(graph, initial, config, thread_id, use_case),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


def _error_response(error_msg: str):
    """Return error response."""
    async def error_stream():
        yield _sse({"type": "error", "message": error_msg})
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        error_stream(),
        media_type="text/event-stream",
    )


async def _event_stream(graph, initial, config, thread_id, use_case):
    """Stream events from graph execution."""
    try:
        async for event in graph.astream_events(initial, config=config, version="v2"):
            etype = event["event"]
            name = event.get("name", "")

            if etype == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    yield _sse({"type": "token", "content": chunk.content})

            elif etype == "on_tool_start":
                yield _sse({
                    "type": "tool_start",
                    "tool": name,
                    "preview": _trunc(str(event["data"].get("input", "")), 100)
                })

            elif etype == "on_tool_end":
                yield _sse({
                    "type": "tool_end",
                    "tool": name,
                    "preview": _trunc(str(event["data"].get("output", "")), 150)
                })

            elif etype == "on_chain_end" and name in (
                "input", "supervisor", "safety_agent", "rag_agent", 
                "data_agent", "synthesizer"
            ):
                yield _sse({"type": "node_done", "node": name})

        # Final structured result
        state = await graph.aget_state(config)
        v = state.values
        yield _sse({
            "type": "final",
            "answer": v.get("final_answer", ""),
            "sources": v.get("sources", []),
            "agents_used": v.get("agents_completed", []),
            "tools_used": v.get("tools_used", []),
            "flag_id": v.get("flag_id"),
            "thread_id": thread_id,
            "error": v.get("error"),
        })
    except Exception as e:
        yield _sse({"type": "error", "message": str(e)})
    finally:
        yield "data: [DONE]\n\n"


def _sse(d: dict) -> str:
    """Format dict as server-sent event."""
    return f"data: {json.dumps(d)}\n\n"


def _trunc(s: str, n: int) -> str:
    """Truncate string to n characters."""
    return s[:n] + "…" if len(s) > n else s