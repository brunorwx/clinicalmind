# app/agent/graph.py
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from app.agent.state import AgentState
from app.agent.nodes import input_node, blocked_node, audit_node
from app.agent.agents.supervisor import supervisor_router_node, supervisor_synthesizer_node
from app.agent.agents.rag_agent import rag_agent_node
from app.agent.agents.data_agent import data_agent_node
from app.agent.agents.safety_agent import safety_agent_node
from app.agent.edges import route_after_supervisor, route_after_safety

def build_graph(checkpointer=None):
    g = StateGraph(AgentState)

    # ── Register all nodes ─────────────────────────────────────────
    g.add_node("input",       input_node)
    g.add_node("supervisor",  supervisor_router_node)
    g.add_node("safety_agent", safety_agent_node)
    g.add_node("rag_agent",   rag_agent_node)
    g.add_node("data_agent",  data_agent_node)
    g.add_node("blocked",     blocked_node)
    g.add_node("synthesizer", supervisor_synthesizer_node)
    g.add_node("audit",       audit_node)

    # ── Wire edges ─────────────────────────────────────────────────

    # 1. Always start at input
    g.add_edge(START, "input")

    # 2. Input → supervisor router
    g.add_edge("input", "supervisor")

    # 3. Supervisor fans out to agents in parallel
    #    send=True enables parallel execution
    g.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        # Map each possible return value to the node it names
        {
            "safety_agent": "safety_agent",
            "rag_agent":    "rag_agent",
            "data_agent":   "data_agent",
        },
    )

    # 4. Safety check: if blocked, skip to blocked node
    #    Otherwise fall through to synthesizer
    #    (other agents converge naturally — they all edge to synthesizer)
    g.add_conditional_edges(
        "safety_agent",
        route_after_safety,
        {
            "synthesizer": "synthesizer",
            "blocked":     "blocked",
        }
    )

    # 5. RAG and Data agents always go to synthesizer when done
    g.add_edge("rag_agent",  "synthesizer")
    g.add_edge("data_agent", "synthesizer")

    # 6. Blocked goes straight to audit then END
    g.add_edge("blocked",    "audit")

    # 7. Synthesizer → audit → END
    g.add_edge("synthesizer", "audit")
    g.add_edge("audit",       END)

    return g.compile(checkpointer=checkpointer)


async def get_graph():
    """
    Factory used by the FastAPI router.
    Reuses a module-level graph instance in production.
    """
    import psycopg
    from app.config import settings

    conn = await psycopg.AsyncConnection.connect(
        settings.database_url_sync.replace("+asyncpg", "")
    )
    checkpointer = AsyncPostgresSaver(conn)
    await checkpointer.setup()
    return build_graph(checkpointer=checkpointer)