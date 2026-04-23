# app/agent/agents/supervisor.py
"""
The supervisor reads the question and decides which agents to invoke.
It also synthesizes the final answer from all agent outputs.
"""
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from app.agent.state import AgentState
from app.config import settings

_llm = ChatOpenAI(model=settings.openai_model, temperature=0)

ROUTING_PROMPT = """You are the supervisor of a clinical research AI system.

Given a user question, decide which specialist agents to invoke.
Available agents:
  - "rag":    searches clinical documents (protocols, reports, notes)
  - "data":   queries structured database (patient counts, lab results, AEs)
  - "safety": classifies safety risk and flags for human review

Rules:
  - Always invoke "safety" first.
  - Invoke "rag" for qualitative/document questions.
  - Invoke "data" for quantitative/statistical questions.
  - Invoke both "rag" and "data" for complex questions needing both.
  - Never invoke an agent unless it adds value.

Respond with ONLY a JSON array of agent names, e.g. ["safety", "rag", "data"]
"""

async def supervisor_router_node(state: AgentState) -> dict:
    """Decides which agents to invoke for this question."""
    resp = await _llm.ainvoke([
        SystemMessage(ROUTING_PROMPT),
        {"role": "user", "content": state["question"]},
    ])
    try:
        agents = json.loads(resp.content)
        # Guarantee safety is always first
        if "safety" not in agents:
            agents = ["safety"] + agents
    except Exception:
        agents = ["safety", "rag"]

    return {"agents_to_invoke": agents, "agents_completed": []}


SYNTHESIS_PROMPT = """You are a clinical research assistant synthesizing findings.

Trial: {trial_id}
Original question: {question}

Specialist agent findings:
{agent_outputs}

Instructions:
1. Synthesize ALL findings into one coherent answer.
2. Cite sources using [Source: filename, p.X] format.
3. If safety flagged this for review, prominently note it.
4. If data is missing or ambiguous, say so explicitly.
5. Never speculate beyond what the agents found.
"""

async def supervisor_synthesizer_node(state: AgentState) -> dict:
    """Synthesizes all agent outputs into the final answer."""
    outputs_text = "\n\n".join(
        f"=== {name.upper()} AGENT ===\n{result}"
        for name, result in state.get("agent_outputs", {}).items()
    )
    prompt = SYNTHESIS_PROMPT.format(
        trial_id=state["trial_id"],
        question=state["question"],
        agent_outputs=outputs_text or "No agent outputs available.",
    )
    resp = await _llm.ainvoke(prompt)

    # Collect sources from retrieved chunks
    sources = [
        {"document": c["metadata"].get("filename"),
         "page": c["metadata"].get("page"),
         "similarity": round(c["similarity"], 3)}
        for c in state.get("retrieved_chunks", [])
    ]

    return {"final_answer": resp.content, "sources": sources}