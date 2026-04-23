# app/infrastructure/llm_providers.py
"""Concrete LLM provider implementations."""
import json
from app.domain.query.interfaces import LLMProvider
from app.domain.query.value_objects import TrialId, Question
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from app.config import settings


class OpenAILLMProvider(LLMProvider):
    """OpenAI-based LLM provider."""
    
    def __init__(self, model: str = settings.openai_model):
        self.llm = ChatOpenAI(model=model, temperature=0)
    
    async def synthesize_answer(
        self,
        trial_id: TrialId,
        question: Question,
        agent_outputs: dict[str, str]
    ) -> str:
        """Synthesize agent outputs into a final answer using OpenAI."""
        outputs_text = "\n\n".join(
            f"=== {name.upper()} AGENT ===\n{result}"
            for name, result in agent_outputs.items()
        )
        
        synthesis_prompt = f"""You are a clinical research assistant synthesizing findings.

Trial: {trial_id.value}
Original question: {question.text}

Specialist agent findings:
{outputs_text or "No agent outputs available."}

Instructions:
1. Synthesize ALL findings into one coherent answer.
2. Cite sources using [Source: filename, p.X] format.
3. If data is missing or ambiguous, say so explicitly.
4. Never speculate beyond what the agents found.
"""
        
        resp = await self.llm.ainvoke(synthesis_prompt)
        return resp.content
