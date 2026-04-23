# app/infrastructure/query_synthesizer.py
"""Query synthesizer implementation."""
from typing import List, Any
from app.domain.query.interfaces import QuerySynthesizer, LLMProvider
from app.domain.query.value_objects import (
    TrialId, Question, Answer, Source
)


class DefaultQuerySynthesizer(QuerySynthesizer):
    """Default implementation of query synthesizer."""
    
    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider
    
    async def synthesize(
        self,
        trial_id: TrialId,
        question: Question,
        agent_outputs: dict[str, Any],
        retrieved_chunks: List[Any]
    ) -> Answer:
        """Synthesize agent outputs into a final answer."""
        # Generate answer text
        answer_text = await self.llm.synthesize_answer(
            trial_id,
            question,
            agent_outputs
        )
        
        # Extract sources from chunks
        sources = [
            Source(
                document=chunk.get("metadata", {}).get("filename", "unknown"),
                page=chunk.get("metadata", {}).get("page"),
                similarity=round(chunk.get("similarity", 0), 3)
            )
            for chunk in retrieved_chunks
        ]
        
        # Determine which agents were used
        agents_used = list(agent_outputs.keys())
        
        return Answer(
            text=answer_text,
            sources=sources,
            agents_used=agents_used
        )
