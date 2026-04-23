# app/domain/query/interfaces.py
"""Interfaces for the query domain."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from app.domain.query.value_objects import Answer, TrialId, Question


class LLMProvider(ABC):
    """Abstract interface for LLM interactions."""
    
    @abstractmethod
    async def synthesize_answer(
        self,
        trial_id: TrialId,
        question: Question,
        agent_outputs: Dict[str, str]
    ) -> str:
        """Synthesize a final answer from agent outputs."""
        pass


class Agent(ABC):
    """Abstract interface for query agents."""
    
    @abstractmethod
    async def execute(
        self,
        trial_id: TrialId,
        question: Question,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the agent and return results."""
        pass


class QuerySynthesizer(ABC):
    """Abstract interface for synthesizing query results."""
    
    @abstractmethod
    async def synthesize(
        self,
        trial_id: TrialId,
        question: Question,
        agent_outputs: Dict[str, Any],
        retrieved_chunks: List[Any]
    ) -> Answer:
        """Synthesize agent outputs into a final answer."""
        pass
