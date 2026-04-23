# app/domain/safety/interfaces.py
"""Interfaces for the safety domain."""
from abc import ABC, abstractmethod
from app.domain.query.value_objects import Question
from app.domain.safety.value_objects import SafetyClassification, SafetyFlag


class SafetyChecker(ABC):
    """Abstract interface for safety checking."""
    
    @abstractmethod
    async def check(self, question: Question, user_id: str) -> SafetyClassification:
        """Check if a question is safe to process."""
        pass
    
    @abstractmethod
    async def create_flag(
        self,
        classification: SafetyClassification,
        question: Question,
        user_id: str
    ) -> SafetyFlag:
        """Create a flag for a safety issue."""
        pass
