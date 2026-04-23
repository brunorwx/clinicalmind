# app/domain/query/value_objects.py
"""Value objects for the query domain."""
from dataclasses import dataclass
from typing import List
from app.domain.shared.exceptions import InvalidInputError


@dataclass(frozen=True)
class TrialId:
    """Represents a trial identifier."""
    value: str
    
    def __post_init__(self):
        if not self.value.startswith("TRIAL-"):
            raise InvalidInputError("Trial ID must start with TRIAL-")
        if len(self.value) < 8:
            raise InvalidInputError("Trial ID too short")


@dataclass(frozen=True)
class Question:
    """Represents a user question in the system."""
    text: str
    
    def __post_init__(self):
        if not self.text or len(self.text) < 3:
            raise InvalidInputError("Question must be at least 3 characters")
        if len(self.text) > 2000:
            raise InvalidInputError("Question exceeds 2000 characters")


@dataclass(frozen=True)
class Source:
    """Represents a source document for an answer."""
    document: str
    page: int | None = None
    similarity: float | None = None


@dataclass(frozen=True)
class Answer:
    """Represents the final synthesized answer to a question."""
    text: str
    sources: List[Source]
    agents_used: List[str]
    flag_id: str | None = None
    
    def __post_init__(self):
        if not self.text:
            raise InvalidInputError("Answer text cannot be empty")
