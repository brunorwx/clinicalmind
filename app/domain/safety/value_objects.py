# app/domain/safety/value_objects.py
"""Value objects for the safety domain."""
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SafetyClassification:
    """Represents a safety classification result."""
    level: Literal["safe", "needs_review", "blocked"]
    reason: str
    
    def is_blocked(self) -> bool:
        return self.level == "blocked"
    
    def needs_review(self) -> bool:
        return self.level == "needs_review"


@dataclass(frozen=True)
class SafetyFlag:
    """Represents a flagged query requiring review."""
    id: str
    classification: SafetyClassification
    query_text: str
    user_id: str
    created_at: str  # ISO format timestamp
