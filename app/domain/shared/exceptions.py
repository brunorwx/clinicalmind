# app/domain/shared/exceptions.py
"""Domain-level exceptions for ClinicalMind."""


class DomainError(Exception):
    """Base exception for all domain errors."""
    pass


class SafetyBlockedError(DomainError):
    """Raised when a query is blocked by safety checks."""
    def __init__(self, classification: str, reason: str):
        self.classification = classification
        self.reason = reason
        super().__init__(f"Query blocked: {reason}")


class TrialNotFoundError(DomainError):
    """Raised when a trial cannot be found."""
    pass


class DocumentNotFoundError(DomainError):
    """Raised when a document cannot be found."""
    pass


class InvalidInputError(DomainError):
    """Raised when input validation fails."""
    pass
