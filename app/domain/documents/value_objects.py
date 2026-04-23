# app/domain/documents/value_objects.py
"""Value objects for the documents domain."""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DocumentId:
    """Represents a document identifier."""
    value: str


@dataclass(frozen=True)
class DocumentChunk:
    """Represents a chunk of text from a document."""
    id: str
    document_id: str
    content: str
    page: int | None = None
    embedding: Optional[list[float]] = None
    
    def has_embedding(self) -> bool:
        return self.embedding is not None


@dataclass(frozen=True)
class DocumentMetadata:
    """Metadata about a document."""
    filename: str
    doc_type: str  # "protocol", "safety_report", "ae_log", "lab_report", "other"
    trial_id: str
    uploaded_by: str
    created_at: str  # ISO format
    s3_key: str
