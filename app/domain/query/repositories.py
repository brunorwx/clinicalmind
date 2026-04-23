# app/domain/query/repositories.py
"""Repository interfaces for the query domain."""
from abc import ABC, abstractmethod
from typing import List
from app.domain.documents.value_objects import DocumentChunk


class ChunkRepository(ABC):
    """Abstract repository for document chunks."""
    
    @abstractmethod
    async def find_similar(
        self, 
        embedding: List[float], 
        trial_id: str, 
        limit: int = 5
    ) -> List[DocumentChunk]:
        """Find chunks similar to the given embedding."""
        pass
    
    @abstractmethod
    async def save(self, chunk: DocumentChunk) -> None:
        """Save a chunk to the repository."""
        pass
    
    @abstractmethod
    async def find_by_trial(self, trial_id: str) -> List[DocumentChunk]:
        """Find all chunks for a trial."""
        pass
