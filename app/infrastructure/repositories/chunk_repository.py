# app/infrastructure/repositories/chunk_repository.py
"""Concrete implementation of chunk repository using PostgreSQL."""
from typing import List
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from app.domain.query.repositories import ChunkRepository
from app.domain.documents.value_objects import DocumentChunk


class PostgresChunkRepository(ChunkRepository):
    """PostgreSQL-backed chunk repository."""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
    
    async def find_similar(
        self,
        embedding: List[float],
        trial_id: str,
        limit: int = 5
    ) -> List[DocumentChunk]:
        """Find chunks similar to embedding using pgvector."""
        query = text("""
            SELECT dc.id, dc.document_id, dc.content, dc.page, dc.embedding
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE d.trial_id = :trial_id
            ORDER BY dc.embedding <-> :embedding
            LIMIT :limit
        """)
        
        result = await self.db.execute(
            query,
            {
                "trial_id": trial_id,
                "embedding": embedding,
                "limit": limit
            }
        )
        rows = result.fetchall()
        
        return [
            DocumentChunk(
                id=row[0],
                document_id=row[1],
                content=row[2],
                page=row[3],
                embedding=row[4]
            )
            for row in rows
        ]
    
    async def save(self, chunk: DocumentChunk) -> None:
        """Save a chunk to the database."""
        query = text("""
            INSERT INTO document_chunks 
            (id, document_id, content, page, embedding)
            VALUES (:id, :document_id, :content, :page, :embedding)
        """)
        
        await self.db.execute(
            query,
            {
                "id": chunk.id,
                "document_id": chunk.document_id,
                "content": chunk.content,
                "page": chunk.page,
                "embedding": chunk.embedding
            }
        )
        await self.db.commit()
    
    async def find_by_trial(self, trial_id: str) -> List[DocumentChunk]:
        """Find all chunks for a trial."""
        query = text("""
            SELECT dc.id, dc.document_id, dc.content, dc.page, dc.embedding
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE d.trial_id = :trial_id
        """)
        
        result = await self.db.execute(query, {"trial_id": trial_id})
        rows = result.fetchall()
        
        return [
            DocumentChunk(
                id=row[0],
                document_id=row[1],
                content=row[2],
                page=row[3],
                embedding=row[4]
            )
            for row in rows
        ]
