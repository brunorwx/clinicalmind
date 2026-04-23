# app/container.py
"""Dependency injection container for ClinicalMind."""
from app.config import settings
from app.database import async_session
from app.infrastructure.llm_providers import OpenAILLMProvider
from app.infrastructure.query_synthesizer import DefaultQuerySynthesizer
from app.infrastructure.safety_checker import OpenAISafetyChecker
from app.infrastructure.repositories.chunk_repository import (
    PostgresChunkRepository
)
from app.application.process_query_use_case import ProcessQueryUseCase
from app.application.upload_document_use_case import UploadDocumentUseCase


class Container:
    """Simple dependency injection container."""
    
    # ─── Singletons ─────────────────────────────────────────────
    _llm_provider = None
    _safety_checker = None
    _synthesizer = None
    
    @classmethod
    def get_llm_provider(cls) -> OpenAILLMProvider:
        """Get or create LLM provider singleton."""
        if cls._llm_provider is None:
            cls._llm_provider = OpenAILLMProvider()
        return cls._llm_provider
    
    @classmethod
    def get_safety_checker(cls) -> OpenAISafetyChecker:
        """Get or create safety checker singleton."""
        if cls._safety_checker is None:
            cls._safety_checker = OpenAISafetyChecker()
        return cls._safety_checker
    
    @classmethod
    def get_synthesizer(cls) -> DefaultQuerySynthesizer:
        """Get or create synthesizer singleton."""
        if cls._synthesizer is None:
            llm = cls.get_llm_provider()
            cls._synthesizer = DefaultQuerySynthesizer(llm)
        return cls._synthesizer
    
    @classmethod
    async def get_chunk_repository(cls) -> PostgresChunkRepository:
        """Create chunk repository with current DB session."""
        async with async_session() as db:
            return PostgresChunkRepository(db)
    
    # ─── Use Cases ───────────────────────────────────────────────
    @classmethod
    def get_process_query_use_case(
        cls,
        agents_dict: dict = None  # Optional agents dict for testing
    ) -> ProcessQueryUseCase:
        """Get process query use case with injected dependencies."""
        return ProcessQueryUseCase(
            safety_checker=cls.get_safety_checker(),
            agents=agents_dict or {},
            synthesizer=cls.get_synthesizer()
        )
    
    @classmethod
    def get_upload_document_use_case(
        cls,
        ingestion_service,
        storage_service
    ) -> UploadDocumentUseCase:
        """Get upload document use case with injected dependencies."""
        return UploadDocumentUseCase(
            ingestion_service=ingestion_service,
            storage_service=storage_service
        )
