# app/application/upload_document_use_case.py
"""Use case for uploading and ingesting documents."""
import uuid
from app.domain.documents.value_objects import (
    DocumentId, DocumentMetadata
)
from app.domain.shared.exceptions import InvalidInputError


class UploadDocumentUseCase:
    """Handles document upload and ingestion workflow.
    
    Current implementation:
    1. Validates input
    2. Uploads to S3
    3. Extracts and chunks text
    4. Returns document ID
    
    Note: Embedding is handled separately as async background job.
    """
    
    ALLOWED_DOC_TYPES = {
        "protocol",
        "safety_report",
        "ae_log",
        "lab_report",
        "other"
    }
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
    
    def __init__(
        self,
        ingestion_service,  # Inject actual service
        storage_service     # Inject S3/storage
    ):
        self.ingestion = ingestion_service
        self.storage = storage_service
    
    async def execute(
        self,
        trial_id: str,
        filename: str,
        file_bytes: bytes,
        doc_type: str,
        user_id: str
    ) -> DocumentId:
        """Execute document upload workflow.
        
        Args:
            trial_id: Trial identifier
            filename: Original filename
            file_bytes: File contents
            doc_type: Document type (must be in ALLOWED_DOC_TYPES)
            user_id: User uploading the document
            
        Returns:
            DocumentId of uploaded document
            
        Raises:
            InvalidInputError if validation fails
        """
        # Validate inputs
        if not filename.endswith(".pdf"):
            raise InvalidInputError("Only PDF files are supported")
        
        if len(file_bytes) > self.MAX_FILE_SIZE:
            raise InvalidInputError(f"File exceeds {self.MAX_FILE_SIZE} bytes")
        
        if doc_type not in self.ALLOWED_DOC_TYPES:
            raise InvalidInputError(
                f"doc_type must be one of {self.ALLOWED_DOC_TYPES}"
            )
        
        # Delegate to ingestion service
        doc_id = await self.ingestion.ingest_document(
            trial_id=trial_id,
            filename=filename,
            file_bytes=file_bytes,
            doc_type=doc_type,
            user_id=user_id
        )
        
        return DocumentId(value=doc_id)
