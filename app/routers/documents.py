# app/routers/documents.py
"""Document routing - HTTP layer only.
Uses UploadDocumentUseCase for business logic.
"""
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from app.auth import get_current_user
from app.services.ingestion import ingest_document
from app.container import Container
from app.domain.shared.exceptions import InvalidInputError

router = APIRouter(prefix="/api/v1", tags=["documents"])

ALLOWED_TYPES = {"protocol", "safety_report", "ae_log", "lab_report", "other"}


@router.post("/documents/upload")
async def upload_document(
    trial_id: str = Form(...),
    doc_type: str = Form(...),
    file: UploadFile = File(...),
    user=Depends(get_current_user),
):
    """Upload a document for a trial.
    
    Validates document type and file size, then delegates to use case.
    """
    try:
        # Get use case from DI container
        use_case = Container.get_upload_document_use_case(
            ingestion_service=type('MockService', (), {
                'ingest_document': ingest_document
            })(),
            storage_service=None  # S3 handled in ingestion
        )
        
        # Validate doc type
        if doc_type not in ALLOWED_TYPES:
            raise InvalidInputError(f"doc_type must be one of {ALLOWED_TYPES}")
        
        # Validate file
        if not file.filename.endswith(".pdf"):
            raise InvalidInputError("Only PDF files are supported")
        
        # Read file
        contents = await file.read()
        if len(contents) > 50 * 1024 * 1024:
            raise InvalidInputError("File too large (max 50 MB)")
        
        # Execute use case
        doc_id = await use_case.execute(
            trial_id=trial_id,
            filename=file.filename,
            file_bytes=contents,
            doc_type=doc_type,
            user_id=user.id
        )
        
        return {
            "document_id": doc_id.value,
            "filename": file.filename,
            "trial_id": trial_id
        }
        
    except InvalidInputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")