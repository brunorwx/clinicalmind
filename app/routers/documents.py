# app/routers/documents.py
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from app.auth import get_current_user
from app.services.ingestion import ingest_document

router = APIRouter(prefix="/api/v1", tags=["documents"])

ALLOWED_TYPES = {"protocol", "safety_report", "ae_log", "lab_report", "other"}

@router.post("/documents/upload")
async def upload_document(
    trial_id: str    = Form(...),
    doc_type: str    = Form(...),
    file: UploadFile = File(...),
    user             = Depends(get_current_user),
):
    if doc_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"doc_type must be one of {ALLOWED_TYPES}")
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    contents = await file.read()
    if len(contents) > 50 * 1024 * 1024:   # 50 MB cap
        raise HTTPException(413, "File too large (max 50 MB).")

    doc_id = await ingest_document(trial_id, file.filename, contents, doc_type)
    return {"document_id": doc_id, "filename": file.filename, "trial_id": trial_id}