# app/services/ingestion.py
"""
Document ingestion pipeline.
Upload PDF → S3 → extract text → chunk → embed → store in pgvector.
"""
import uuid, io
from pypdf import PdfReader
import tiktoken
import boto3
from sqlalchemy import text
from app.database import async_session
from app.services.embedder import embed
from app.config import settings
from app.models.db import Document, DocumentChunk

enc  = tiktoken.encoding_for_model("gpt-4o")
s3   = boto3.client("s3", region_name=settings.aws_region)

CHUNK_SIZE   = 400   # tokens
CHUNK_OVERLAP = 60   # tokens

def _chunk_text(text_: str) -> list[str]:
    tokens = enc.encode(text_)
    chunks, start = [], 0
    while start < len(tokens):
        end = min(start + CHUNK_SIZE, len(tokens))
        chunks.append(enc.decode(tokens[start:end]))
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

async def ingest_document(
    trial_id: str, filename: str, file_bytes: bytes, doc_type: str
) -> str:
    """Returns the document_id after full ingestion."""
    # 1. Upload raw file to S3
    s3_key = f"{trial_id}/{uuid.uuid4()}/{filename}"
    s3.put_object(Bucket=settings.s3_bucket, Key=s3_key, Body=file_bytes)

    # 2. Extract text from PDF
    reader   = PdfReader(io.BytesIO(file_bytes))
    pages    = [p.extract_text() or "" for p in reader.pages]
    full_text = "\n\n".join(pages)

    # 3. Chunk
    raw_chunks = _chunk_text(full_text)

    # 4. Persist document record
    doc_id = str(uuid.uuid4())
    async with async_session() as db:
        await db.execute(text("""
            INSERT INTO documents (id, trial_id, filename, s3_key, doc_type, created_at)
            VALUES (:id, :tid, :fn, :s3, :dt, NOW())
        """), {"id": doc_id, "tid": trial_id, "fn": filename,
               "s3": s3_key, "dt": doc_type})
        await db.commit()

    # 5. Embed and store chunks (batched)
    async with async_session() as db:
        for i, chunk in enumerate(raw_chunks):
            vec = await embed(chunk)
            tokens = len(enc.encode(chunk))
            await db.execute(text("""
                INSERT INTO document_chunks
                    (id, document_id, trial_id, chunk_index, content, token_count, embedding, meta)
                VALUES (:id, :doc_id, :tid, :idx, :content, :tokens, :vec::vector, :meta)
            """), {
                "id": str(uuid.uuid4()), "doc_id": doc_id,
                "tid": trial_id, "idx": i, "content": chunk,
                "tokens": tokens, "vec": str(vec),
                "meta": {"filename": filename, "page": i // 3, "doc_type": doc_type},
            })
        await db.commit()

    return doc_id