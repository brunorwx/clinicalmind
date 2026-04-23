# app/services/retriever.py
from sqlalchemy import text
from app.database import async_session
from app.services.embedder import embed

async def retrieve(
    query: str, trial_id: str, top_k: int = 6, threshold: float = 0.70
) -> list[dict]:
    vec = await embed(query)
    async with async_session() as db:
        rows = (await db.execute(text("""
            SELECT id, content, token_count, meta,
                   1 - (embedding <=> :vec::vector) AS similarity
            FROM document_chunks
            WHERE trial_id = :tid
            ORDER BY embedding <=> :vec::vector
            LIMIT :k
        """), {"vec": str(vec), "tid": trial_id, "k": top_k})).mappings().all()

    return [
        {"id": str(r["id"]), "content": r["content"],
         "token_count": r["token_count"], "metadata": r["meta"],
         "similarity": float(r["similarity"])}
        for r in rows if float(r["similarity"]) >= threshold
    ]