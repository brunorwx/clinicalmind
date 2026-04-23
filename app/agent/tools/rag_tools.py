# app/agent/tools/rag_tools.py
from langchain_core.tools import tool
from app.services.retriever import retrieve

@tool
async def search_documents(query: str, trial_id: str, top_k: int = 6) -> str:
    """
    Semantic search over clinical trial documents (protocols, reports, AE logs).
    Use for qualitative questions: protocols, procedures, documented events.

    Args:
        query:    Natural language search query. Be specific.
        trial_id: The trial identifier e.g. TRIAL-ONC2024.
        top_k:    Number of chunks (1-10).
    """
    chunks = await retrieve(query, trial_id, min(top_k, 10))
    if not chunks:
        return "No relevant documents found."
    parts = []
    for i, c in enumerate(chunks, 1):
        fn = c["metadata"].get("filename", "unknown")
        pg = c["metadata"].get("page", "?")
        parts.append(f"[{i}] {fn} p.{pg} ({c['similarity']:.0%})\n{c['content']}")
    return "\n\n".join(parts)

@tool
async def get_document_metadata(trial_id: str) -> str:
    """
    List all documents ingested for this trial.
    Use to understand what source material is available.
    """
    from sqlalchemy import text
    from app.database import async_session
    async with async_session() as db:
        rows = (await db.execute(text(
            "SELECT filename, doc_type, created_at FROM documents WHERE trial_id=:tid ORDER BY created_at DESC"
        ), {"tid": trial_id})).mappings().all()
    if not rows:
        return "No documents found for this trial."
    return "\n".join(f"- {r['filename']} ({r['doc_type']}) — {r['created_at'].date()}" for r in rows)

RAG_TOOLS = [search_documents, get_document_metadata]