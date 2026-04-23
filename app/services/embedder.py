# app/services/embedder.py
import hashlib, json
from openai import AsyncOpenAI
import redis.asyncio as aioredis
from app.config import settings

client  = AsyncOpenAI(api_key=settings.openai_api_key)
_redis  = aioredis.from_url(settings.redis_url)

async def embed(text: str) -> list[float]:
    key = f"embed:{hashlib.sha256(text.encode()).hexdigest()}"
    if cached := await _redis.get(key):
        return json.loads(cached)
    resp = await client.embeddings.create(
        model=settings.openai_embed_model, input=text
    )
    vec = resp.data[0].embedding
    await _redis.setex(key, 86_400, json.dumps(vec))
    return vec