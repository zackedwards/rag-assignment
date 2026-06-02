"""
embed.py
Turns each chunk into a 768 dimension embedding using gemini-embedding-001.

Two things the assignment text does not spell out but matter a lot.

1. gemini-embedding-001 returns 3072 dimensions by default. pgvector HNSW
   indexes cap at 2000 dimensions, so a vector(3072) column cannot be indexed.
   We ask for output_dimensionality=768, which the Google docs say loses almost
   no quality, and our column is vector(768).

2. For any dimension other than 3072, gemini-embedding-001 does not normalize
   the vector for you. We normalize to unit length so cosine search behaves.
"""
import asyncio
import os
import numpy as np
from google import genai
from google.genai import types

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))
CONCURRENCY = int(os.getenv("EMBED_CONCURRENCY", "15"))

_client = None
_semaphore = None


def client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(
            vertexai=True,
            project=os.environ["GCP_PROJECT"],
            location=os.environ["GCP_REGION"],
        )
    return _client


def semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(CONCURRENCY)
    return _semaphore


def _normalize(values: list[float]) -> list[float]:
    vec = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec.tolist()
    return (vec / norm).tolist()


async def embed_one(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    """Embed a single string with retry and exponential backoff."""
    async with semaphore():
        for attempt in range(5):
            try:
                response = await client().aio.models.embed_content(
                    model=EMBED_MODEL,
                    contents=text,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=EMBED_DIM,
                    ),
                )
                return _normalize(response.embeddings[0].values)
            except Exception as e:
                if attempt < 4:
                    wait = 2 ** attempt
                    print(f"embed retry {attempt + 1} after {wait}s ({e})")
                    await asyncio.sleep(wait)
                else:
                    raise


async def embed_query(text: str) -> list[float]:
    """Embed a user question. Note the RETRIEVAL_QUERY task type."""
    return await embed_one(text, task_type="RETRIEVAL_QUERY")


async def embed_chunks(records: list[dict]) -> list[dict]:
    """Attach an embedding to every chunk record, running calls concurrently."""
    tasks = [embed_one(r["content"]) for r in records]
    embeddings = await asyncio.gather(*tasks)
    for record, emb in zip(records, embeddings):
        record["embedding"] = emb
    print(f"embedded {len(records)} chunks at {EMBED_DIM} dimensions")
    return records
