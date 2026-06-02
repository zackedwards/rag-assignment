"""
embed.py
Turns each chunk into a 768 dimension embedding using gemini-embedding-001.

Three things the assignment text does not spell out but matter a lot.

1. gemini-embedding-001 returns 3072 dimensions by default. pgvector HNSW
   indexes cap at 2000 dimensions, so a vector(3072) column cannot be indexed.
   We ask for output_dimensionality=768, which the Google docs say loses almost
   no quality, and our column is vector(768).

2. For any dimension other than 3072, gemini-embedding-001 does not normalize
   the vector for you. We normalize to unit length so cosine search behaves.

3. The real rate limit is TOKENS PER MINUTE PER PROJECT (5,000,000 by default
   for the gemini-embedding base model), and that budget is SHARED across the
   whole project. Firing thousands of calls in a burst blows it and triggers
   429 RESOURCE_EXHAUSTED. We pace calls with a token bucket sized by EMBED_TPM
   so a run stays under budget. On a shared class project, set EMBED_TPM low
   enough that all the students running at once still add up to under 5,000,000.
"""
import asyncio
import os
import random
import time
import numpy as np
from google import genai
from google.genai import types

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))
CONCURRENCY = int(os.getenv("EMBED_CONCURRENCY", "5"))
# tokens per minute this process is allowed to send. keep the room's total
# under the project quota of 5,000,000.
EMBED_TPM = int(os.getenv("EMBED_TPM", "250000"))
MAX_ATTEMPTS = int(os.getenv("EMBED_MAX_ATTEMPTS", "8"))

_client = None
_semaphore = None
_bucket = None


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


class TokenBucket:
    """Paces token throughput so we stay under a tokens per minute budget."""

    def __init__(self, tokens_per_minute: int):
        self.capacity = float(tokens_per_minute)
        self.tokens = float(tokens_per_minute)
        self.rate = tokens_per_minute / 60.0  # tokens refilled per second
        self.updated = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self, amount: int) -> None:
        amount = min(float(amount), self.capacity)
        while True:
            async with self.lock:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
                self.updated = now
                if self.tokens >= amount:
                    self.tokens -= amount
                    return
                wait = (amount - self.tokens) / self.rate
            await asyncio.sleep(wait)


def bucket() -> TokenBucket:
    global _bucket
    if _bucket is None:
        _bucket = TokenBucket(EMBED_TPM)
    return _bucket


def estimate_tokens(text: str) -> int:
    # rough proxy, about 4 characters per token
    return max(1, len(text) // 4)


def _normalize(values: list[float]) -> list[float]:
    vec = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec.tolist()
    return (vec / norm).tolist()


async def embed_one(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    """Embed a single string. Paced by the token bucket, with retry and
    jittered exponential backoff. Raises if it still fails after MAX_ATTEMPTS."""
    await bucket().acquire(estimate_tokens(text))
    async with semaphore():
        for attempt in range(MAX_ATTEMPTS):
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
                if attempt < MAX_ATTEMPTS - 1:
                    # cap the wait near a minute since the quota window is per minute
                    wait = min(60, 2 ** attempt) + random.uniform(0, 1)
                    await asyncio.sleep(wait)
                else:
                    raise


async def embed_query(text: str) -> list[float]:
    """Embed a user question. Note the RETRIEVAL_QUERY task type."""
    return await embed_one(text, task_type="RETRIEVAL_QUERY")


async def embed_chunks(records: list[dict]) -> list[dict]:
    """Attach an embedding to every chunk. A chunk that still fails after all
    retries is dropped with a warning rather than killing the whole run."""
    tasks = [embed_one(r["content"]) for r in records]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    good, failed = [], 0
    for record, result in zip(records, results):
        if isinstance(result, Exception):
            failed += 1
            continue
        record["embedding"] = result
        good.append(record)
    msg = f"embedded {len(good)} chunks at {EMBED_DIM} dimensions"
    if failed:
        msg += f", {failed} dropped after retries"
    print(msg)
    return good
