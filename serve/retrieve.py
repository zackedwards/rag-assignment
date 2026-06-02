"""
retrieve.py
Embeds a question and finds the top k most similar chunks using pgvector's
cosine distance operator (<=>). k defaults to 5 and is configurable via the
TOP_K environment variable.
"""
import os
import asyncpg
from pgvector.asyncpg import register_vector

from ingest.embed import embed_query

TOP_K = int(os.getenv("TOP_K", "5"))


def dsn() -> str:
    return (
        f"postgresql://{os.environ['DB_USER']}:{os.environ['DB_PASSWORD']}"
        f"@{os.getenv('DB_HOST', '127.0.0.1')}:{os.getenv('DB_PORT', '5432')}"
        f"/{os.environ['DB_NAME']}"
    )


async def retrieve(question: str, k: int | None = None, series: str | None = None) -> list[dict]:
    """Return the k chunks most similar to the question.

    Pass series to restrict results to one manual, for example
    series="benefits_policy" to ignore the Claims Processing manual.
    """
    k = k or TOP_K
    q_emb = await embed_query(question)

    conn = await asyncpg.connect(dsn())
    await register_vector(conn)
    try:
        if series:
            rows = await conn.fetch(
                """
                SELECT source_file, series, page, chunk_index, content,
                       embedding <=> $1 AS distance
                FROM chunks
                WHERE series = $2
                ORDER BY embedding <=> $1
                LIMIT $3
                """,
                q_emb, series, k,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT source_file, series, page, chunk_index, content,
                       embedding <=> $1 AS distance
                FROM chunks
                ORDER BY embedding <=> $1
                LIMIT $2
                """,
                q_emb, k,
            )
    finally:
        await conn.close()

    return [
        {
            "source_file": r["source_file"],
            "series": r["series"],
            "page": r["page"],
            "chunk_index": r["chunk_index"],
            "content": r["content"],
            "distance": float(r["distance"]),
        }
        for r in rows
    ]
