"""
store.py
Inserts chunk text, metadata, and the embedding vector into Postgres.

Uses ON CONFLICT DO NOTHING against a unique key of
(source_file, page, chunk_index) so that re running after a partial run just
fills in the gaps instead of erroring or duplicating. This is what makes the
ingestion resumable if Cloud Shell disconnects mid run.
"""
import os
import asyncpg
from pgvector.asyncpg import register_vector


def dsn() -> str:
    return (
        f"postgresql://{os.environ['DB_USER']}:{os.environ['DB_PASSWORD']}"
        f"@{os.getenv('DB_HOST', '127.0.0.1')}:{os.getenv('DB_PORT', '5432')}"
        f"/{os.environ['DB_NAME']}"
    )


async def connect() -> asyncpg.Connection:
    conn = await asyncpg.connect(dsn())
    await register_vector(conn)
    return conn


async def count_rows() -> int:
    conn = await connect()
    try:
        return await conn.fetchval("SELECT count(*) FROM chunks")
    finally:
        await conn.close()


async def existing_keys() -> set[tuple]:
    """Return the (source_file, page, chunk_index) already stored, so the
    orchestrator can skip re embedding work that is already done."""
    conn = await connect()
    try:
        rows = await conn.fetch("SELECT source_file, page, chunk_index FROM chunks")
        return {(r["source_file"], r["page"], r["chunk_index"]) for r in rows}
    finally:
        await conn.close()


async def store_chunks(records: list[dict]) -> None:
    if not records:
        return
    conn = await connect()
    try:
        await conn.executemany(
            """
            INSERT INTO chunks
                (source_file, series, page, chunk_index, content, embedding)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (source_file, page, chunk_index) DO NOTHING
            """,
            [
                (
                    r["source_file"],
                    r["series"],
                    r["page"],
                    r["chunk_index"],
                    r["content"],
                    r["embedding"],
                )
                for r in records
            ],
        )
    finally:
        await conn.close()
