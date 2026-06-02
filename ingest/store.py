"""
store.py
Inserts chunk text, metadata, and the embedding vector into Postgres.

Assumes the table and pgvector extension already exist (created by
sql/schema.sql during setup). If the table already has rows, we skip
re-ingestion so we do not burn embedding calls twice.
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


async def store_chunks(records: list[dict]) -> None:
    conn = await connect()
    try:
        existing = await conn.fetchval("SELECT count(*) FROM chunks")
        if existing and existing > 0:
            print(f"table already has {existing} rows, skipping insert")
            return
        await conn.executemany(
            """
            INSERT INTO chunks
                (source_file, series, page, chunk_index, content, embedding)
            VALUES ($1, $2, $3, $4, $5, $6)
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
        print(f"inserted {len(records)} rows into chunks")
    finally:
        await conn.close()
