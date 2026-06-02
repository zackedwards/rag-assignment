-- schema.sql
-- Run this against your database (YOURID_ragdb) during setup, before ingestion.
-- In Cloud Shell:  psql "$DB_URL" -f sql/schema.sql
-- or paste it into a psql session opened with gcloud sql connect.

-- 1. Turn on pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. The chunks table.
--    embedding is vector(768) on purpose. gemini-embedding-001 defaults to
--    3072 dimensions, but pgvector HNSW indexes cap at 2000, so we request
--    768 dimensional embeddings in embed.py and store them here.
CREATE TABLE IF NOT EXISTS chunks (
    id           BIGSERIAL PRIMARY KEY,
    source_file  TEXT        NOT NULL,
    series       TEXT        NOT NULL,
    page         INT,
    chunk_index  INT         NOT NULL,
    content      TEXT        NOT NULL,
    embedding    vector(768) NOT NULL,
    embedded_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 3. HNSW index with cosine distance.
--    HNSW gives fast approximate nearest neighbor search at this data size.
--    Cosine matches how our normalized embeddings encode similarity.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- A plain index on series makes the metadata filter cheap.
CREATE INDEX IF NOT EXISTS chunks_series_idx ON chunks (series);

-- 4. Metadata filter demo for Part 2.
--    This restricts results to the Benefits Policy manual only and ignores
--    the Claims Processing manual, proving metadata is stored and usable.
--    (Run after ingestion has populated the table.)
--
-- SELECT source_file, series, page, left(content, 80) AS preview
-- FROM chunks
-- WHERE series = 'benefits_policy'
-- LIMIT 10;
