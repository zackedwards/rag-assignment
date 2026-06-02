# Medicare RAG on GCP

A retrieval augmented question answering system over the CMS Medicare manuals. It splits the manual PDFs into chunks, embeds them with `gemini-embedding-001`, stores them in Postgres with `pgvector`, retrieves the most relevant chunks for a question, and asks Gemini 2.5 Flash for a grounded answer with citations.

## What you need first

A GCP project with billing on, a Cloud SQL Postgres 15 instance, and the APIs `aiplatform`, `sqladmin`, and `run` enabled. Python 3.11 or newer. The Cloud SQL Auth Proxy running so the database is reachable on `localhost:5432`.

## Setup

```bash
# 1. install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. configure environment
cp .env.example .env
# edit .env, fill in GCP_PROJECT, DB_NAME, DB_PASSWORD

# 3. start the Cloud SQL Auth Proxy in its own terminal
./cloud-sql-proxy YOUR_PROJECT:us-central1:YOURID-rag-db --port 5432

# 4. create the schema, extension, and index
psql "postgresql://postgres:$DB_PASSWORD@127.0.0.1:5432/$DB_NAME" -f sql/schema.sql
```

## Download the corpus

```bash
mkdir -p corpus && cd corpus
for i in $(seq -w 1 16); do curl -fsSL -O "https://www.cms.gov/Regulations-and-Guidance/Guidance/Manuals/Downloads/bp102c${i}.pdf" || echo "ch${i} skipped"; done
for i in $(seq -w 1 34); do curl -fsSL -O "https://www.cms.gov/Regulations-and-Guidance/Guidance/Manuals/Downloads/clm104c${i}.pdf" || echo "ch${i} skipped"; done
cd ..
```

## Run ingestion

```bash
# smoke test on 10 PDFs first to confirm the whole path works
python -m ingest.run_ingest --limit 10

# then the full corpus once
python -m ingest.run_ingest
```

## Start the API

```bash
uvicorn serve.api:app --host 0.0.0.0 --port 8080
```

Query it.

```bash
curl -s -X POST localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many days must a patient be hospitalized to qualify for skilled nursing facility care?"}' | python -m json.tool
```

## Run the eval

With the API still running, in another terminal.

```bash
python -m eval.run_eval
```

Scores land in `eval/eval_results.json`.

## Environment variables

| Variable | Meaning |
| --- | --- |
| GCP_PROJECT | the shared GCP project id |
| GCP_REGION | region for Vertex calls, us-central1 |
| EMBED_DIM | embedding size, 768 |
| EMBED_CONCURRENCY | max concurrent embed calls, 15 |
| TOP_K | chunks retrieved per question, 5 |
| DB_HOST DB_PORT DB_NAME DB_USER DB_PASSWORD | database connection |
| CORPUS_DIR | folder holding the PDFs |
| API_URL | endpoint the eval hits |

## Design choices

Chunk size is about 1,200 characters with 200 of overlap, split on paragraph breaks then merged. That stays far under the 2,048 token input limit so the embedding model never silently truncates a chunk.

Embeddings are 768 dimensional. The model defaults to 3072, but pgvector HNSW indexes cap at 2000 dimensions, so 3072 cannot be indexed. 768 keeps almost all of the quality while fitting comfortably under the cap. Because the model only auto normalizes at 3072, the code normalizes the 768 vectors to unit length before storing them.

The index is HNSW with cosine distance. HNSW gives fast approximate nearest neighbor search at this corpus size, and cosine matches the normalized embeddings. Top k defaults to 5, which gives Gemini enough context without diluting the prompt.

Eval uses Gemini 2.5 Flash as a judge with a JSON only scoring prompt, graded on accuracy and faithfulness from 0 to 5.

## AI tools used

Reference scaffold drafted with help from an LLM, then reviewed and adapted.
