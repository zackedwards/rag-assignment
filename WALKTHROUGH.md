# Instructor Runbook, Week 3 RAG on GCP

This is your step by step script for the live walkthrough, and for the dry run you do today. Follow it top to bottom. Every command says where to run it. The one thing that will quietly break a student's afternoon is flagged near the top, so read the gotcha section before you start.

## The single most important gotcha

The assignment says use `gemini-embedding-001` and build an HNSW index with `vector_cosine_ops`. Those two instructions fight each other. The model returns 3072 dimensional vectors by default, and pgvector HNSW indexes refuse anything over 2000 dimensions. A student who follows the text literally will hit this exact error when they run the index command.

```
ERROR: column cannot have more than 2000 dimensions for hnsw index
```

The fix is baked into the reference code. We request `output_dimensionality=768` on every embed call and make the column `vector(768)`. Because the model only auto normalizes at 3072, we also normalize the 768 vectors to unit length. Tell students this up front and they will sail past the most common failure of the whole assignment.

Two smaller gotchas, both also handled in the reference code. The model needs a Postgres password that nobody sets by default, and the Python client cannot reach Cloud SQL without the Auth Proxy running. Both are steps below.

## What to have open before you start

Open these tabs.

1. Cloud Shell, the terminal that comes with the GCP console. This is where almost every command runs. Open a second Cloud Shell tab too, because you will run the API in one and the eval in the other.
2. The Cloud SQL page in the GCP console, so you can show the instance, click into the database, and prove rows exist visually.
3. The Vertex AI page, just to confirm the API is enabled if anyone asks.
4. The assignment page in Canvas.
5. This runbook.
6. A local terminal or editor open to the reference repo, so you can show the code files as you talk through each part.

Confirm you are in the right project before anything else. Run this in Cloud Shell.

```bash
gcloud config get-value project
```

If it is wrong, set it.

```bash
gcloud config set project YOUR_SHARED_PROJECT_ID
```

## A note on your dry run ID

For your own dry run today, do not use a student style ID that might collide. Use something obviously yours, for example prefix everything with `instr`. So your instance becomes `instr-rag-db` and your database `instr_ragdb`. When you teach tomorrow, students substitute their real Stevens ID like `jedwards3`. Wherever you see YOURID below, that is the substitution point.

---

## Phase 0, setup, about 25 minutes

### Step 0.1 Enable the APIs and start the database building

Run in Cloud Shell. The database takes 5 to 10 minutes, so kick it off first and keep reading.

```bash
gcloud services enable aiplatform.googleapis.com sqladmin.googleapis.com run.googleapis.com

gcloud sql instances create YOURID-rag-db \
  --database-version=POSTGRES_15 \
  --tier=db-f1-micro \
  --region=us-central1

gcloud sql databases create YOURID_ragdb \
  --instance=YOURID-rag-db
```

### Step 0.2 Set the Postgres password

This is the step the assignment never mentions. Without it the Python code cannot log in. Run in Cloud Shell.

```bash
gcloud sql users set-password postgres \
  --instance=YOURID-rag-db \
  --password='ChangeThisToSomethingStrong123'
```

Remember whatever you type here. It goes into `.env` as DB_PASSWORD.

### Step 0.3 Get the code onto Cloud Shell

You can either clone your reference repo or copy the files in. Once the files are present, install dependencies. Run in Cloud Shell, inside the project folder.

```bash
cd rag-assignment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then create your env file and edit it.

```bash
cp .env.example .env
nano .env
```

Fill in GCP_PROJECT, set DB_NAME to YOURID_ragdb, and set DB_PASSWORD to the password from Step 0.2. Save and exit.

### Step 0.4 Start the Cloud SQL Auth Proxy

This creates a secure tunnel so the database looks like it is running on localhost. Run in Cloud Shell. Download the proxy once, then run it.

```bash
curl -o cloud-sql-proxy https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.11.0/cloud-sql-proxy.linux.amd64
chmod +x cloud-sql-proxy

# leave this running. open a NEW Cloud Shell tab for the next steps
./cloud-sql-proxy YOUR_SHARED_PROJECT_ID:us-central1:YOURID-rag-db --port 5432
```

Leave that tab alone. The proxy must stay running the whole time you work. Do everything else in a different tab.

### Step 0.5 Create the schema, the extension, and the index

This is technically Part 2 of the assignment, but the table has to exist before you can ingest into it, so we do it now. Many students get confused by the numbering. Call that out. Run in your second Cloud Shell tab, with the venv active.

```bash
source .venv/bin/activate
source .env   # picks up DB_PASSWORD and DB_NAME

psql "postgresql://postgres:${DB_PASSWORD}@127.0.0.1:5432/${DB_NAME}" -f sql/schema.sql
```

You should see CREATE EXTENSION, CREATE TABLE, and CREATE INDEX succeed with no dimension error. That success is the proof point that the 768 fix works. Pause here and show it.

### Step 0.6 Download the corpus

Run in your second Cloud Shell tab.

```bash
mkdir -p corpus && cd corpus
for i in $(seq -w 1 16); do curl -fsSL -O "https://www.cms.gov/Regulations-and-Guidance/Guidance/Manuals/Downloads/bp102c${i}.pdf" || echo "ch${i} skipped"; done
for i in $(seq -w 1 34); do curl -fsSL -O "https://www.cms.gov/Regulations-and-Guidance/Guidance/Manuals/Downloads/clm104c${i}.pdf" || echo "ch${i} skipped"; done
echo "downloaded $(ls *.pdf | wc -l) PDFs"
cd ..
```

Expect roughly 45 to 55 PDFs. If the CMS links have changed and you get far fewer, check the CMS Internet Only Manuals page for current links. Download the corpus during your dry run today so you find out before class if a link is dead.

Also drop the provided `golden.jsonl` into the `eval` folder now.

```bash
ls eval/golden.jsonl
```

---

## Phase 1, ingestion, about 15 minutes live

Walk the four files in order while you talk. load reads PDFs, chunk splits text, embed turns text into vectors, store writes to Postgres.

### Step 1.1 Smoke test on ten PDFs

Always do the limited run first. It spends only a handful of embedding calls and proves the entire path before you commit to the full corpus. Run in your second tab.

```bash
python -m ingest.run_ingest --limit 10
```

You want to see it load pages, produce chunks, embed them at 768 dimensions, and insert rows. If this works, the full run is just more of the same.

### Step 1.2 Full ingestion

Run once. This is a few minutes. Run in your second tab.

```bash
python -m ingest.run_ingest
```

The code checks the row count first and skips if the table already has data, so re running it is safe and cheap. That is your guard against burning embedding calls twice.

### Step 1.3 Show the rows, the screenshot deliverable

Prove it two ways. First in the terminal.

```bash
psql "postgresql://postgres:${DB_PASSWORD}@127.0.0.1:5432/${DB_NAME}" \
  -c "SELECT count(*) FROM chunks;"
```

Then in the Cloud SQL console tab, open the database and run the same count, or browse the table. That console view is the screenshot students submit for Part 1.

---

## Phase 2, vector store, about 10 minutes live

The table, extension, and HNSW index already exist from Step 0.5, so here you explain the choices and demonstrate the metadata filter.

### Step 2.1 Explain the index and distance choice

Point at `sql/schema.sql`. HNSW for fast approximate search at this size, cosine distance because the embeddings are normalized. That is the one or two sentences the README asks for.

### Step 2.2 Demonstrate the metadata filter

This is the Part 2 screenshot. It proves metadata is stored and usable at query time. Run in your second tab.

```bash
psql "postgresql://postgres:${DB_PASSWORD}@127.0.0.1:5432/${DB_NAME}" -c \
"SELECT source_file, series, page, left(content, 60) AS preview
 FROM chunks
 WHERE series = 'benefits_policy'
 LIMIT 10;"
```

Every row should be a bp102c file, no claims processing files. That is the proof. You can run it again with `series = 'claims_processing'` to show the other slice.

---

## Phase 3, retrieval and generation, about 15 minutes live

### Step 3.1 Start the API

Run in your second tab and leave it running.

```bash
uvicorn serve.api:app --host 0.0.0.0 --port 8080
```

### Step 3.2 Hit the endpoint with three questions

Open a third tab for this, with the venv active. Run three questions so students see the full JSON shape, including the I don't have enough information refusal.

```bash
# a question the manuals answer
curl -s -X POST localhost:8080/query -H "Content-Type: application/json" \
  -d '{"question": "How many days must a patient be hospitalized to qualify for skilled nursing facility coverage?"}' | python -m json.tool

# a question scoped to one manual using the metadata filter
curl -s -X POST localhost:8080/query -H "Content-Type: application/json" \
  -d '{"question": "What does the manual say about durable medical equipment?", "series": "benefits_policy"}' | python -m json.tool

# a question the manuals do NOT cover, to show the refusal
curl -s -X POST localhost:8080/query -H "Content-Type: application/json" \
  -d '{"question": "What is the capital of France?"}' | python -m json.tool
```

The first two should return an answer, a citations list, and the retrieved chunks. The third should return I don't have enough information. Save these three responses, they are the Part 3 deliverable.

---

## Phase 4, evaluation, about 10 minutes live

Keep the API running in its tab. Run the eval in another tab with the venv active.

```bash
python -m eval.run_eval
```

It walks all 25 golden questions, calls the API, then asks Gemini 2.5 Flash to score accuracy and faithfulness from 0 to 5, and writes `eval/eval_results.json`. It is hard capped at 25 with no retry loop, so it stays well under a dollar. Open the results file and show the two averages.

```bash
cat eval/eval_results.json | python -m json.tool | head -40
```

---

## Phase 5, packaging, a few minutes

Pin the exact versions for the rubric, then zip.

```bash
pip freeze > requirements.txt

cd ..
zip -r YOURID-rag-assignment.zip rag-assignment \
  -x "*/.venv/*" "*/corpus/*" "*/__pycache__/*"
```

Exclude the venv and the corpus PDFs, they are large and not part of the deliverable. Confirm the zip contains README, requirements, .env.example, the ingest, serve, eval, and sql folders, and eval_results.json.

---

## Teardown, do not skip

The biggest cost risk is leaving Cloud SQL running. Stop it the moment you finish.

```bash
gcloud sql instances patch YOURID-rag-db --activation-policy=NEVER
```

To bring it back for class tomorrow.

```bash
gcloud sql instances patch YOURID-rag-db --activation-policy=ALWAYS
```

For final cleanup after the assignment, delete the instance entirely, then screenshot the empty resource list.

```bash
gcloud sql instances delete YOURID-rag-db
```

---

## Suggested live session sequencing

You have a 60 minute office hours slot. A workable shape.

First 5 minutes, frame the assignment and call out the 768 dimension gotcha before anyone writes code. Next 10 minutes, run Phase 0 setup live, kicking off the database first so it provisions while you talk. While it builds, walk the code files. Around 15 minutes in, the database is ready, run the schema and the limited ingest. Next 15 minutes, full ingest, show the rows, demonstrate the metadata filter. Next 15 minutes, start the API, run the three example queries, run the eval. Last 5 minutes, packaging and teardown, and remind everyone to stop their instance.

If you are tight on time, the parts students most need to see live are the schema creation succeeding with 768, the three example queries, and the teardown command. Everything else they can follow from the README.

---

## Troubleshooting table

| Symptom | Cause | Fix |
| --- | --- | --- |
| column cannot have more than 2000 dimensions | embedding is 3072, the default | confirm EMBED_DIM is 768 and the column is vector(768) |
| connection refused on 5432 | Auth Proxy not running | start the proxy in its own tab, leave it running |
| password authentication failed | postgres password never set | run the set-password step 0.2 |
| 429 during embedding | too many concurrent calls | lower EMBED_CONCURRENCY to 5 to 10, backoff already handles retries |
| answers have no citations | system prompt or context issue | confirm generate.py system prompt and that chunks are passed in |
| relation chunks does not exist | schema run against wrong database | run schema.sql against YOURID_ragdb, not the default postgres db |
| queries feel slow | HNSW index missing | confirm the index was created in schema.sql |
| eval bill creeping up | API or DB left running | stop the instance, the eval has no retry loop by design |

## Dry run checklist for today

Tick these off so tomorrow has no surprises.

1. Instance and database created, password set.
2. Proxy connects, schema runs clean with no dimension error.
3. Corpus downloads at least 40 PDFs from the current CMS links.
4. Limited ingest works, then full ingest populates the table.
5. Metadata filter returns only one series.
6. All three example queries behave, including the refusal.
7. Eval finishes 25 questions and writes results.
8. Zip builds without the venv or corpus.
9. Instance stopped at the end.
