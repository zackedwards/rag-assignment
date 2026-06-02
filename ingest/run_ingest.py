"""
run_ingest.py
Runs the whole pipeline. load -> chunk -> embed -> store.

It embeds and stores in batches so progress persists. If it dies partway
through, just run it again and it resumes from where it stopped, because the
chunks already in the database are skipped before any embedding happens.

Usage
    python -m ingest.run_ingest
    python -m ingest.run_ingest --limit 10        # test on 10 PDFs first
    python -m ingest.run_ingest --batch-size 200  # tune the batch size

Run with --limit 10 the first time so you only spend a few embedding calls
while you confirm everything works end to end.
"""
import argparse
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

from ingest.load import load_corpus, load_pdf
from ingest.chunk import chunk_pages
from ingest.embed import embed_chunks
from ingest.store import store_chunks, existing_keys

load_dotenv()


async def main(limit: int | None, batch_size: int) -> None:
    corpus_dir = os.getenv("CORPUS_DIR", "corpus")
    if limit:
        pdfs = sorted(Path(corpus_dir).glob("*.pdf"))[:limit]
        pages = []
        for pdf in pdfs:
            pages.extend(load_pdf(pdf))
        print(f"limited run, loaded {len(pages)} pages from {len(pdfs)} PDFs")
    else:
        pages = load_corpus(corpus_dir)

    chunks = chunk_pages(pages)

    # skip anything already stored, so a rerun resumes instead of redoing work
    done = await existing_keys()
    todo = [c for c in chunks if (c["source_file"], c["page"], c["chunk_index"]) not in done]
    if done:
        print(f"{len(done)} chunks already stored, {len(todo)} left to do")
    if not todo:
        print("nothing left to ingest")
        return

    total = len(todo)
    for start in range(0, total, batch_size):
        batch = todo[start : start + batch_size]
        embedded = await embed_chunks(batch)
        await store_chunks(embedded)
        print(f"stored {min(start + batch_size, total)}/{total}")

    print("ingestion complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only process N PDFs")
    parser.add_argument("--batch-size", type=int, default=200, help="chunks per embed and store batch")
    args = parser.parse_args()
    asyncio.run(main(args.limit, args.batch_size))
