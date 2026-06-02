"""
run_ingest.py
Runs the whole pipeline. load -> chunk -> embed -> store.

Usage
    python -m ingest.run_ingest
    python -m ingest.run_ingest --limit 10     # test on 10 PDFs first

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
from ingest.store import store_chunks, count_rows

load_dotenv()


async def main(limit: int | None) -> None:
    existing = await count_rows()
    if existing > 0:
        print(f"chunks table already populated with {existing} rows, nothing to do")
        return

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
    chunks = await embed_chunks(chunks)
    await store_chunks(chunks)
    print("ingestion complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only process N PDFs")
    args = parser.parse_args()
    asyncio.run(main(args.limit))
