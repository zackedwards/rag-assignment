"""
api.py
FastAPI app with a POST /query endpoint.

Start it with
    uvicorn serve.api:app --host 0.0.0.0 --port 8080
"""
import re
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from serve.retrieve import retrieve
from serve.generate import generate_answer

load_dotenv()

app = FastAPI(title="Medicare RAG")


class QueryRequest(BaseModel):
    question: str
    series: str | None = None  # optional metadata filter


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    retrieved_chunks: list[str]


def extract_citations(answer: str) -> list[str]:
    """Pull [source: ...] markers out of the answer text, de-duplicated."""
    found = re.findall(r"\[source:\s*([^\]]+)\]", answer)
    seen = []
    for f in found:
        cleaned = f.strip()
        if cleaned not in seen:
            seen.append(cleaned)
    return seen


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    chunks = await retrieve(req.question, series=req.series)
    answer = await generate_answer(req.question, chunks)
    return QueryResponse(
        answer=answer,
        citations=extract_citations(answer),
        retrieved_chunks=[c["content"] for c in chunks],
    )
