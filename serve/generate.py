"""
generate.py
Sends the retrieved chunks to Gemini 2.5 Flash as context and gets back a
grounded answer with inline citations.
"""
import asyncio
import os
import random
from google import genai
from google.genai import types

GEN_MODEL = os.getenv("GEN_MODEL", "gemini-2.5-flash")

SYSTEM_PROMPT = (
    "You answer questions about the Medicare manuals using only the context "
    "provided below. Follow these rules strictly. "
    "Only use facts found in the context. Do not use outside knowledge. "
    "If the context does not contain the answer, reply exactly with "
    "'I don't have enough information'. "
    "Every claim must include an inline citation in the format "
    "[source: filename, section] using the source_file and page from the "
    "context block the claim came from."
)

_client = None


def client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(
            vertexai=True,
            project=os.environ["GCP_PROJECT"],
            location=os.environ["GCP_REGION"],
        )
    return _client


def build_context(chunks: list[dict]) -> str:
    blocks = []
    for c in chunks:
        header = f"[source: {c['source_file']}, page {c['page']}]"
        blocks.append(f"{header}\n{c['content']}")
    return "\n\n---\n\n".join(blocks)


async def generate_answer(question: str, chunks: list[dict]) -> str:
    context = build_context(chunks)
    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer using only the context above, with inline citations."
    )
    # retry with backoff so a transient 429 or 5xx does not 500 the endpoint
    for attempt in range(5):
        try:
            response = await client().aio.models.generate_content(
                model=GEN_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.3,
                    max_output_tokens=800,
                ),
            )
            return (response.text or "").strip()
        except Exception:
            if attempt < 4:
                await asyncio.sleep(min(30, 2 ** attempt) + random.uniform(0, 1))
            else:
                raise
