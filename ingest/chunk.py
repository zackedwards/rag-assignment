"""
chunk.py
Splits page text into chunks of roughly 800 to 1200 characters with about
200 characters of overlap. Uses a simple split-on-paragraphs then merge
approach, which is plenty for this assignment.

Why character based and not token based. gemini-embedding-001 has a 2,048
token input limit. Roughly 4 characters per token means 2,048 tokens is about
8,000 characters, so a 1,200 character ceiling leaves a very wide safety margin
and the model will never silently truncate our chunks.
"""

CHUNK_SIZE = 1200      # target max characters per chunk
CHUNK_OVERLAP = 200    # characters carried from the end of one chunk into the next
MIN_CHUNK = 200        # do not emit fragments smaller than this on their own


def split_page(text: str) -> list[str]:
    """Turn one page of text into a list of chunk strings."""
    # paragraph break first, fall back to single newlines
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 1 <= CHUNK_SIZE:
            current = f"{current}\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            # if a single paragraph is itself too big, hard split it
            if len(para) > CHUNK_SIZE:
                for i in range(0, len(para), CHUNK_SIZE - CHUNK_OVERLAP):
                    chunks.append(para[i : i + CHUNK_SIZE])
                current = ""
            else:
                current = para
    if current:
        chunks.append(current)

    # add overlap by prefixing each chunk with the tail of the previous one
    overlapped = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            overlapped.append(chunk)
        else:
            tail = chunks[i - 1][-CHUNK_OVERLAP:]
            overlapped.append(f"{tail} {chunk}")
    return [c for c in overlapped if len(c) >= MIN_CHUNK or len(overlapped) == 1]


def chunk_pages(pages: list[dict]) -> list[dict]:
    """Expand page records into chunk records, carrying metadata forward."""
    records = []
    for page in pages:
        for idx, piece in enumerate(split_page(page["text"])):
            records.append(
                {
                    "source_file": page["source_file"],
                    "series": page["series"],
                    "page": page["page"],
                    "chunk_index": idx,
                    "content": piece,
                }
            )
    print(f"produced {len(records)} chunks")
    return records


if __name__ == "__main__":
    import os
    from load import load_corpus
    pages = load_corpus(os.getenv("CORPUS_DIR", "corpus"))
    chunks = chunk_pages(pages)
    if chunks:
        sizes = [len(c["content"]) for c in chunks]
        print(f"chunk size min {min(sizes)}, max {max(sizes)}, avg {sum(sizes)//len(sizes)}")
