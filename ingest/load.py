"""
load.py
Reads every PDF in the corpus folder and returns the text page by page,
keeping the filename and page number so we can cite sources later.
"""
from pathlib import Path
from pypdf import PdfReader


def series_for(filename: str) -> str:
    """Tag each file with the manual series it came from.

    bp102c..  -> Benefits Policy Manual
    clm104c.. -> Claims Processing Manual
    This is what lets us run a metadata filter at query time.
    """
    name = filename.lower()
    if name.startswith("bp102c"):
        return "benefits_policy"
    if name.startswith("clm104c"):
        return "claims_processing"
    return "other"


def load_pdf(path: Path) -> list[dict]:
    """Return one record per page with text and metadata."""
    pages = []
    reader = PdfReader(str(path))
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if not text:
            continue
        pages.append(
            {
                "source_file": path.name,
                "series": series_for(path.name),
                "page": page_number,
                "text": text,
            }
        )
    return pages


def load_corpus(corpus_dir: str) -> list[dict]:
    """Load every PDF in the corpus directory."""
    corpus = Path(corpus_dir)
    pdfs = sorted(corpus.glob("*.pdf"))
    all_pages = []
    for pdf in pdfs:
        try:
            all_pages.extend(load_pdf(pdf))
        except Exception as e:
            print(f"skipping {pdf.name}, could not read it ({e})")
    print(f"loaded {len(all_pages)} pages from {len(pdfs)} PDFs")
    return all_pages


if __name__ == "__main__":
    import os
    pages = load_corpus(os.getenv("CORPUS_DIR", "corpus"))
    if pages:
        sample = pages[0]
        print(f"first page from {sample['source_file']} ({sample['series']})")
        print(sample["text"][:300])
