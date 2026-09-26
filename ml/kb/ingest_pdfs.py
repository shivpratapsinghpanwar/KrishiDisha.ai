"""Turn packages-of-practices PDFs (ICAR / state agricultural universities) into knowledge-base chunks.

    python -m ml.kb.ingest_pdfs --pdf docs/pop/ICAR-IIWBR_wheat_2024.pdf --source "ICAR-IIWBR Wheat Package of Practices 2024" \
        --url https://iiwbr.icar.gov.in/... --crop wheat --out data/knowledge/chunks/iiwbr_wheat.jsonl
    python -m ml.kb.ingest_pdfs --pdf docs/pop/*.pdf --source-from-filename --out data/knowledge/chunks/

Each chunk is ~300 words with a 40-word overlap and carries source, page, crop and url so the assistant can
cite it. ``KnowledgeBase`` loads every ``data/knowledge/chunks/*.jsonl`` as documents of kind ``pop``.
PDFs themselves are NOT committed (government publications, retrieval-only) - keep them under ``docs/pop/``
which is gitignored.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

WORDS_PER_CHUNK = 300
OVERLAP = 40


def extract_pages(pdf: Path) -> list[str]:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{2,}", "\n", text)
        pages.append(text.strip())
    return pages


def chunk_pages(pages: list[str], words_per_chunk: int = WORDS_PER_CHUNK, overlap: int = OVERLAP):
    """Yield (page_start, page_end, text) chunks over the concatenated page stream."""
    tokens: list[tuple[int, str]] = []
    for pno, text in enumerate(pages, start=1):
        for w in text.split():
            tokens.append((pno, w))
    i = 0
    while i < len(tokens):
        window = tokens[i:i + words_per_chunk]
        if len(window) < 40 and i > 0:
            break
        yield window[0][0], window[-1][0], " ".join(w for _, w in window)
        i += words_per_chunk - overlap


def ingest(pdf: Path, source: str, url: str | None, crop: str | None, licence: str) -> list[dict]:
    pages = extract_pages(pdf)
    rows = []
    for k, (p0, p1, text) in enumerate(chunk_pages(pages)):
        title = f"{source} (p. {p0}{'' if p0 == p1 else f'-{p1}'})"
        rows.append({"id": f"pop:{pdf.stem}:{k}", "title": title, "text": text, "source": source, "url": url,
                     "crop": crop, "page_start": p0, "page_end": p1, "licence": licence, "kind": "pop"})
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pdf", nargs="+", required=True, help="PDF paths or globs")
    p.add_argument("--source", help="human-readable source name (one PDF)")
    p.add_argument("--source-from-filename", action="store_true")
    p.add_argument("--url")
    p.add_argument("--crop")
    p.add_argument("--licence", default="Government of India / state agricultural university publication; retrieval only")
    p.add_argument("--out", type=Path, required=True, help="a .jsonl file (one PDF) or a directory")
    args = p.parse_args(argv)

    pdfs = [Path(f) for pattern in args.pdf for f in glob.glob(pattern)]
    if not pdfs:
        raise SystemExit("no PDFs matched")
    total = 0
    for pdf in pdfs:
        source = args.source or (pdf.stem.replace("_", " ").replace("-", " ") if args.source_from_filename else pdf.stem)
        rows = ingest(pdf, source, args.url, args.crop, args.licence)
        out = args.out / f"{pdf.stem}.jsonl" if args.out.is_dir() or args.out.suffix != ".jsonl" else args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        total += len(rows)
        print(f"{pdf.name}: {len(rows)} chunks -> {out}")
    print(f"{total} chunks total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
