"""PDF handling via PyMuPDF.

PyMuPDF does double duty here — text-layer extraction and page rasterization for
the OCR path — which is why it was chosen over pdfplumber. One self-contained
Windows wheel, no poppler-adjacent tooling to install alongside.
"""

from __future__ import annotations

from pathlib import Path


def extract_text(path: Path) -> tuple[str, int]:
    """Pull the text layer out of a PDF. Returns ``(text, page_count)``."""
    import fitz

    with fitz.open(str(path)) as doc:
        pages = [page.get_text("text") for page in doc]
    return "\n\n".join(pages).strip(), len(pages)


def rasterize(path: Path, out_dir: Path, dpi: int = 200) -> list[Path]:
    """Render each page to PNG so PaddleOCR can read a scanned PDF.

    200 dpi is the useful floor for OCR on 10-12pt body text; below it, character
    recognition degrades sharply on the small print CVs like to use for contact
    details.
    """
    import fitz

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with fitz.open(str(path)) as doc:
        for i, page in enumerate(doc):
            target = out_dir / f"page_{i:03d}.png"
            page.get_pixmap(dpi=dpi).save(str(target))
            written.append(target)
    return written
