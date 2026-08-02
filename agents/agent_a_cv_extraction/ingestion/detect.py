"""File-type detection.

Classification drives routing, so it reads magic bytes rather than trusting the
extension — a .pdf that is really a JPEG would otherwise fail deep inside PyMuPDF
with a confusing error.

The interesting case is a PDF with no usable text layer. That is a scan wearing a
PDF wrapper, and it must go to OCR, not to the text extractor that would return
an empty string and hand the LLM nothing to work with.
"""

from __future__ import annotations

from pathlib import Path

import filetype

from ..state import DocKind

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def sniff_mime(path: Path) -> str | None:
    kind = filetype.guess(str(path))
    return kind.mime if kind else None


# What the user is told when the document needs OCR and this build has none.
# Written for them, not for the log: it names what the file is, says plainly that
# this deployment cannot read it, and points at the way out — the manual-entry
# screen the frontend already has for `agent_a_unreadable_document`.
_NO_OCR = (
    "this looks like a scanned or photographed document, and this deployment "
    "cannot read those — upload a PDF with selectable text, or enter your "
    "details by hand"
)


def _can_ocr() -> bool:
    # Imported here rather than at module scope: `ocr.py` pulls in shared.config
    # for paddle's env-var side effects, and detect_kind is on the hot path for
    # every file including the text-layer ones that never touch OCR.
    from .ocr import ocr_available

    return ocr_available()


def detect_kind(
    path: Path,
    *,
    min_chars: int = 200,
    min_chars_per_page: int = 50,
) -> tuple[DocKind, str | None]:
    """Classify a file. Returns ``(kind, detail)`` where detail explains the call."""
    path = Path(path)
    if not path.exists():
        return "unsupported", f"file not found: {path}"

    mime = sniff_mime(path)

    # A plain .txt has no magic bytes; filetype returns None. Useful for tests
    # and for feeding pre-extracted text through the pipeline.
    if mime is None and path.suffix.lower() in {".txt", ".md"}:
        return "text", "plain text file"

    if mime == "application/pdf":
        try:
            import fitz
        except ImportError:
            return "unsupported", "PyMuPDF not installed; cannot read PDFs"

        with fitz.open(str(path)) as doc:
            page_count = doc.page_count
            total = sum(len(page.get_text("text").strip()) for page in doc)

        if page_count == 0:
            return "unsupported", "PDF has no pages"
        if total >= min_chars and (total / page_count) >= min_chars_per_page:
            return "pdf_text", f"{total} chars across {page_count} page(s)"
        if not _can_ocr():
            return "unsupported", _NO_OCR
        return "pdf_scanned", (
            f"only {total} chars across {page_count} page(s) — treating as scanned"
        )

    if (mime and mime.startswith("image/")) or path.suffix.lower() in IMAGE_EXTENSIONS:
        if not _can_ocr():
            return "unsupported", _NO_OCR
        return "image", mime or path.suffix.lower()

    return "unsupported", f"unrecognised type: {mime or path.suffix or 'unknown'}"
