"""Document classification, and what happens when a build has no OCR.

PaddleOCR is 1.2 GB installed and wants ~2 GB of RAM, so whether it ships is a
per-image decision (`--build-arg WITH_OCR`). That makes "can this build read a
scanned page?" a real runtime question, and the answer something a user can be
given rather than a traceback in a log.

Both branches are forced here rather than inferred from what happens to be
installed, so this suite says the same thing on a laptop with paddle and in CI
without it.
"""

from __future__ import annotations

import pytest

from agents.agent_a_cv_extraction.ingestion.detect import detect_kind

fitz = pytest.importorskip("fitz", reason="PyMuPDF builds the fixtures")


@pytest.fixture
def text_pdf(tmp_path):
    """A PDF with a real text layer — the case that never needs OCR."""
    doc = fitz.open()
    page = doc.new_page()
    # A text BOX, not insert_text at a point: a point writes one unwrapped line
    # that runs off the page, and the threshold is 200 characters.
    page.insert_textbox(fitz.Rect(56, 56, 540, 760),
                        "Amin Mohamed Amin\n" + ("Python SQL C++ data analysis " * 20),
                        fontsize=11)
    path = tmp_path / "cv.pdf"
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def scanned_pdf(tmp_path):
    """A PDF with no extractable text — what a scan or a phone photo looks like."""
    doc = fitz.open()
    doc.new_page()
    path = tmp_path / "scan.pdf"
    doc.save(str(path))
    doc.close()
    return path


def _ocr(monkeypatch, available: bool) -> None:
    monkeypatch.setattr(
        "agents.agent_a_cv_extraction.ingestion.ocr.ocr_available",
        lambda: available)


# ---------------------------------------------------------------------------
def test_a_text_layer_pdf_never_needs_ocr(monkeypatch, text_pdf):
    """Why a no-OCR build is worth having at all: an ordinary CV exported from
    Word or LaTeX has a text layer and reads identically either way."""
    for available in (True, False):
        _ocr(monkeypatch, available)
        assert detect_kind(text_pdf)[0] == "pdf_text"


def test_a_scanned_pdf_is_read_when_ocr_is_present(monkeypatch, scanned_pdf):
    _ocr(monkeypatch, True)
    kind, detail = detect_kind(scanned_pdf)
    assert kind == "pdf_scanned" and "scanned" in (detail or "")


def test_a_scanned_pdf_is_refused_by_name_when_ocr_is_absent(monkeypatch, scanned_pdf):
    """Refused, not crashed, and the reason is written for the person holding the
    file. It travels the existing `unsupported` path, which Agent A reports as
    `agent_a_unreadable_document` — the code the UI's re-upload / manual-entry
    screen is gated on. An ImportError three layers down reaches nobody."""
    _ocr(monkeypatch, False)
    kind, detail = detect_kind(scanned_pdf)
    assert kind == "unsupported"
    assert "scanned" in detail and "enter your details by hand" in detail


def test_an_image_is_refused_the_same_way(monkeypatch, tmp_path):
    png = tmp_path / "cv.png"
    png.write_bytes(bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000a49444154789c6300010000050001" "0d0a2db4" "0000000049454e44ae426082"))
    _ocr(monkeypatch, False)
    kind, detail = detect_kind(png)
    assert kind == "unsupported" and "scanned" in detail

    _ocr(monkeypatch, True)
    assert detect_kind(png)[0] == "image"


def test_ocr_available_does_not_import_paddle():
    """It is called for every file, including the text-layer ones. Importing
    paddle to answer costs seconds and hundreds of megabytes of RSS, so the check
    is `find_spec` — and answering must not itself load paddle."""
    import sys

    from agents.agent_a_cv_extraction.ingestion.ocr import ocr_available

    before = "paddle" in sys.modules
    ocr_available()
    assert ("paddle" in sys.modules) == before
