"""Multi-file documents: one CV spread across several files.

Photographing a CV a page at a time is the normal way a phone-submitted document
arrives, so a "document" is an ordered list of parts rather than a single file.
Order is the property that matters most here — it is the page order, and getting
it wrong silently scrambles what the LLM reads.
"""

from __future__ import annotations

import json
from pathlib import Path

from agents.agent_a_cv_extraction.graph import build_graph
from agents.agent_a_cv_extraction.nodes.ingest import _assemble, _dominant_kind
from shared.config import Config
from tests.fake_llm import FakeStructuredLLM


def write_pages(tmp_path: Path) -> list[Path]:
    """A CV split across three text files, as three photographed pages would be."""
    pages = [
        "Sara Al-Balushi\nEmail: sara.b@squ.edu.om\n",
        "EDUCATION\nBSc Computer Science, Sultan Qaboos University, 2020-2024\nGPA: 3.71\n",
        "SKILLS\nPython, PyTorch, Teamwork\n",
    ]
    paths = []
    for i, text in enumerate(pages):
        path = tmp_path / f"cv_page{i}.txt"
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    return paths


def run_multi(tmp_path: Path, cv_paths: list[Path], transcript_paths: list[Path] | None = None):
    app = build_graph(FakeStructuredLLM(), Config())
    config = {"configurable": {"thread_id": f"multi-{tmp_path.name}"}}
    return app.invoke(
        {
            "cv_paths": [str(p) for p in cv_paths],
            "transcript_paths": [str(p) for p in (transcript_paths or [])],
            "run_id": "multi",
            "output_dir": str(tmp_path / "out"),
            "interactive": False,
            "review_rounds": 0,
        },
        config=config,
    )


def test_pages_are_concatenated_in_argument_order(tmp_path):
    pages = write_pages(tmp_path)
    state = run_multi(tmp_path, pages)
    text = state["cv_doc"]["text"]

    assert "Sara Al-Balushi" in text and "SKILLS" in text
    assert text.index("Sara Al-Balushi") < text.index("EDUCATION") < text.index("SKILLS")


def test_reversed_arguments_reverse_the_document(tmp_path):
    """Order is the caller's responsibility and must be honoured, not sorted —
    filenames are not reliably page-ordered."""
    pages = write_pages(tmp_path)
    state = run_multi(tmp_path, list(reversed(pages)))
    text = state["cv_doc"]["text"]
    assert text.index("SKILLS") < text.index("Sara Al-Balushi")


def test_every_file_appears_in_the_envelope(tmp_path):
    pages = write_pages(tmp_path)
    state = run_multi(tmp_path, pages)
    profile = json.loads(Path(state["final_output_path"]).read_text(encoding="utf-8"))

    cv_docs = [d for d in profile["source_documents"] if d["role"] == "cv"]
    assert len(cv_docs) == 3, "each submitted file should be individually visible"
    assert [Path(d["path"]).name for d in cv_docs] == [p.name for p in pages]


def test_grounding_spans_all_pages(tmp_path):
    """A field extracted from page 2 must ground against page 2 — if only the
    first file were searched, most of the CV would be dropped as unverifiable."""
    pages = write_pages(tmp_path)
    state = run_multi(tmp_path, pages)
    report = state["grounding_report"]

    # The fake extracts a name from page 1 and a degree/GPA from page 2.
    assert report["full_name"]["grounded"] is True
    assert report["education[0].institution"]["grounded"] is True
    assert report["education[0].gpa"]["grounded"] is True


def test_one_unreadable_page_does_not_lose_the_document(tmp_path):
    pages = write_pages(tmp_path)
    broken = tmp_path / "not_a_document.xyz"
    broken.write_bytes(b"\x00\x01\x02 binary junk")

    state = run_multi(tmp_path, pages + [broken])
    assert not state.get("errors"), state.get("errors")
    assert "Sara Al-Balushi" in state["cv_doc"]["text"]
    assert any("Skipping" in w for w in state.get("warnings", []))


def test_all_pages_unreadable_is_fatal_for_the_cv(tmp_path):
    broken = tmp_path / "junk.xyz"
    broken.write_bytes(b"\x00\x01\x02")
    state = run_multi(tmp_path, [broken])
    assert state.get("errors")


def test_transcript_accepts_multiple_files(tmp_path):
    pages = write_pages(tmp_path)
    t1 = tmp_path / "t1.txt"
    t1.write_text("TRANSCRIPT\nStudent: Sara Al-Balushi\nCGPA 3.71\n", encoding="utf-8")
    t2 = tmp_path / "t2.txt"
    t2.write_text("COMP3202 Machine Learning A-\n", encoding="utf-8")

    state = run_multi(tmp_path, pages, [t1, t2])
    assert len(state["transcript_doc"]["parts"]) == 2
    assert "Machine Learning" in state["transcript_doc"]["text"]

    profile = json.loads(Path(state["final_output_path"]).read_text(encoding="utf-8"))
    assert len([d for d in profile["source_documents"] if d["role"] == "transcript"]) == 2


def test_confidence_is_block_weighted_not_a_flat_average():
    """A one-line photo must not drag down a full page's score equally."""
    assembled = _assemble(
        [
            {"text": "a", "pages": 1, "mean_confidence": 0.99, "blocks": [{}] * 100},
            {"text": "b", "pages": 1, "mean_confidence": 0.50, "blocks": [{}] * 2},
        ]
    )
    assert 0.97 < assembled["mean_confidence"] < 0.99  # not the flat mean of 0.745
    assert assembled["pages"] == 2


def test_mixed_part_types_report_the_weakest_link():
    """A document with any OCR'd part is OCR-derived for reporting purposes."""
    assert _dominant_kind([{"kind": "pdf_text"}, {"kind": "image"}]) == "image"
    assert _dominant_kind([{"kind": "pdf_text"}, {"kind": "text"}]) == "pdf_text"
