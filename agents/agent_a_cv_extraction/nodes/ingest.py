"""Ingest + text extraction.

The CV is required and an unreadable one is fatal. The transcript is optional, so
a failure there degrades to a warning and the run continues with CV data alone —
losing coursework is not a reason to lose the whole extraction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from shared.artifacts import run_dir, write_json
from shared.config import Config

from ..ingestion.detect import detect_kind
from ..ingestion.pdf_text import extract_text, rasterize
from ..state import AgentState, DocRecord


def make_ingest_node(config: Config):
    """Classify both documents from magic bytes."""

    def ingest(state: AgentState) -> dict[str, Any]:
        updates: dict[str, Any] = {"trace": ["ingest"]}
        errors: list[str] = []
        warnings: list[str] = []

        cv_path = Path(state["cv_path"])
        kind, detail = detect_kind(
            cv_path,
            min_chars=config.pdf_text_min_chars,
            min_chars_per_page=config.pdf_text_min_chars_per_page,
        )
        if kind == "unsupported":
            errors.append(f"CV is unreadable ({detail}). Cannot continue.")
        cv_doc: DocRecord = {"path": str(cv_path), "role": "cv", "kind": kind}
        updates["cv_doc"] = cv_doc

        transcript_path = state.get("transcript_path")
        if transcript_path:
            t_path = Path(transcript_path)
            t_kind, t_detail = detect_kind(
                t_path,
                min_chars=config.pdf_text_min_chars,
                min_chars_per_page=config.pdf_text_min_chars_per_page,
            )
            if t_kind == "unsupported":
                warnings.append(
                    f"Transcript is unreadable ({t_detail}); continuing with CV only."
                )
                updates["transcript_doc"] = None
            else:
                updates["transcript_doc"] = {
                    "path": str(t_path),
                    "role": "transcript",
                    "kind": t_kind,
                }
        else:
            updates["transcript_doc"] = None

        if errors:
            updates["errors"] = errors
        if warnings:
            updates["warnings"] = warnings
        return updates

    return ingest


def _read_document(doc: DocRecord, config: Config, out_dir: Path) -> DocRecord:
    """Turn one classified document into text, running OCR when needed."""
    path = Path(doc["path"])
    kind = doc["kind"]
    role = doc.get("role", "cv")

    if kind == "text":
        text = path.read_text(encoding="utf-8", errors="replace")
        return {**doc, "text": text, "pages": 1, "mean_confidence": None, "blocks": []}

    if kind == "pdf_text":
        text, pages = extract_text(path)
        return {**doc, "text": text, "pages": pages, "mean_confidence": None, "blocks": []}

    # Both remaining kinds end up in OCR; scanned PDFs get rasterized first.
    from ..ingestion.ocr import run_ocr

    if kind == "pdf_scanned":
        images = rasterize(path, out_dir / f"{role}_pages", dpi=config.pdf_raster_dpi)
    else:
        images = [path]

    result = run_ocr(images, lang=config.ocr_lang)
    ocr_json_path = write_json(
        out_dir / f"ocr_{role}.json",
        {
            "source_path": str(path),
            "kind": kind,
            "engine": result["engine"],
            "lang": result["lang"],
            "mean_confidence": result["mean_confidence"],
            "pages": result["pages"],
            "reconstructed_text": result["text"],
        },
    )
    return {
        **doc,
        "text": result["text"],
        "pages": len(result["pages"]),
        "mean_confidence": result["mean_confidence"],
        "blocks": result["blocks"],
        "ocr_json_path": ocr_json_path,
    }


def make_extract_text_node(config: Config):
    """Produce raw text for each document, persisting OCR output as JSON."""

    def extract_text_node(state: AgentState) -> dict[str, Any]:
        updates: dict[str, Any] = {"trace": ["extract_text"]}
        artifacts: list[str] = []
        warnings: list[str] = []
        out_dir = run_dir(Path(state["output_dir"]), state["run_id"])

        cv_doc = _read_document(state["cv_doc"], config, out_dir)
        updates["cv_doc"] = cv_doc
        if cv_doc.get("ocr_json_path"):
            artifacts.append(cv_doc["ocr_json_path"])
        if not cv_doc.get("text", "").strip():
            updates["errors"] = ["No text could be extracted from the CV."]

        transcript_doc = state.get("transcript_doc")
        if transcript_doc:
            transcript_doc = _read_document(transcript_doc, config, out_dir)
            updates["transcript_doc"] = transcript_doc
            if transcript_doc.get("ocr_json_path"):
                artifacts.append(transcript_doc["ocr_json_path"])
            if not transcript_doc.get("text", "").strip():
                warnings.append("No text could be extracted from the transcript.")

        for doc in (cv_doc, transcript_doc):
            if doc and doc.get("mean_confidence") is not None:
                if doc["mean_confidence"] < config.low_ocr_confidence:
                    warnings.append(
                        f"{doc['role']} OCR confidence is low "
                        f"({doc['mean_confidence']:.2f}); expect gaps to fill manually."
                    )

        if artifacts:
            updates["artifacts"] = artifacts
        if warnings:
            updates["warnings"] = warnings
        return updates

    return extract_text_node
