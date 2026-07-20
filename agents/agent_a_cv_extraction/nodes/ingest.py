"""Ingest + text extraction.

A CV or transcript may arrive as several files — one photo per page is the normal
way people submit a scanned document from a phone. Each file is classified and
read independently, then concatenated in argument order into one logical
document. Mixed types within a document are fine: page one a PDF and page two a
photo is unusual but not wrong, and nothing here needs them to match.

The CV is required and an unreadable one is fatal. The transcript is optional, so
a failure there degrades to a warning and the run continues on CV data alone —
losing coursework is not a reason to lose the whole extraction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from shared.artifacts import run_dir, write_json
from shared.config import Config

from ..ingestion.detect import detect_kind
from ..ingestion.pdf_text import extract_text, rasterize
from ..state import AgentState, DocPart, DocRecord


def _dominant_kind(parts: list[DocPart]) -> str:
    """One label for a document assembled from possibly-mixed parts.

    Any part needing OCR makes the whole document OCR-derived for reporting
    purposes, because that is the weakest link in its text quality.
    """
    kinds = [p.get("kind") for p in parts]
    for candidate in ("image", "pdf_scanned"):
        if candidate in kinds:
            return candidate
    return kinds[0] if kinds else "unsupported"


def make_ingest_node(config: Config):
    """Classify every input file from magic bytes."""

    def _classify(paths: list[str], role: str) -> tuple[DocRecord, list[str]]:
        parts: list[DocPart] = []
        problems: list[str] = []

        for raw in paths:
            path = Path(raw)
            kind, detail = detect_kind(
                path,
                min_chars=config.pdf_text_min_chars,
                min_chars_per_page=config.pdf_text_min_chars_per_page,
            )
            if kind == "unsupported":
                problems.append(f"{path.name} ({detail})")
            parts.append({"path": str(path), "kind": kind})

        record: DocRecord = {"role": role, "parts": parts, "kind": _dominant_kind(parts)}
        return record, problems

    def ingest(state: AgentState) -> dict[str, Any]:
        updates: dict[str, Any] = {"trace": ["ingest"]}
        errors: list[str] = []
        warnings: list[str] = []

        cv_paths = state.get("cv_paths") or []
        cv_doc, cv_problems = _classify(cv_paths, "cv")
        updates["cv_doc"] = cv_doc

        # One unreadable page of several is recoverable; all of them is not.
        readable = [p for p in cv_doc["parts"] if p["kind"] != "unsupported"]
        if not readable:
            errors.append(
                "None of the CV file(s) could be read: " + "; ".join(cv_problems)
            )
        elif cv_problems:
            warnings.append(
                f"Skipping {len(cv_problems)} unreadable CV file(s): "
                + "; ".join(cv_problems)
            )

        transcript_paths = state.get("transcript_paths") or []
        if transcript_paths:
            t_doc, t_problems = _classify(transcript_paths, "transcript")
            t_readable = [p for p in t_doc["parts"] if p["kind"] != "unsupported"]
            if not t_readable:
                warnings.append(
                    "No transcript file could be read ("
                    + "; ".join(t_problems)
                    + "); continuing with the CV alone."
                )
                updates["transcript_doc"] = None
            else:
                if t_problems:
                    warnings.append(
                        f"Skipping {len(t_problems)} unreadable transcript file(s): "
                        + "; ".join(t_problems)
                    )
                updates["transcript_doc"] = t_doc
        else:
            updates["transcript_doc"] = None

        if errors:
            updates["errors"] = errors
        if warnings:
            updates["warnings"] = warnings
        return updates

    return ingest


def _read_part(part: DocPart, role: str, index: int, config: Config, out_dir: Path) -> DocPart:
    """Turn one classified file into text, running OCR when needed."""
    path = Path(part["path"])
    kind = part["kind"]

    if kind == "text":
        text = path.read_text(encoding="utf-8", errors="replace")
        return {**part, "text": text, "pages": 1, "mean_confidence": None, "blocks": []}

    if kind == "pdf_text":
        text, pages = extract_text(path)
        return {**part, "text": text, "pages": pages, "mean_confidence": None, "blocks": []}

    from ..ingestion.ocr import run_ocr

    if kind == "pdf_scanned":
        # Per-part subdirectory: two scanned PDFs in one document would otherwise
        # both write page_000.png and silently overwrite each other.
        images = rasterize(
            path, out_dir / f"{role}_part{index:02d}_pages", dpi=config.pdf_raster_dpi
        )
    else:
        images = [path]

    result = run_ocr(images, lang=config.ocr_lang)
    return {
        **part,
        "text": result["text"],
        "pages": len(result["pages"]),
        "mean_confidence": result["mean_confidence"],
        "blocks": result["blocks"],
        "_ocr_pages": result["pages"],
    }


def _assemble(parts: list[DocPart]) -> dict[str, Any]:
    """Concatenate read parts into the aggregates downstream nodes consume."""
    texts = [p.get("text", "") for p in parts if p.get("text", "").strip()]
    blocks = [b for p in parts for b in (p.get("blocks") or [])]

    # Block-weighted, so a one-line photo cannot drag down a full page's score.
    scored = [
        (p["mean_confidence"], len(p.get("blocks") or []))
        for p in parts
        if p.get("mean_confidence") is not None and p.get("blocks")
    ]
    total_blocks = sum(n for _, n in scored)
    mean_confidence = (
        round(sum(c * n for c, n in scored) / total_blocks, 4) if total_blocks else None
    )

    return {
        "text": "\n\n".join(texts).strip(),
        "blocks": blocks,
        "pages": sum(p.get("pages", 0) for p in parts),
        "mean_confidence": mean_confidence,
    }


def make_extract_text_node(config: Config):
    """Produce text for each document, persisting OCR output as JSON."""

    def _read_document(doc: DocRecord, out_dir: Path) -> tuple[DocRecord, str | None]:
        role = doc.get("role", "cv")
        read_parts: list[DocPart] = []
        ocr_pages: list[dict[str, Any]] = []

        for index, part in enumerate(doc.get("parts") or []):
            if part["kind"] == "unsupported":
                continue  # already reported at ingest
            read = _read_part(part, role, index, config, out_dir)
            for page in read.pop("_ocr_pages", []):
                page["source_part"] = part["path"]
                ocr_pages.append(page)
            read_parts.append(read)

        assembled = _assemble(read_parts)
        record: DocRecord = {**doc, "parts": read_parts, **assembled}

        ocr_json_path: str | None = None
        if ocr_pages:
            # One artifact per document, covering every page of every part, so a
            # consumer does not have to know how many files it arrived in.
            for i, page in enumerate(ocr_pages):
                page["page_index"] = i
            ocr_json_path = write_json(
                out_dir / f"ocr_{role}.json",
                {
                    "role": role,
                    "source_paths": [p["path"] for p in read_parts],
                    "engine": "PaddleOCR",
                    "lang": config.ocr_lang,
                    "mean_confidence": assembled["mean_confidence"],
                    "pages": ocr_pages,
                    "reconstructed_text": assembled["text"],
                },
            )
            record["ocr_json_path"] = ocr_json_path

        return record, ocr_json_path

    def extract_text_node(state: AgentState) -> dict[str, Any]:
        updates: dict[str, Any] = {"trace": ["extract_text"]}
        artifacts: list[str] = []
        warnings: list[str] = []
        out_dir = run_dir(Path(state["output_dir"]), state["run_id"])

        cv_doc, cv_artifact = _read_document(state["cv_doc"], out_dir)
        updates["cv_doc"] = cv_doc
        if cv_artifact:
            artifacts.append(cv_artifact)
        if not cv_doc.get("text", "").strip():
            updates["errors"] = ["No text could be extracted from the CV."]

        transcript_doc = state.get("transcript_doc")
        if transcript_doc:
            transcript_doc, t_artifact = _read_document(transcript_doc, out_dir)
            updates["transcript_doc"] = transcript_doc
            if t_artifact:
                artifacts.append(t_artifact)
            if not transcript_doc.get("text", "").strip():
                warnings.append("No text could be extracted from the transcript.")

        for doc in (cv_doc, transcript_doc):
            if not doc:
                continue
            # Flag the weakest page rather than the average: one badly-lit photo
            # in an otherwise clean set is exactly what produces silent errors.
            weak = [
                p
                for p in doc.get("parts") or []
                if p.get("mean_confidence") is not None
                and p["mean_confidence"] < config.low_ocr_confidence
            ]
            for part in weak:
                warnings.append(
                    f"{doc['role']}: {Path(part['path']).name} OCR confidence is low "
                    f"({part['mean_confidence']:.2f}); expect gaps to fill manually."
                )

        if artifacts:
            updates["artifacts"] = artifacts
        if warnings:
            updates["warnings"] = warnings
        return updates

    return extract_text_node
