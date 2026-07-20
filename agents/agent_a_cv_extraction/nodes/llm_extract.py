"""LLM extraction nodes: raw text -> schema-constrained structured data."""

from __future__ import annotations

from typing import Any

from shared.config import Config
from shared.llm import as_dict, structured

from ..prompts import (
    CV_EXTRACTION_PROMPT,
    TRANSCRIPT_EXTRACTION_PROMPT,
    quality_note,
)
from ..schemas import CVExtraction, TranscriptExtraction
from ..state import AgentState


def make_llm_extract_cv_node(llm: Any, config: Config):
    chain = CV_EXTRACTION_PROMPT | structured(llm, CVExtraction)

    def llm_extract_cv(state: AgentState) -> dict[str, Any]:
        doc = state["cv_doc"]
        text = doc.get("text", "")
        if not text.strip():
            return {
                "cv_extraction": CVExtraction().model_dump(),
                "errors": ["CV had no text to extract from."],
                "trace": ["llm_extract_cv(empty)"],
            }

        try:
            result = chain.invoke(
                {
                    "source_text": text,
                    "ocr_quality_note": quality_note(doc.get("mean_confidence")),
                }
            )
        except Exception as exc:
            return {
                "cv_extraction": CVExtraction().model_dump(),
                "errors": [f"CV extraction failed: {exc}"],
                "trace": ["llm_extract_cv(failed)"],
            }

        return {"cv_extraction": as_dict(result), "trace": ["llm_extract_cv"]}

    return llm_extract_cv


def make_llm_extract_transcript_node(llm: Any, config: Config):
    chain = TRANSCRIPT_EXTRACTION_PROMPT | structured(llm, TranscriptExtraction)

    def llm_extract_transcript(state: AgentState) -> dict[str, Any]:
        doc = state.get("transcript_doc")
        if not doc or not doc.get("text", "").strip():
            return {"transcript_extraction": None, "trace": ["llm_extract_transcript(skip)"]}

        try:
            result = chain.invoke(
                {
                    "source_text": doc["text"],
                    "ocr_quality_note": quality_note(doc.get("mean_confidence")),
                }
            )
        except Exception as exc:
            # The transcript is optional, so a failure here is a warning, not an error.
            return {
                "transcript_extraction": None,
                "warnings": [f"Transcript extraction failed: {exc}"],
                "trace": ["llm_extract_transcript(failed)"],
            }

        return {
            "transcript_extraction": as_dict(result),
            "trace": ["llm_extract_transcript"],
        }

    return llm_extract_transcript


def route_after_cv(state: AgentState) -> str:
    """Only pay for a transcript extraction call when there is a transcript."""
    doc = state.get("transcript_doc")
    if doc and doc.get("text", "").strip():
        return "llm_extract_transcript"
    return "verify_grounding"
