"""The inter-agent contract.

This is the ONLY thing a downstream agent should import from this project.
Agent B reads ``CandidateProfile``; it must never reach into
``agents.agent_a_cv_extraction.*``. That boundary is what lets Agent A be
rewritten — or have its OCR stack swapped out entirely — without touching
anything downstream.

``provenance`` is the part that makes this genuinely useful to another agent:
it can tell which facts came from OCR, which a human typed in, and which the
grounding pass could only partially verify. A consumer that treats a 0.41-
confidence OCR field the same as a human-confirmed one is making a mistake the
envelope gives it enough information to avoid.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

SCHEMA_VERSION = "itqan.candidate_profile/1.1"

GroundingMethod = Literal["exact", "fuzzy", "llm", "human", "dropped"]


class SourceDocument(BaseModel):
    path: str
    kind: Literal["pdf_text", "pdf_scanned", "image", "text"]
    role: Literal["cv", "transcript"]
    ocr_json_path: Optional[str] = None
    mean_confidence: Optional[float] = None
    pages: int = 0


class FieldProvenance(BaseModel):
    grounded: bool
    score: float = 0.0
    method: GroundingMethod = "fuzzy"
    evidence_quote: Optional[str] = None


class CredentialCurriculum(BaseModel):
    """What one of the candidate's credentials typically teaches.

    Model background knowledge, NOT a statement from the documents and NOT a
    claim about the candidate. Published so a consumer can see exactly which
    skill ratings rested on inferred curriculum rather than on CV text — and
    discount them if it disagrees with the inference.
    """

    credential_name: str
    credential_kind: Literal["course", "certification"]
    typical_skills: list[str] = Field(default_factory=list)
    typical_concepts: list[str] = Field(default_factory=list)


class Provenance(BaseModel):
    grounding: dict[str, FieldProvenance] = Field(default_factory=dict)
    human_supplied_fields: list[str] = Field(default_factory=list)
    dropped_fields: list[str] = Field(default_factory=list)
    review_rounds: int = 0
    # Credentials whose curriculum was used as corroborating evidence. Anything
    # the model did not recognise never appears here.
    curriculum_researched: list[CredentialCurriculum] = Field(default_factory=list)


class Confidence(BaseModel):
    overall: float = 0.0
    per_section: dict[str, float] = Field(default_factory=dict)


class CandidateProfile(BaseModel):
    """The envelope Agent A writes and Agent B consumes."""

    schema_version: str = SCHEMA_VERSION
    run_id: str
    generated_at: str

    source_documents: list[SourceDocument] = Field(default_factory=list)

    # Free-form on purpose: the candidate block mirrors Agent A's extraction
    # schema, and pinning it twice would mean editing two files for every field.
    candidate: dict[str, Any] = Field(default_factory=dict)

    skills: dict[str, list[dict[str, Any]]] = Field(
        default_factory=lambda: {"accepted": [], "rejected": []}
    )
    summary: dict[str, Any] = Field(default_factory=dict)

    provenance: Provenance = Field(default_factory=Provenance)
    confidence: Confidence = Field(default_factory=Confidence)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


def load_profile(path: str) -> CandidateProfile:
    """Read an envelope from disk. This is Agent B's entry point."""
    import json
    from pathlib import Path

    return CandidateProfile.model_validate(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )
