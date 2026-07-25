"""The inter-agent contract.

``CandidateProfile`` is Agent A's output and **Agent C's** input — Agent C does
the matching, and is the only consumer that reads it. Agent B never touches it:
Agent B ingests job postings and its outputs are the ``job_postings`` and
``skill_demand_stats`` tables. Wiring Agent B to a candidate profile would be a
category error, and it is written here explicitly because the earlier version of
this docstring said "Agent B reads CandidateProfile" and would have led someone
straight into it.

No agent may reach into ``agents.agent_a_cv_extraction.*``. That boundary is
what lets Agent A be rewritten — or have its OCR stack swapped out entirely —
without touching anything downstream. Anything genuinely shared moves into
``shared/`` first; ``shared/grounding.py`` is the precedent.

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
    # Which of the candidate's claimed skills this credential was judged to teach,
    # and the grade they achieved if a transcript recorded one. This is the
    # inference a consumer most needs to see, because it is what lifted those
    # skills above an unevidenced claim.
    covers_claimed_skills: list[str] = Field(default_factory=list)
    grade_achieved: Optional[str] = None


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
    """The envelope Agent A writes and Agent C consumes."""

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
    """Read an envelope from disk. This is Agent C's entry point."""
    import json
    from pathlib import Path

    return CandidateProfile.model_validate(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


# ---------------------------------------------------------------------------
# Agent B -> Agent C: the job-market read contracts.
#
# Field lists mirror the job_postings / skill_demand_stats migrations. Fields
# DELIBERATELY absent from JobPostingExport, so their omission is a decision and
# not an accident:
#   embedding            — 1536 floats Agent C never needs back; it queries BY
#                          vector, it does not read vectors
#   status               — every exported row is 'active' by filter; exporting
#                          the column would invite consumers to re-filter wrongly
#   content_hash, duplicate_of, missed_cycles, stale_since, review_reason,
#   extraction_model     — Agent B's internal bookkeeping, not market data
# ---------------------------------------------------------------------------

class JobPostingExport(BaseModel):
    """One retrievable posting, as served by ``shared.job_market``."""

    posting_id: str
    source: str
    source_group: str
    source_type: str
    source_url: str

    title: str
    raw_description: str

    sector: Optional[str] = None            # ISCO-08 major group, '0'-'9'
    required_skills: list[str] = Field(default_factory=list)
    seniority_level: Optional[str] = None
    location: Optional[str] = None
    country: Optional[str] = None
    posted_date: Optional[str] = None       # ISO date; None when never stated

    legitimacy_score: Optional[float] = None
    listing_intent: str = "vacancy"         # by filter; carried for transparency
    poster_type: str = "company"

    first_seen_at: str
    last_seen_at: str

    # Computed per query: cosine similarity of this posting's essence embedding
    # to the caller's query vector. 1.0 identical, 0.0 unrelated.
    similarity: float


class SkillDemandStatRow(BaseModel):
    """One row of the aggregated demand table, latest window only.

    ``esco_code`` is the canonicalization handle: group by it to merge raw
    phrasings of one concept. NULL means the mapper found no concept — never a
    guess, so treat unmapped rows as their raw ``skill_key``.
    """

    sector: str
    skill: str
    skill_key: str
    esco_code: Optional[str] = None
    window_start: str
    window_end: str
    frequency_count: int
    prior_frequency_count: int = 0
    trend: str = "stable"
    co_occurring_skills: list[dict[str, Any]] = Field(default_factory=list)
    sample_postings: list[dict[str, Any]] = Field(default_factory=list)
    low_confidence: bool = False
    computed_at: str


# ---------------------------------------------------------------------------
# Agent D -> Agent E: the course price sub-shape.
#
# The one structured (non-scalar) course quality signal, published here so a
# future course consumer (Agent E) has a typed home for it. The rest of the
# quality signals (rating, review_count, enrollment_count, last_updated) are
# plain scalars and live directly on the course record / table.
# ---------------------------------------------------------------------------

class CoursePrice(BaseModel):
    """A course's price as reported by its provider at ingestion time.

    Stored as reported — NEVER normalized across providers, and never inferred.
    A provider that exposes no price at all yields ``price = None`` on the record
    (not this object with guessed values); this object is only built when the
    provider actually reports price. A free course is
    ``{amount: 0.0, currency: None, is_free: True}`` — amount is 0.0, never null.

    ``currency`` is Optional (deviating from a bare ``str``): a $0 course has no
    meaningful currency, and inventing one ("USD") would be exactly the kind of
    fabrication the rest of this system refuses. Cross-provider comparability is
    a query-time concern for the consumer, not an ingestion-time normalization.
    """

    amount: Optional[float] = None
    currency: Optional[str] = None
    is_free: bool = False


# ---------------------------------------------------------------------------
# Agent D -> Agent E: one retrievable course, as served by shared.course_market.
#
# The supply-side analog of JobPostingExport. Carries exactly the facts Agent E
# needs to pick a course and write a grounded rationale — the taught skills, the
# link, and the volatile quality/price signals — and nothing about Agent D's
# internal bookkeeping (content_hash, duplicate_of, embedding, staleness). Every
# quality field is Optional because "the provider did not report it" is a real,
# common answer that must stay null, never a guessed 0.
# ---------------------------------------------------------------------------

class CourseCandidate(BaseModel):
    """One course that teaches at least one requested skill.

    ``rating`` is stored on the provider's own native scale (NOT normalized) and
    is None when the provider publishes none. ``price`` is a full CoursePrice or
    None (never a fabricated placeholder). Which of the caller's skills this
    course covers is set-relative, so it is computed by Agent E per run, not
    carried on this record.
    """

    course_id: str
    title: str                                  # the course's `name` column
    provider: Optional[str] = None
    url: str                                     # the course's `source_url` column
    taught_skills: list[str] = Field(default_factory=list)

    rating: Optional[float] = None
    review_count: Optional[int] = None
    enrollment_count: Optional[int] = None
    last_updated: Optional[str] = None           # ISO8601, or None
    price: Optional[CoursePrice] = None
