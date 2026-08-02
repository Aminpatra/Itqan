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
    # For a skill: could the span the extractor cited be found verbatim? None when
    # no span was offered. Deliberately advisory — it reports how well the model
    # quoted, which is a fact about the model, not about the candidate, and must
    # never override the field's own grounding (doing so once deleted every skill
    # on a real CV).
    span_verified: Optional[bool] = None


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
    # The credential's full core syllabus (independent of what the candidate
    # claimed) — the source Agent A promotes coursework-derived skills from.
    key_skills_taught: list[str] = Field(default_factory=list)
    # Which of the candidate's claimed skills this credential was judged to teach,
    # and the grade they achieved if a transcript recorded one. This is the
    # inference a consumer most needs to see, because it is what lifted those
    # skills above an unevidenced claim.
    covers_claimed_skills: list[str] = Field(default_factory=list)
    grade_achieved: Optional[str] = None


class UnresolvedGap(BaseModel):
    """Something the producer knows is missing, stated as a fact rather than prose.

    The agent computes this deterministically (a required field absent, a field the
    OCR read too weakly to trust) and used to discard it — the structured list
    existed only in memory while the envelope carried nothing but the model's
    free-text `summary.gaps_or_unknowns`. Publishing it lets a consumer distinguish
    "the candidate has no LinkedIn" from "we could not read it", and lets an
    operator running --no-hitl see what a human would have been asked.
    """

    field_path: str
    reason: str
    ocr_confidence: Optional[float] = None


class Provenance(BaseModel):
    grounding: dict[str, FieldProvenance] = Field(default_factory=dict)
    human_supplied_fields: list[str] = Field(default_factory=list)
    dropped_fields: list[str] = Field(default_factory=list)
    # What is still missing at the end of the run — including gaps never put to a
    # human because the prompt was already full.
    unresolved_gaps: list[UnresolvedGap] = Field(default_factory=list)
    review_rounds: int = 0
    # Credentials whose curriculum was used as corroborating evidence. Anything
    # the model did not recognise never appears here.
    curriculum_researched: list[CredentialCurriculum] = Field(default_factory=list)


class Confidence(BaseModel):
    """How trustworthy is this profile — asked as two separate questions.

    ``overall`` describes the artifact that SHIPPED: the mean grounding score of
    the fields present in this envelope. ``extraction_precision`` describes the
    model that produced it: the share of proposed fields that survived
    verification. They were previously multiplied into a single number, which had
    the perverse property of falling when the anti-hallucination layer worked —
    a profile got *less* confident as its unsupported fields were removed.

    Both are None when nothing was measured, which is not the same as 0.0.
    """

    overall: Optional[float] = None
    per_section: dict[str, float] = Field(default_factory=dict)
    extraction_precision: Optional[float] = None
    # {verified, human, dropped} — the raw counts behind the ratios.
    fields: dict[str, int] = Field(default_factory=dict)


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
    # The employer as the posting names it; None when it names none. Persisted
    # from migration 0010 onward — rows written before it carry None.
    company: Optional[str] = None
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
    # The denominators. frequency_count is an absolute count that reads like a
    # rate: "58 postings ask for X" means nothing without "out of how many". And
    # since one roundup post can yield many postings, distinct_posts separates
    # broad demand from a single employer's hiring drive — measured on the live
    # corpus, one sector's 51 postings came from only 26 distinct posts.
    # Optional because rows written before these columns existed carry NULL.
    sector_volume: Optional[int] = None
    distinct_posts: Optional[int] = None


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
    # The platform the course is hosted on ('coursera', 'freecodecamp'), which is
    # NOT `provider` — that is the authoring partner, so a Coursera course names a
    # university. Consumers need the platform to say anything about how a course
    # is sold.
    source: Optional[str] = None
    url: str                                     # the course's `source_url` column
    taught_skills: list[str] = Field(default_factory=list)

    rating: Optional[float] = None
    review_count: Optional[int] = None
    enrollment_count: Optional[int] = None
    # beginner / intermediate / advanced as the provider labels it, or None when
    # it does not. Agent E prefers introductory courses for a MISSING skill — by
    # definition one the candidate has shown no evidence in.
    level: Optional[str] = None
    last_updated: Optional[str] = None           # ISO8601, or None
    price: Optional[CoursePrice] = None


# ---------------------------------------------------------------------------
# Agent C -> Agent E, and Agent E -> whoever reads it.
#
# These two envelopes went unmodelled for a long time while A->C had
# `CandidateProfile` from the start, and the asymmetry cost something concrete:
# Agent E read the gap file by `data.get("aggregate").get("missing_skill_details")`
# and, when that key was absent, fell back to the far lossier
# `most_common_missing_skills` with a warning nobody was watching. A renamed
# field on the producing side would therefore not fail — it would quietly
# produce worse recommendations.
#
# ADDITIVE-TOLERANT ON PURPOSE. Pydantic ignores unknown fields by default, so a
# producer adding a key never breaks a consumer; that is what makes it safe to
# introduce these models over envelopes already in the wild. Every field a
# consumer does not strictly require is Optional for the same reason. What these
# models catch is the opposite case: a field that VANISHES or changes type.
# ---------------------------------------------------------------------------

class MissingSkillDetail(BaseModel):
    """One gap, as Agent C measures it and Agent E consumes it."""

    skill: str
    esco_code: Optional[str] = None
    # Demand weight, counted once per concept and log1p-damped. An opaque
    # ordering weight to Agent E — deliberately not interpreted downstream.
    priority_score: float = 0.0

    # The market evidence proper. `jobs_missing_in` is a plain count of retrieved
    # postings that asked for this skill, which is the only claim about the
    # labour market either agent can actually support.
    jobs_missing_in: Optional[int] = None
    demand_rate: Optional[float] = None
    low_confidence: bool = False

    # How close the candidate came, for telling a near miss from a real gap.
    best_similarity: Optional[float] = None
    nearest_candidate_skill: Optional[str] = None
    also_phrased_as: list[str] = Field(default_factory=list)


class SkillGapAggregate(BaseModel):
    missing_skill_details: list[MissingSkillDetail] = Field(default_factory=list)
    most_common_missing_skills: list[str] = Field(default_factory=list)
    unresolved_skills: list[str] = Field(default_factory=list)
    # None when no job had parsable requirements — never 0.0, which on this
    # scale is the BEST possible value and would read as a perfect fit.
    average_gap_score: Optional[float] = None
    jobs_scored: int = 0
    jobs_without_parsable_requirements: int = 0


class SkillGap(BaseModel):
    """The envelope Agent C writes and Agent E consumes."""

    schema_version: str = "itqan.skill_gap/1.0"
    user_id: str = ""
    generated_at: Optional[str] = None
    # True when retrieval was thin enough that the SECTOR aggregate was also
    # computed. Since the delivered Agent C change it SUPPLEMENTS the per-job
    # results rather than replacing them, so it no longer means "these numbers
    # came from aggregate stats" — a consumer softening every claim on this flag
    # alone now overstates its case.
    used_fallback: bool = False
    aggregate: SkillGapAggregate = Field(default_factory=SkillGapAggregate)
    matched_jobs: list[dict[str, Any]] = Field(default_factory=list)
    fallback_sector_gap: Optional[dict[str, Any]] = None
    calibration: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


def load_gap(path: str) -> SkillGap:
    """Read a gap envelope from disk. This is Agent E's entry point, and the
    mirror of ``load_profile``."""
    import json
    from pathlib import Path

    return SkillGap.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
