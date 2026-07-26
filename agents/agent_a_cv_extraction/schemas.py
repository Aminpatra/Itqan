"""Agent-A-internal output schemas.

Two conventions here are load-bearing and worth not "tidying up" later:

1. Optional fields are declared ``Optional[X] = None``. A null means "this was
   not present in the document" — that is the contract the extraction prompt is
   built around. Never give these a non-null default; a default value is
   indistinguishable from a hallucination once it reaches the grounding pass.

2. ``SkillClaim.source_span`` forces the model to point at real text. The Python
   grounder then checks that span actually exists in the source. A model that
   invents a skill usually invents a span to go with it, and the span is far
   easier to falsify than the skill.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ContactInfo(BaseModel):
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None


class EducationEntry(BaseModel):
    institution: Optional[str] = None
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    gpa: Optional[str] = None


class ExperienceEntry(BaseModel):
    company: Optional[str] = None
    title: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    responsibilities: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)


class ProjectEntry(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    technologies: list[str] = Field(default_factory=list)
    link: Optional[str] = None


class CertificationEntry(BaseModel):
    name: Optional[str] = None
    issuer: Optional[str] = None
    date: Optional[str] = None


class CourseGrade(BaseModel):
    code: Optional[str] = None
    title: Optional[str] = None
    grade: Optional[str] = None
    credits: Optional[str] = None
    term: Optional[str] = None


class SkillClaim(BaseModel):
    name: str
    source_span: Optional[str] = Field(
        default=None,
        description="A short VERBATIM substring of the source text where this skill appears.",
    )
    category: Optional[
        Literal["technical", "domain", "tool", "language", "soft", "unknown"]
    ] = None


class CVExtraction(BaseModel):
    full_name: Optional[str] = None
    contact: ContactInfo = Field(default_factory=ContactInfo)
    skills: list[SkillClaim] = Field(default_factory=list)
    education: list[EducationEntry] = Field(default_factory=list)
    experience: list[ExperienceEntry] = Field(default_factory=list)
    projects: list[ProjectEntry] = Field(default_factory=list)
    certifications: list[CertificationEntry] = Field(default_factory=list)
    courses: list[CourseGrade] = Field(default_factory=list)
    summary_statement: Optional[str] = None


class TranscriptExtraction(BaseModel):
    student_name: Optional[str] = None
    institution: Optional[str] = None
    program: Optional[str] = None
    cgpa: Optional[str] = None
    courses: list[CourseGrade] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------
class FieldVerdict(BaseModel):
    field_path: str
    value: Optional[str] = None
    grounded: bool
    evidence_quote: Optional[str] = Field(
        default=None,
        description="Exact contiguous substring of the source that supports the value.",
    )
    reason: Optional[str] = None


class GroundingReport(BaseModel):
    verdicts: list[FieldVerdict] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Curriculum research
# --------------------------------------------------------------------------
class CredentialCurriculum(BaseModel):
    """What a named course or certificate typically teaches.

    A statement about a syllabus, never about the candidate. ``recognized=False``
    with empty lists is a valid and expected answer for local events and
    anything the model does not genuinely know.
    """

    credential_name: str
    credential_kind: Literal["course", "certification"]
    recognized: bool
    typical_skills: list[str] = Field(default_factory=list)
    typical_concepts: list[str] = Field(default_factory=list)
    # The concrete, named skills this credential centrally teaches, listed in FULL
    # and independent of what the candidate claimed — the raw material for
    # promoting skills the candidate gained but never wrote on their CV. Unlike
    # typical_skills (which is steered toward the claimed set for corroboration),
    # this is the credential's own core syllabus. See nodes/derive_skills.py.
    key_skills_taught: list[str] = Field(default_factory=list)
    # The decisive field. Which of the skills the candidate ACTUALLY CLAIMED does
    # this credential normally teach? Asking directly beats emitting a generic
    # syllabus and hoping its vocabulary happens to match the candidate's wording
    # — "Data ingestion tools" and "Apache Sqoop" are the same thing and share no
    # words. Entries are validated against the claimed list, so this cannot
    # introduce a skill.
    covers_claimed_skills: list[str] = Field(default_factory=list)


class CurriculumResearch(BaseModel):
    credentials: list[CredentialCurriculum] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Skill judging
# --------------------------------------------------------------------------
class SkillVerdict(BaseModel):
    name: str
    keep: bool
    quality: Literal["high", "medium", "low"]
    category: Literal["technical", "domain", "tool", "language", "soft", "generic"]
    evidence_type: Literal[
        "project", "experience", "course", "certification", "adjacent", "claim_only"
    ]
    evidence_quote: Optional[str] = None
    # Set when the corroboration came from a credential's typical curriculum
    # rather than from text in the CV, so a consumer can tell "studied this" from
    # "the CV says they did this".
    corroborating_credential: Optional[str] = None
    rationale: str


class SkillJudgement(BaseModel):
    verdicts: list[SkillVerdict] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Human input validation
# --------------------------------------------------------------------------
class ValidatedField(BaseModel):
    field_path: str
    normalized_value: Optional[str] = None
    accepted: bool
    issue: Optional[str] = None


class HumanInputValidation(BaseModel):
    fields: list[ValidatedField] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Final summary (few-shot output)
# --------------------------------------------------------------------------
class CandidateSummary(BaseModel):
    headline: str
    profile: str
    core_strengths: list[str] = Field(default_factory=list)
    education_summary: str
    experience_summary: str
    academic_performance: Optional[str] = None
    gaps_or_unknowns: list[str] = Field(default_factory=list)
