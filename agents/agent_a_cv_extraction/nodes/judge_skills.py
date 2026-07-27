"""Skill quality judging.

The evidence context assembled here is what makes the cross-check possible: the
model sees projects, responsibilities, courses and certifications alongside the
claims, so "Python" backed by a PyTorch project can be distinguished from
"Python" appearing only in a skills list.
"""

from __future__ import annotations

import json
from typing import Any

from shared.config import Config
from shared.llm import as_dict, structured

from shared.grounding import compact_nulls, verify_quote
from ..prompts import SKILL_JUDGE_PROMPT
from ..schemas import SkillJudgement
from ..state import AgentState


def build_evidence_context(
    extraction: dict[str, Any],
    transcript: dict[str, Any] | None,
    curriculum: list[dict[str, Any]] | None = None,
) -> str:
    """Everything that could corroborate a skill claim, as readable text."""
    extraction = compact_nulls(extraction)   # pruning leaves nulls inside lists
    transcript = compact_nulls(transcript or {})
    lines: list[str] = []

    for project in extraction.get("projects") or []:
        if not isinstance(project, dict):
            continue
        tech = ", ".join(project.get("technologies") or [])
        lines.append(
            f"PROJECT: {project.get('name') or 'untitled'} — "
            f"{project.get('description') or ''} [tech: {tech or 'none listed'}]"
        )

    for role in extraction.get("experience") or []:
        if not isinstance(role, dict):
            continue
        lines.append(f"EXPERIENCE: {role.get('title') or '?'} at {role.get('company') or '?'}")
        for item in role.get("responsibilities") or []:
            lines.append(f"  - {item}")
        if role.get("technologies"):
            lines.append(f"  [tech: {', '.join(role['technologies'])}]")

    for cert in extraction.get("certifications") or []:
        if isinstance(cert, dict):
            lines.append(f"CERTIFICATION: {cert.get('name') or '?'} ({cert.get('issuer') or '?'})")

    courses = list(extraction.get("courses") or [])
    if transcript:
        courses.extend(transcript.get("courses") or [])
    for course in courses:
        if isinstance(course, dict) and course.get("title"):
            grade = f" (grade {course['grade']})" if course.get("grade") else ""
            lines.append(f"COURSE: {course['title']}{grade}")

    # Curriculum expansions go last and are clearly fenced: they are background
    # about a syllabus, not text from the candidate's documents, and the judge is
    # told to weight them accordingly.
    for entry in curriculum or []:
        if not isinstance(entry, dict):
            continue
        label = entry.get("credential_kind", "credential").upper()
        grade = entry.get("grade_achieved")
        header = f"CURRICULUM OF {label} \"{entry.get('credential_name')}\""
        if grade:
            header += f" - completed with grade {grade}"
        lines.append(header + " (typical syllabus, not stated in the CV):")

        # Stated first and explicitly: this is the finding the judge acts on.
        # Leaving it to infer a match between "Data ingestion tools" and a
        # claimed "Apache Sqoop" is what made corroboration unreliable before.
        covered = entry.get("covers_claimed_skills") or []
        if covered:
            lines.append(
                "  CORROBORATES THESE CLAIMED SKILLS: " + ", ".join(covered)
            )
        if skills := ", ".join(entry.get("typical_skills") or []):
            lines.append(f"  usually teaches: {skills}")
        if concepts := ", ".join(entry.get("typical_concepts") or []):
            lines.append(f"  concepts: {concepts}")

    return "\n".join(lines) if lines else "(no corroborating evidence found in the documents)"


# quality ceilings by how the skill was corroborated. The prompt states these
# rules; this enforces them, because a model under a long prompt drifts. On one
# run every skill came back "medium" — including claim_only ones, which the
# prompt explicitly caps at "low" — flattening the distinction the rating exists
# to express.
_MAX_QUALITY = {
    "project": "high",
    "experience": "high",
    "certification": "high",
    "course": "medium",
    # A specific tool inferred to be standard for a domain the candidate has
    # DIRECT evidence in: credible, so above a bare claim, but never "high" —
    # they did not demonstrate it, they are presumed to use it. Caps at medium.
    "adjacent": "medium",
    "claim_only": "low",
}
_RANK = {"low": 0, "medium": 1, "high": 2}


def _clean_evidence_quote(verdict: dict[str, Any], source_text: str) -> dict[str, Any]:
    """Drop an ``evidence_quote`` that is not actually a span of the documents.

    The contract calls this field an exact substring of the source, and a consumer
    is entitled to read it that way. In practice the model returned its own
    reasoning there — real output carried "used in a named project" and, via the
    derived-skills path, the prompt's own rubric line "taught in a completed
    course, or in a certification's curriculum". Those are descriptions of *why*,
    not evidence, and publishing them in an evidence field is a small lie that
    costs a consumer their ability to trust any of the field.

    Nulled rather than corrected: the rating and rationale still stand on their
    own, and "no quote" is honest where a paraphrase is not.
    """
    quote = verdict.get("evidence_quote")
    if quote and not verify_quote(quote, source_text):
        verdict = dict(verdict)
        verdict["evidence_quote"] = None
    return verdict


def _enforce_quality_rules(verdict: dict[str, Any]) -> dict[str, Any]:
    """Clamp quality to what the evidence actually supports.

    Downgrades only — this never invents confidence the model did not assert.
    """
    verdict = dict(verdict)
    evidence = verdict.get("evidence_type", "claim_only")
    quality = verdict.get("quality", "low")
    ceiling = _MAX_QUALITY.get(evidence, "low")

    if _RANK.get(quality, 0) > _RANK[ceiling]:
        verdict["quality"] = ceiling
        note = verdict.get("rationale") or ""
        verdict["rationale"] = (
            f"{note} [rating capped at {ceiling}: evidence is {evidence}]".strip()
        )
    return verdict


def make_judge_skills_node(llm: Any, config: Config):
    chain = SKILL_JUDGE_PROMPT | structured(llm, SkillJudgement)

    def judge_skills(state: AgentState) -> dict[str, Any]:
        extraction = state.get("cv_extraction") or {}
        claims = [s for s in (extraction.get("skills") or []) if isinstance(s, dict) and s.get("name")]

        if not claims:
            return {
                "accepted_skills": [],
                "rejected_skills": [],
                "warnings": ["No skills were available to judge."],
                "trace": ["judge_skills(none)"],
            }

        try:
            judgement = as_dict(
                chain.invoke(
                    {
                        "skills_json": json.dumps(
                            [{"name": c["name"], "category": c.get("category")} for c in claims],
                            indent=2,
                            ensure_ascii=False,
                        ),
                        "evidence_context": build_evidence_context(
                            extraction,
                            state.get("transcript_extraction"),
                            state.get("curriculum"),
                        ),
                    }
                )
            )
        except Exception as exc:
            # Fail open: an unjudged skill list is still useful downstream, and it
            # is labelled as unjudged so the consumer knows not to trust the filter.
            return {
                "accepted_skills": [
                    {
                        "name": c["name"],
                        "keep": True,
                        "quality": "low",
                        "category": c.get("category") or "unknown",
                        "evidence_type": "claim_only",
                        "origin": "unjudged_claim",
                        "rationale": "not judged — skill assessment failed",
                    }
                    for c in claims
                ],
                "rejected_skills": [],
                "warnings": [f"Skill judging failed ({exc}); passing all skills through unjudged."],
                "trace": ["judge_skills(failed)"],
            }

        cv_doc = state.get("cv_doc") or {}
        transcript_doc = state.get("transcript_doc") or {}
        source_text = "\n\n".join(
            t for t in (cv_doc.get("text", ""), transcript_doc.get("text", "")) if t
        )

        claimed_names = {c["name"].casefold(): c["name"] for c in claims}
        verdicts: dict[str, dict[str, Any]] = {}

        for verdict in judgement.get("verdicts", []):
            key = verdict.get("name", "").casefold()
            # The prompt forbids inventing skills; enforce it rather than trust it.
            if key not in claimed_names or key in verdicts:
                continue
            verdict = _clean_evidence_quote(verdict, source_text)
            verdict = _enforce_quality_rules(verdict)
            verdict["origin"] = "cv_claim"
            verdicts[key] = verdict

        # A skill the model simply failed to return must not vanish. Dropping it
        # silently loses a real claim and makes the count depend on model whim —
        # on one run this quietly discarded four of twenty skills.
        unjudged = [name for key, name in claimed_names.items() if key not in verdicts]
        for name in unjudged:
            verdicts[name.casefold()] = {
                "name": name,
                "keep": True,
                "quality": "low",
                "category": "unknown",
                "evidence_type": "claim_only",
                # Machine-readable, so a consumer can weight "the judge ruled on
                # this" differently from "the judge never saw it" without having to
                # string-match the English rationale below.
                "origin": "unjudged_claim",
                "rationale": "not returned by the skill judge; recorded as an unverified claim",
            }

        accepted = [v for v in verdicts.values() if v.get("keep")]
        rejected = [v for v in verdicts.values() if not v.get("keep")]

        updates: dict[str, Any] = {
            "accepted_skills": accepted,
            "rejected_skills": rejected,
            "trace": [f"judge_skills({len(accepted)} kept / {len(rejected)} filtered)"],
        }
        if unjudged:
            updates["warnings"] = [
                f"{len(unjudged)} skill(s) were not judged and default to unverified "
                f"claims: {', '.join(sorted(unjudged)[:6])}"
            ]
        return updates

    return judge_skills
