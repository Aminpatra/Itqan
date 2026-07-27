"""Coursework-derived skills — the transcript's half of the picture.

A candidate who passed *Database Systems* but never wrote "SQL" on their CV still
has that skill, and the jobs it unlocks. Skill *judging* rules only on what the
CV claimed (by design — it must never invent a claim); this node does the
complementary job: it PROMOTES skills a completed credential normally teaches
into the candidate's set, clearly flagged and deliberately weighted below claimed
skills that a project or certificate backs.

It never invents from thin air. The raw material is the ``typical_skills`` that
``research_curriculum`` already produced for each recognised credential — a
statement about a real, completed course or certificate. Those candidates are run
back through the SAME skill judge, so the exact "reject generic / vague, keep
specific" logic applies; whatever survives is recorded as ``evidence_type=course``
(studied, capped at "medium" — never the "high" a project or held certification
earns) and marked ``origin="coursework_derived"`` so a consumer can weigh or
discount it. Enrichment, not a requirement: any failure leaves the claimed skills
untouched.
"""

from __future__ import annotations

import json
from typing import Any

from shared.config import Config
from shared.llm import as_dict, structured

from ..prompts import SKILL_JUDGE_PROMPT
from ..schemas import SkillJudgement
from ..state import AgentState
from .judge_skills import _RANK, _enforce_quality_rules, build_evidence_context


def _credential_rank(entry: dict[str, Any]) -> int:
    """Which credential to cite when several teach the same skill: a certification
    is the strongest, then a course completed to a recorded grade, then a plain
    course. Lower sorts first."""
    if entry.get("credential_kind") == "certification":
        return 0
    return 1 if entry.get("grade_achieved") else 2


def collect_derived_candidates(
    curriculum: list[dict[str, Any]], known: set[str]
) -> dict[str, dict[str, str]]:
    """Each unclaimed skill a credential centrally teaches, mapped to the
    strongest credential that teaches it. Keyed by casefolded name so the caller
    can dedup against skills already claimed, accepted, or rejected (``known``).

    Draws from ``key_skills_taught`` — the credential's full core syllabus,
    populated regardless of what the candidate claimed — and falls back to
    ``typical_skills`` for any entry that predates that field."""
    candidates: dict[str, dict[str, str]] = {}
    for entry in sorted(curriculum, key=_credential_rank):
        credential = entry.get("credential_name") or "coursework"
        taught = entry.get("key_skills_taught") or entry.get("typical_skills") or []
        for skill in taught:
            if not isinstance(skill, str) or not skill.strip():
                continue
            key = skill.strip().casefold()
            # First writer wins, and credentials are pre-sorted strongest-first,
            # so the citation is the best available and skills stay deduped.
            if key in known or key in candidates:
                continue
            candidates[key] = {"name": skill.strip(), "corroborating_credential": credential}
    return candidates


def make_derive_coursework_skills_node(llm: Any, config: Config):
    chain = SKILL_JUDGE_PROMPT | structured(llm, SkillJudgement)

    def derive_coursework_skills(state: AgentState) -> dict[str, Any]:
        if not config.derive_coursework_skills:
            return {"trace": ["derive_coursework_skills(disabled)"]}

        curriculum = state.get("curriculum") or []
        accepted = list(state.get("accepted_skills") or [])
        rejected = state.get("rejected_skills") or []
        extraction = state.get("cv_extraction") or {}

        # Everything already decided — do not re-propose it.
        known: set[str] = set()
        for skill in [*accepted, *rejected]:
            if isinstance(skill, dict) and skill.get("name"):
                known.add(str(skill["name"]).casefold())
        for skill in extraction.get("skills") or []:
            if isinstance(skill, dict) and skill.get("name"):
                known.add(str(skill["name"]).casefold())

        candidates = collect_derived_candidates(curriculum, known)
        if not candidates:
            return {"trace": ["derive_coursework_skills(none)"]}

        try:
            judgement = as_dict(
                chain.invoke(
                    {
                        "skills_json": json.dumps(
                            [{"name": c["name"]} for c in candidates.values()],
                            indent=2, ensure_ascii=False,
                        ),
                        "evidence_context": build_evidence_context(
                            extraction, state.get("transcript_extraction"), curriculum
                        ),
                    }
                )
            )
        except Exception as exc:
            # Best-effort: a failed derivation must not disturb the claimed skills.
            return {
                "warnings": [f"Coursework skill derivation failed ({exc}); no skills derived."],
                "trace": ["derive_coursework_skills(failed)"],
            }

        derived: list[dict[str, Any]] = []
        seen: set[str] = set()
        for verdict in judgement.get("verdicts", []):
            key = str(verdict.get("name", "")).casefold()
            cand = candidates.get(key)
            # Not one we asked about (the judge is told never to add), a duplicate,
            # or rejected as generic/vague — skip. That reuse is the whole point:
            # the generic-skill filter is the judge's, not re-implemented here.
            if cand is None or key in seen or not verdict.get("keep", False):
                continue
            seen.add(key)

            v = dict(verdict)
            v["name"] = cand["name"]                 # the credential's wording, not the judge's
            v["evidence_type"] = "course"            # studied, by construction
            v = _enforce_quality_rules(v)            # course -> caps quality at "medium"
            v["corroborating_credential"] = cand["corroborating_credential"]
            v["origin"] = "coursework_derived"
            # A derived skill has no source span by construction — it is precisely
            # the skill the CV never mentioned. Carrying the judge's text through
            # published the prompt's own rubric ("taught in a completed course, or
            # in a certification's curriculum") in a field the contract defines as a
            # verbatim quote from the documents. corroborating_credential above is
            # the honest provenance; this stays null.
            v["evidence_quote"] = None
            note = "[derived from completed coursework; not claimed in the CV]"
            v["rationale"] = f"{(v.get('rationale') or '').strip()} {note}".strip()
            derived.append(v)

        if not derived:
            return {"trace": ["derive_coursework_skills(0 kept)"]}

        # Strongest first, then alphabetical, before the cap — so a long transcript
        # keeps its best derivations rather than whatever the judge listed first.
        derived.sort(key=lambda v: (-_RANK.get(v.get("quality", "low"), 0), v["name"].casefold()))
        derived = derived[: config.max_coursework_derived_skills]

        return {
            "accepted_skills": accepted + derived,
            "warnings": [
                f"Added {len(derived)} skill(s) derived from completed coursework, not "
                f"claimed in the CV: {', '.join(d['name'] for d in derived[:6])}"
            ],
            "trace": [f"derive_coursework_skills({len(derived)} derived)"],
        }

    return derive_coursework_skills
