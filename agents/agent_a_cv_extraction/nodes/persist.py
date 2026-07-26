"""Build and write the final envelope.

The envelope is validated against ``shared.contracts.CandidateProfile`` before it
is written, so a malformed profile fails here — at the producer — rather than
confusing a downstream agent that has no way to tell whether the shape it got is
a bug or a schema change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from shared.artifacts import run_dir, utc_now, write_json
from shared.config import Config
from shared.contracts import (
    CandidateProfile,
    Confidence,
    CredentialCurriculum,
    FieldProvenance,
    Provenance,
    SourceDocument,
)

from shared.grounding import compact_nulls
from ..state import AgentState


def _section_confidence(report: dict[str, Any], prefix: str) -> float | None:
    scores = [
        entry.get("score", 0.0)
        for path, entry in report.items()
        if path.startswith(prefix)
    ]
    return round(sum(scores) / len(scores), 4) if scores else None


def make_persist_node(config: Config):
    def persist(state: AgentState) -> dict[str, Any]:
        # Sweep the nulls that pruning left inside lists. This is the boundary
        # where internal representation becomes the published contract, so it is
        # the right and only place to do it — the grounding report above still
        # needs the original indices to line up.
        extraction = compact_nulls(dict(state.get("cv_extraction") or {}))
        transcript = compact_nulls(state.get("transcript_extraction") or {})
        report = state.get("grounding_report") or {}

        # The transcript is the authority on coursework; merge without duplicating.
        courses = list(extraction.get("courses") or [])
        seen = {
            (c.get("code"), c.get("title"))
            for c in courses
            if isinstance(c, dict)
        }
        for course in transcript.get("courses") or []:
            if isinstance(course, dict) and (course.get("code"), course.get("title")) not in seen:
                courses.append(course)

        candidate = {
            "full_name": extraction.get("full_name") or transcript.get("student_name"),
            "contact": extraction.get("contact") or {},
            "education": extraction.get("education") or [],
            "experience": extraction.get("experience") or [],
            "projects": extraction.get("projects") or [],
            "certifications": extraction.get("certifications") or [],
            "courses": courses,
            "academic_record": {
                "institution": transcript.get("institution"),
                "program": transcript.get("program"),
                "cgpa": transcript.get("cgpa"),
            }
            if transcript
            else None,
        }

        # One entry per input file, not per document: a CV submitted as three
        # photos should show all three, each with its own type and confidence, so
        # a consumer can tell which page was the badly-lit one.
        documents: list[SourceDocument] = []
        for doc in (state.get("cv_doc"), state.get("transcript_doc")):
            if not doc:
                continue
            role = doc.get("role", "cv")
            for part in doc.get("parts") or []:
                kind = part.get("kind")
                documents.append(
                    SourceDocument(
                        path=part.get("path", ""),
                        kind=kind if kind in {"pdf_text", "pdf_scanned", "image", "text"} else "text",
                        role=role,
                        ocr_json_path=doc.get("ocr_json_path"),
                        mean_confidence=part.get("mean_confidence"),
                        pages=part.get("pages", 0),
                    )
                )

        grounding = {
            path: FieldProvenance(
                grounded=bool(entry.get("grounded")),
                score=float(entry.get("score", 0.0)),
                method=entry.get("method", "fuzzy"),
                evidence_quote=entry.get("evidence_quote"),
            )
            for path, entry in report.items()
        }

        # Human-supplied values never go through grounding — they have no reason to
        # appear in the OCR text. Record them explicitly so every field in the
        # candidate block has provenance, rather than silently having none.
        human_supplied = list(dict.fromkeys(state.get("human_supplied_fields") or []))
        for path in human_supplied:
            grounding[path] = FieldProvenance(grounded=True, score=1.0, method="human")

        grounded_scores = [g.score for g in grounding.values() if g.grounded]
        overall = round(sum(grounded_scores) / len(grounded_scores), 4) if grounded_scores else 0.0
        # An extraction nobody could verify should not report high confidence just
        # because the few fields that survived scored well.
        if grounding:
            overall = round(overall * (len(grounded_scores) / len(grounding)), 4)

        profile = CandidateProfile(
            run_id=state["run_id"],
            generated_at=utc_now(),
            source_documents=documents,
            candidate=candidate,
            skills={
                "accepted": state.get("accepted_skills") or [],
                "rejected": state.get("rejected_skills") or [],
            },
            summary=state.get("summary") or {},
            provenance=Provenance(
                grounding=grounding,
                human_supplied_fields=human_supplied,
                dropped_fields=list(dict.fromkeys(state.get("dropped_fields") or [])),
                review_rounds=state.get("review_rounds", 0),
                curriculum_researched=[
                    CredentialCurriculum(
                        credential_name=entry.get("credential_name", ""),
                        credential_kind=entry.get("credential_kind", "course"),
                        typical_skills=entry.get("typical_skills") or [],
                        typical_concepts=entry.get("typical_concepts") or [],
                        key_skills_taught=entry.get("key_skills_taught") or [],
                        covers_claimed_skills=entry.get("covers_claimed_skills") or [],
                        grade_achieved=entry.get("grade_achieved"),
                    )
                    for entry in (state.get("curriculum") or [])
                ],
            ),
            confidence=Confidence(
                overall=overall,
                per_section={
                    section: value
                    for section, prefix in (
                        ("contact", "contact"),
                        ("education", "education"),
                        ("experience", "experience"),
                        ("skills", "skills"),
                    )
                    if (value := _section_confidence(report, prefix)) is not None
                },
            ),
            warnings=list(state.get("warnings") or []),
            errors=list(state.get("errors") or []),
        )

        out_dir = run_dir(Path(state["output_dir"]), state["run_id"])
        path = write_json(out_dir / "candidate_profile.json", profile.model_dump())

        return {
            "final_output_path": path,
            "artifacts": [path],
            "trace": ["persist"],
        }

    return persist
