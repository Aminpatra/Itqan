"""Few-shot summarization.

The profile handed to the model is deliberately built from *verified* data only:
skills that survived judging, fields that survived grounding. Feeding it the raw
extraction would let a dropped hallucination reappear in the prose, having been
laundered through a stage that no longer checks it.

Low-confidence and human-supplied fields are labelled inline, because Example 3
in the few-shot set teaches the model what to do with those labels.
"""

from __future__ import annotations

from typing import Any

from shared.config import Config
from shared.llm import as_dict, structured

from ..grounding import compact_nulls
from ..prompts import SUMMARY_PROMPT
from ..schemas import CandidateSummary
from ..state import AgentState


def build_input_profile(state: AgentState) -> str:
    # Pruning leaves nulls inside lists (see grounding.compact_nulls); strip them
    # before formatting, or a joined field raises on the first null.
    extraction = compact_nulls(state.get("cv_extraction") or {})
    transcript = compact_nulls(state.get("transcript_extraction") or {})
    report = state.get("grounding_report") or {}
    human_fields = set(state.get("human_supplied_fields") or [])

    lines: list[str] = []

    name = extraction.get("full_name") or transcript.get("student_name")
    lines.append(f"name: {name or '(not found)'}")

    contact = extraction.get("contact") or {}
    contact_bits = [f"{k} {v}" for k, v in contact.items() if v]
    if contact_bits:
        marks = " [human-supplied]" if any(
            f"contact.{k}" in human_fields for k in contact if contact.get(k)
        ) else ""
        lines.append(f"contact: {', '.join(contact_bits)}{marks}")

    for entry in extraction.get("education") or []:
        if not isinstance(entry, dict):
            continue
        parts = [
            entry.get("degree"),
            entry.get("field_of_study"),
            entry.get("institution"),
            "-".join(x for x in [entry.get("start_date"), entry.get("end_date")] if x),
            f"GPA {entry['gpa']}" if entry.get("gpa") else None,
        ]
        lines.append("education: " + ", ".join(p for p in parts if p))

    if transcript:
        t_parts = [
            transcript.get("program"),
            transcript.get("institution"),
            f"CGPA {transcript['cgpa']}" if transcript.get("cgpa") else None,
        ]
        if any(t_parts):
            lines.append("transcript: " + ", ".join(p for p in t_parts if p))

    courses = list(extraction.get("courses") or []) + list(transcript.get("courses") or [])
    named: list[str] = []
    for course in courses:
        if not isinstance(course, dict) or not course.get("title"):
            continue
        grade = course.get("grade")
        named.append(f"{course['title']} ({grade})" if grade else course["title"])
    lines.append("courses: " + (", ".join(named) if named else "(none extracted)"))

    experience = extraction.get("experience") or []
    if experience:
        for role in experience:
            if not isinstance(role, dict):
                continue
            dates = "-".join(x for x in [role.get("start_date"), role.get("end_date")] if x)
            head = f"experience: {role.get('title') or '?'}, {role.get('company') or '?'}"
            if dates:
                head += f", {dates}"
            responsibilities = "; ".join(role.get("responsibilities") or [])
            lines.append(head + (f" — {responsibilities}" if responsibilities else ""))
    else:
        lines.append("experience: (none)")

    for project in extraction.get("projects") or []:
        if not isinstance(project, dict):
            continue
        tech = ", ".join(project.get("technologies") or [])
        lines.append(
            f"projects: \"{project.get('name') or 'untitled'}\" — "
            f"{project.get('description') or ''}"
            + (f", tech: {tech}" if tech else "")
        )

    accepted = state.get("accepted_skills") or []
    if accepted:
        lines.append(
            "accepted_skills: "
            + ", ".join(
                f"{s['name']} ({s.get('quality', '?')}/{s.get('evidence_type', '?')})"
                for s in accepted
            )
        )
    else:
        lines.append("accepted_skills: (none)")

    rejected = state.get("rejected_skills") or []
    if rejected:
        lines.append(
            "rejected_skills: "
            + ", ".join(f"{s['name']} ({s.get('category', 'generic')})" for s in rejected)
        )

    low_confidence = [
        path
        for path, entry in report.items()
        if entry.get("grounded") and entry.get("score", 1.0) < 0.95
    ]
    if low_confidence:
        lines.append(f"low_confidence_fields: {low_confidence[:6]}")

    dropped = state.get("dropped_fields") or []
    if dropped:
        lines.append(f"removed_unverifiable_fields: {sorted(set(dropped))[:6]}")

    return "\n".join(lines)


def make_summarize_node(llm: Any, config: Config):
    chain = SUMMARY_PROMPT | structured(llm, CandidateSummary)

    def summarize(state: AgentState) -> dict[str, Any]:
        profile = build_input_profile(state)
        try:
            summary = as_dict(chain.invoke({"input_profile": profile}))
        except Exception as exc:
            return {
                "summary": {},
                "warnings": [f"Summarization failed: {exc}"],
                "trace": ["summarize(failed)"],
            }
        return {"summary": summary, "trace": ["summarize"]}

    return summarize
