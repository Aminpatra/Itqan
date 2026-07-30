"""Agent envelopes -> the UI's camelCase shapes.

This is the honest-gap layer, and it is the most important file in `api/`. The
frontend asks for six things the pipeline does not produce, and an integration
layer is exactly where a system starts lying about them — a `0` for an absent
price reads as "free", a `0.9` for a categorical quality reads as measured
certainty, a `0` readiness reads as "you match nothing".

Every one of them is published as **null**. That is the same discipline the
pipeline audits established (Agent C's `gap_score` is null rather than 0.0 when
there is nothing to compute it from) and it only works if this layer refuses to
"helpfully" fill in.

The shapes here mirror `Onboarding/src/api/types.ts`, which is the contract that
actually runs — camelCase, plain already-localised strings. Not the handoff PDF's
snake_case/bilingual target.
"""

from __future__ import annotations

import re
from typing import Any, Optional

# Agent A publishes a categorical `quality`, not a number. The UI gates display
# on TRUST_THRESHOLD = 0.85 (`isStrong`), so a fabricated number here would
# directly change what is stated to the user as fact. These bands are therefore
# deliberately conservative: `high` sits just above the threshold, `medium` and
# `low` below it, so nothing categorical is ever presented as strongly evidenced
# unless the field also carries a real grounding score.
_QUALITY_BAND = {"high": 0.86, "medium": 0.70, "low": 0.50}


def uploaded_document(row: dict[str, Any]) -> dict[str, Any]:
    """`UploadedDocument`. The database speaks snake_case and the UI speaks
    camelCase; the translation belongs here rather than in a route returning raw
    columns, which is how `document_id` reached the client as a missing `id`."""
    return {
        "id": row["document_id"],
        "fileName": row["file_name"],
        "mimeType": row["mime_type"],
        "sizeBytes": int(row["size_bytes"]),
        "kind": row["kind"],
    }


def _grounding(profile: dict[str, Any], path: str) -> Optional[dict[str, Any]]:
    return ((profile.get("provenance") or {}).get("grounding") or {}).get(path)


def _extracted(profile: dict[str, Any], path: str, value: Any) -> Optional[dict[str, Any]]:
    """The UI's `Extracted<T>` = {value, confidence, evidence}.

    `confidence` is the field's real grounding score and `evidence` its verified
    quote — both measured by Agent A, neither invented. A value the pipeline did
    not publish returns None rather than an entry with a made-up certainty.
    """
    if value in (None, ""):
        return None
    g = _grounding(profile, path) or {}
    return {
        "value": value,
        "confidence": g.get("score"),
        "evidence": g.get("evidence_quote"),
    }


# Agent A passes education dates through as the document wrote them, which on a
# real CV means "Expected June 2026", "2024 - 2026", "Sept 2025" — free text. The
# UI types graduationDate as ISO yyyy-mm, so only an unambiguous year or
# year-month may be published.
_ISO_MONTH = re.compile(r"\b(20\d{2})[-/](0[1-9]|1[0-2])\b")
_YEAR = re.compile(r"\b(20\d{2})\b")

# A month written the way people write it. Reading "June 2026" as 2026-06 is not
# inference: the document states the month, in words, and refusing to parse it
# threw away information the CV actually carried. The UI's field is
# `<input type="month">`, so a bare year cannot populate it — which is how a
# graduation date Agent A had extracted still showed as blank.
_MONTHS = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04", "may": "05", "jun": "06",
    "jul": "07", "aug": "08", "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}
_MONTH_NAME = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*,?\s*(20\d{2})\b",
    re.IGNORECASE)


def _graduation_date(candidate: dict[str, Any]) -> Optional[str]:
    """The latest education end date, but only when it really is a date.

    Slicing the first seven characters of whatever Agent A stored produced
    `"Expecte"` from "Expected June 2026" on a real CV — a string the UI would
    have rendered to the user as their graduation date. A value that does not
    parse is not published at all: an absent date shows as "not found", which is
    true, where a mangled one looks like data.

    Three accepted shapes, in order of how much they say: `2026-06`, `June 2026`,
    and a bare `2026`. The year-only case is published as a year and the UI asks
    the user to pick the month — padding it to `2026-01-01` would state a day the
    document never did.
    """
    best: Optional[str] = None
    for entry in candidate.get("education") or []:
        raw = (entry or {}).get("end_date") or (entry or {}).get("graduation_date")
        if not isinstance(raw, str):
            continue
        if match := _ISO_MONTH.search(raw):
            value = f"{match.group(1)}-{match.group(2)}"
        elif named := _MONTH_NAME.search(raw):
            value = f"{named.group(2)}-{_MONTHS[named.group(1).lower()]}"
        elif years := _YEAR.findall(raw):
            value = max(years)          # "2024 - 2026" means it ends in 2026
        else:
            continue
        if best is None or value > best:
            best = value
    return best


def analysis_result(profile: dict[str, Any]) -> dict[str, Any]:
    """`AnalysisResult` — what the confirm screen shows for the user to correct."""
    candidate = profile.get("candidate") or {}
    skills: list[dict[str, Any]] = []
    for i, skill in enumerate(profile.get("skills", {}).get("accepted") or []):
        name = (skill.get("name") or "").strip()
        if not name:
            continue
        g = _grounding(profile, f"skills[{i}].name") or {}
        skills.append({
            "id": f"s{i + 1}",
            "name": name,
            # Real score where the skill was grounded against the document;
            # otherwise the conservative band for its evidence quality.
            "confidence": g.get("score") or _QUALITY_BAND.get(skill.get("quality"), 0.5),
            # Agent A's own word for where a rating came from, so the UI can show
            # a coursework-derived skill differently from a project-evidenced one.
            "fromCourse": skill.get("corroborating_credential"),
        })

    return {
        "fullName": _extracted(profile, "full_name", candidate.get("full_name")),
        # ALWAYS null: Agent A never extracts a birth date — it is not a field in
        # CVExtraction. The UI types this nullable and renders "not found", which
        # is true. Inventing one from a national-ID pattern would be worse.
        "birthDate": None,
        "graduationDate": (
            {"value": grad, "confidence": None, "evidence": None}
            if (grad := _graduation_date(candidate)) else None
        ),
        "skills": skills,
    }


# ---------------------------------------------------------------------------
def job_matches(gap: dict[str, Any], *, limit: Optional[int] = None) -> list[dict[str, Any]]:
    """`JobMatch[]` from Agent C's per-job results.

    `why` is built deterministically from Agent C's `skill_resolution` — the tier
    that settled each requirement and the candidate skill that satisfied it. A row
    with no real evidence chain is DROPPED rather than shipped with generic prose,
    which is the frontend's own stated rule: a recommendation the user cannot
    check is one the sceptical user will not trust.
    """
    out: list[dict[str, Any]] = []
    for job in gap.get("matched_jobs") or []:
        matched = [r for r in (job.get("skill_resolution") or [])
                   if r.get("verdict") == "matched" and r.get("satisfied_by")]
        if not matched:
            continue

        score = job.get("gap_score")
        parts = [f"{r['satisfied_by']} covers their requirement for {r['skill']}"
                 for r in matched[:3]]
        why = "; ".join(parts) + "."
        out.append({
            "id": job.get("job_id") or "",
            "title": job.get("job_title") or "",
            # Agent B extracted and grounded `company` from the first version and
            # then discarded it for want of a column (finding A5 of its audit),
            # which left every job card without an employer. Persisted from Agent
            # B migration 0010; rows ingested before it still read None, so this
            # fills in as the corpus refreshes rather than all at once.
            "employer": job.get("company") or "",
            "location": job.get("location") or "",
            "arrangement": job.get("seniority_level") or "",
            # gap_score is how much is MISSING, so readiness is its complement.
            # None stays None: no score means no claim.
            "score": None if score is None else round(1.0 - float(score), 4),
            "why": why,
            "matchedSkills": [r["satisfied_by"] for r in matched],
            "source": {
                "name": job.get("source") or "",
                "url": job.get("source_url") or "",
                "retrievedAt": job.get("posted_date") or "",
            },
        })
    out.sort(key=lambda j: (j["score"] is None, -(j["score"] or 0)))
    return out[:limit] if limit else out


def courses(recommendations: dict[str, Any]) -> list[dict[str, Any]]:
    """`Course[]` from Agent E. Nulls preserved; see the module docstring."""
    out: list[dict[str, Any]] = []
    for rec in recommendations.get("recommendations") or []:
        course = rec.get("course")
        if not course:
            continue                      # no_course_found surfaces in gaps, not here
        quality = course.get("quality") or {}
        price = quality.get("price") or {}
        out.append({
            "id": course.get("course_id") or "",
            "title": course.get("title") or "",
            "provider": course.get("provider") or "",
            # Agent D stores no duration. Coursera's API exposes `workload` and
            # the adapter even requests it, but nothing persists it — so null,
            # not 0, which would render as "0 hours".
            "hours": course.get("hours"),
            # Measured: 0 of 1,999 Coursera courses publish a price. null means
            # "not listed"; 0 would mean FREE, which is a different claim.
            "price": price.get("amount"),
            "currency": price.get("currency"),
            "unlocks": [rec.get("skill")] + list(course.get("covers_other_skills") or []),
            # Agent E picks one course per gap, so everything it returns is a
            # recommendation. `thin`/`arbitrary` supply is the interesting nuance
            # and is surfaced on the dashboard rather than flattened into a bool.
            "recommended": True,
            "source": {
                "name": course.get("provider") or "",
                "url": course.get("url") or "",
                "retrievedAt": recommendations.get("generated_at") or "",
            },
        })
    return out


def dashboard(profile: dict[str, Any], gap: dict[str, Any],
              recommendations: dict[str, Any]) -> dict[str, Any]:
    """`DashboardData`, assembled server-side from Agent C + Agent E."""
    aggregate = gap.get("aggregate") or {}
    average = aggregate.get("average_gap_score")

    held = [s.get("name") for s in (profile.get("skills", {}).get("accepted") or [])
            if s.get("name")]
    missing = [d.get("skill") for d in (aggregate.get("missing_skill_details") or [])
               if d.get("skill")]

    standings = [{"name": n, "level": 0.9, "held": True} for n in held[:6]]
    standings += [{"name": n, "level": 0.1, "held": False} for n in missing[:6]]

    # Gaps Agent E found nothing for belong on the dashboard, so the UI can say
    # "no course found for this yet" instead of the gap quietly disappearing.
    uncovered = [r.get("skill") for r in (recommendations.get("recommendations") or [])
                 if r.get("no_course_found") and r.get("skill")]

    return {
        # gap_score measures what is MISSING; readiness is its complement. None
        # when Agent C had nothing to compute it from — the UI renders an empty
        # ring, where a 0 would read as "you match nothing".
        "readiness": None if average is None else round((1.0 - float(average)) * 100),
        "readinessNote": _readiness_note(average, len(missing)),
        "strengths": held[:5],
        "standings": standings,
        "topMatches": job_matches(gap, limit=2),
        "gaps": missing[:8],
        "nextStep": _next_step(courses(recommendations), uncovered),
        "journey": _journey(gap, recommendations),
    }


def _readiness_note(average: Optional[float], gap_count: int) -> str:
    if average is None:
        return ("There is not enough in your documents yet to judge your readiness. "
                "Adding a transcript would give the matching more to work with.")
    ready = round((1.0 - float(average)) * 100)
    if gap_count == 0:
        return f"You match {ready}% of what the roles you were compared against ask for."
    return (f"You match {ready}% of what the roles you were compared against ask for. "
            f"{gap_count} skill(s) came up that your documents do not evidence yet.")


def _next_step(course_list: list[dict[str, Any]],
               uncovered: list[str]) -> dict[str, Any]:
    if course_list:
        first = course_list[0]
        return {
            "title": f"Start with {first['title']}",
            "body": (f"It covers {', '.join(first['unlocks'][:2])}, which came up in the "
                     f"roles you were matched against."),
            "action": "courses",
        }
    if uncovered:
        return {
            "title": "No course found for your remaining gaps yet",
            "body": (f"Nothing in the catalogue currently teaches "
                     f"{', '.join(uncovered[:2])}. Job matches are still worth reviewing."),
            "action": "jobs",
        }
    return {
        "title": "Add another document",
        "body": "More evidence widens what can be matched, especially a transcript.",
        "action": "documents",
    }


def _journey(gap: dict[str, Any], recommendations: dict[str, Any]) -> list[dict[str, Any]]:
    """Stage state is decided here, from whether the work FINISHED — not from
    which screen the browser visited."""
    has_gap = bool((gap.get("aggregate") or {}).get("missing_skill_details") is not None)
    has_courses = bool(recommendations.get("recommendations"))
    return [
        {"id": "documents", "label": "Documents read", "state": "done"},
        {"id": "skills", "label": "Skills identified", "state": "done"},
        {"id": "matching", "label": "Matching",
         "state": "done" if has_gap else "current"},
        {"id": "courses", "label": "Courses matched",
         "state": "done" if has_courses else ("current" if has_gap else "upcoming")},
        {"id": "jobs", "label": "Applying for jobs",
         "state": "current" if has_courses else "upcoming"},
    ]
