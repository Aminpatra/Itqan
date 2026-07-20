"""Gap detection — deterministic, no LLM.

A gap is a field we want, do not have, and could plausibly get from the user.
Detecting these with a model would be both slower and less predictable than a
handful of explicit rules, and the rules double as documentation of what the
agent considers a complete profile.
"""

from __future__ import annotations

import re
from typing import Any

from shared.config import Config

from ..grounding import normalize
from ..state import AgentState, Gap

# field_path -> (human label, hint shown at the prompt)
REQUIRED_FIELDS: dict[str, tuple[str, str]] = {
    "full_name": ("Full name", "as written on the CV"),
    "contact.email": ("Email address", "e.g. name@example.com"),
}

# Fields that are not required, but are worth one prompt if we could not read
# them. Anything outside this set is left alone: a CV can drop dozens of fields
# to grounding, and interrogating the user about each one is worse than useless.
WORTH_ASKING: dict[str, tuple[str, str]] = {
    "contact.phone": ("Phone number", "e.g. +968 9123 4567"),
    "contact.location": ("Location", "city and country"),
    "contact.linkedin": ("LinkedIn", "profile URL"),
    "education[0].degree": ("Degree", "exactly as written, e.g. BSc Computer Science"),
    "education[0].institution": ("Institution", "e.g. Sultan Qaboos University"),
    "education[0].gpa": ("GPA", "e.g. 3.71/4.0"),
}

# Ceiling on prompts per round. Past this the CLI stops being something a person
# will actually sit through, and unanswered gaps are recorded in the output anyway.
MAX_GAPS_PER_ROUND = 8


_INDEX = re.compile(r"^(.*?)\[(\d+)\]$")


def _get_path(data: dict[str, Any], path: str) -> Any:
    """Read ``contact.email`` or ``education[0].degree``. Missing -> None."""
    cursor: Any = data
    for part in path.split("."):
        match = _INDEX.match(part)
        if match:
            key, index = match.group(1), int(match.group(2))
            if not isinstance(cursor, dict):
                return None
            container = cursor.get(key)
            if not isinstance(container, list) or len(container) <= index:
                return None
            cursor = container[index]
        else:
            if not isinstance(cursor, dict):
                return None
            cursor = cursor.get(part)
    return cursor


def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _lowest_supporting_confidence(
    value: str, blocks: list[dict[str, Any]]
) -> float | None:
    """Confidence of the weakest OCR block that contributed to ``value``.

    A field assembled from a 0.4-confidence block is suspect even when it is
    perfectly grounded — grounding proves it matches the OCR text, not that the
    OCR read the page correctly. This is the check that catches that case.
    """
    if not blocks or not value:
        return None
    needle = normalize(value)
    if not needle:
        return None

    scores = [
        block["confidence"]
        for block in blocks
        if block.get("text")
        and (normalize(block["text"]) in needle or needle in normalize(block["text"]))
    ]
    return min(scores) if scores else None


def make_assess_gaps_node(config: Config):
    def assess_gaps(state: AgentState) -> dict[str, Any]:
        extraction = state.get("cv_extraction") or {}
        transcript = state.get("transcript_extraction")
        report = state.get("grounding_report") or {}
        blocks = (state.get("cv_doc") or {}).get("blocks") or []
        already_supplied = set(state.get("human_supplied_fields") or [])

        gaps: list[Gap] = []

        def add(path: str, reason: str, hint: str, current: Any = None, conf=None) -> None:
            if path in already_supplied:
                return  # the user already answered this; do not ask twice
            gaps.append(
                {
                    "field_path": path,
                    "reason": reason,
                    "prompt_hint": hint,
                    "current_value": current if isinstance(current, str) else None,
                    "ocr_confidence": conf,
                }
            )

        # 1. Required fields that are missing or were pruned as ungrounded.
        for path, (label, hint) in REQUIRED_FIELDS.items():
            value = _get_path(extraction, path)
            if _blank(value):
                entry = report.get(path) or {}
                reason = (
                    "could not be verified against the document and was removed"
                    if entry.get("method") == "dropped"
                    else "not found in the document"
                )
                add(path, f"{label} {reason}", hint)

        # 2. Useful-but-optional fields that are missing or failed verification.
        #    A field dropped as unverifiable is precisely what the manual-entry
        #    path exists for: the document gave us nothing usable, but the
        #    candidate can supply it.
        for path, (label, hint) in WORTH_ASKING.items():
            if not _blank(_get_path(extraction, path)):
                continue
            # Skip education sub-fields when there is no education section at
            # all — rule 3 asks for the institution once instead of three times.
            if path.startswith("education[") and not extraction.get("education"):
                continue
            dropped = (report.get(path) or {}).get("method") == "dropped"
            add(
                path,
                f"{label} "
                + (
                    "could not be verified against the document"
                    if dropped
                    else "was not found in the document"
                ),
                hint,
            )

        # 3. Structural minimums.
        if not extraction.get("education"):
            add(
                "education[0].institution",
                "No education entries were found",
                "institution name, e.g. Sultan Qaboos University",
            )
        if not extraction.get("skills"):
            add(
                "skills_freeform",
                "No skills were found",
                "comma-separated, e.g. Python, SQL, AutoCAD",
            )

        # 3. Fields resting on weak OCR, even where grounding passed.
        for path, entry in report.items():
            if not entry.get("grounded") or path in already_supplied:
                continue
            value = entry.get("value")
            if not isinstance(value, str):
                continue
            confidence = _lowest_supporting_confidence(value, blocks)
            if confidence is not None and confidence < config.low_ocr_confidence:
                add(
                    path,
                    f"OCR read this with low confidence ({confidence:.2f}) — please confirm",
                    "press Enter to keep the current value",
                    current=value,
                    conf=round(confidence, 4),
                )

        # 4. A transcript that produced nothing is a signal worth surfacing.
        if state.get("transcript_doc") and not (transcript or {}).get("courses"):
            add(
                "transcript.courses",
                "A transcript was provided but no courses could be read from it",
                "list a few key courses, comma-separated (or press Enter to skip)",
            )

        overflow = len(gaps) - MAX_GAPS_PER_ROUND
        if overflow > 0:
            gaps = gaps[:MAX_GAPS_PER_ROUND]

        updates: dict[str, Any] = {
            "gaps": gaps,
            "trace": [f"assess_gaps({len(gaps)})"],
        }
        if overflow > 0:
            updates["warnings"] = [
                f"{overflow} further gap(s) were not raised this round to keep the "
                f"prompt manageable; they remain recorded in the output."
            ]
        return updates

    return assess_gaps


def make_route_gaps(config: Config):
    def route_gaps(state: AgentState) -> str:
        if state.get("errors"):
            return "judge_skills"  # already broken; do not interrogate the user
        if not state.get("interactive", True):
            return "judge_skills"
        if not state.get("gaps"):
            return "judge_skills"
        if state.get("review_rounds", 0) >= config.max_review_rounds:
            return "judge_skills"
        return "human_review"

    return route_gaps
