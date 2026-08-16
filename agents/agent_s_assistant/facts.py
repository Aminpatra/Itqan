"""The fact sheet, and the check that the answer stayed inside it.

This module is the fence. Everything Agent S is capable of saying comes from
`build_fact_sheet`, and `verify_answer` rejects a reply that went beyond it.

Both halves matter, and the second exists because the first is not enough on its
own. Agent E was given a fact sheet too, and on its first real-model run **6 of 8
rationales** made claims the record contradicted while passing a numbers-only
check. A fence you do not test is a fence you hope is holding.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Optional

# Reused from Agent E's verifier, deliberately identical in behaviour: comma
# stripping and trailing-zero normalisation so "1,919" and "1919" — and "4.70"
# and "4.7" — are the same token.
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")

# Internal vocabulary that must never reach a user. `gap_score` is a number
# whose scale is backwards from intuition (0.0 is the BEST), and a user reading
# "your gap score is 0.0" would draw exactly the wrong conclusion.
_FORBIDDEN = ("esco", "gap_score", "priority_score", "skill_key", "posting_id",
              "duplicate_of", "final_url", "fact sheet", "facts block")


def _numbers(text: str) -> set[str]:
    out: set[str] = set()
    for raw in _NUMBER.findall(text or ""):
        token = raw.replace(",", "")
        if "." in token:
            token = token.rstrip("0").rstrip(".")
        out.add(token or "0")
    return out


def _when(value: Any) -> str:
    """A date a person can read, and how long ago it was.

    The raw value reached a real user verbatim — *"The results last produced on
    2026-08-09T04:09:49.388371+00:00"* — inside a sentence about their career.
    The model echoed exactly what it was handed, so the fix belongs in what it is
    handed rather than in an instruction not to repeat it.

    "6 days ago" is not decoration either: "are my results out of date?" is one
    of the questions this assistant exists to answer, and it cannot be answered
    from a timestamp without arithmetic the model should not be doing.
    """
    if not value:
        return "no completed match yet"
    try:
        when = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return str(value)
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)

    # `%-d` strips the leading zero on Linux and raises on Windows, so the day
    # is assembled by hand.
    stamp = f"{when.day} {when:%B %Y}"
    days = (datetime.now(timezone.utc) - when).days
    if days < 0:
        return stamp
    if days == 0:
        return f"{stamp} (today)"
    if days == 1:
        return f"{stamp} (yesterday)"
    if days <= 30:
        return f"{stamp} ({days} days ago)"
    return stamp


def build_fact_sheet(*, readiness: Any, jobs: list[dict[str, Any]],
                     courses: list[dict[str, Any]], gaps: list[str],
                     suggested_role: Optional[dict[str, Any]],
                     matched_at: Optional[str]) -> str:
    """One user's results, as lines the model may quote from.

    Built from `api/mapping`'s already-published shapes, so the honesty work
    already done there is inherited rather than repeated: a null readiness stays
    null instead of becoming 0, and a job with no evidence chain was already
    dropped upstream.

    Absent values are written as "not measured" rather than omitted. An omitted
    line invites the model to fill the gap; a line that states the absence gives
    it the true answer to quote.
    """
    lines: list[str] = []

    lines.append(
        f"Readiness score: {readiness}/100" if readiness is not None
        else "Readiness score: not measured yet"
    )
    lines.append(f"Results last produced: {_when(matched_at)}")

    if suggested_role and suggested_role.get("title"):
        lines.append(f"Role the analysis suggests: {suggested_role['title']}"
                     + (f" — {suggested_role['why']}" if suggested_role.get("why") else ""))
    else:
        lines.append("Role the analysis suggests: none yet")

    lines.append("")
    lines.append(f"Matched jobs ({len(jobs)}):" if jobs else "Matched jobs: none yet")
    for job in jobs:
        bits = [job.get("title") or "untitled"]
        if job.get("employer"):
            bits.append(f"at {job['employer']}")
        if job.get("location"):
            bits.append(f"in {job['location']}")
        line = f"- {' '.join(bits)}"
        if job.get("why"):
            line += f" — why it matched: {job['why']}"
        lines.append(line)

    lines.append("")
    if not gaps:
        lines.append("Skills they are missing: none recorded")
    else:
        # TWO DIFFERENT RANKINGS, and conflating them was a real bug in this
        # file — caught on live data within minutes of adding the counts.
        #
        # The list arrives in Agent C's order: summed demand across the WHOLE
        # market. The number after each skill is something else entirely: how
        # many of THIS PERSON'S matched roles asked for it. They can disagree
        # violently. Measured on a real profile: `API integration` was wanted by
        # 5 of the matched roles — the highest count in the list — and sat LAST,
        # because market-wide it weighs least.
        #
        # The first label here said "most in demand first", and the model
        # sensibly ignored it and reasoned from the visible counts instead. It
        # was right and the label was wrong. So the ordering is now named for
        # what it actually is, and the model is told plainly that position and
        # count are not the same claim.
        lines.append(
            f"Skills they are missing ({len(gaps)}), ordered by demand across the "
            f"whole job market. The number after each is how many of THIS "
            f"PERSON'S matched roles asked for it — a skill low in this list can "
            f"still have the highest count, so do not call the first one the most "
            f"needed:")
        for gap in gaps:
            if isinstance(gap, str):
                lines.append(f"- {gap}")
                continue
            line = f"- {gap.get('skill') or gap.get('name')}"
            wanted = gap.get("jobs_missing_in")
            if wanted:
                # A measured count, publishable as-is. `priority_score` is NOT
                # published: it is an internal weight on no meaningful scale, and
                # `_FORBIDDEN` rejects any answer that names it.
                line += f" — {wanted} of the roles they matched asked for it"
                if gap.get("low_confidence"):
                    line += " (from thin data)"
            lines.append(line)

    lines.append("")
    lines.append(f"Recommended courses ({len(courses)}):" if courses
                 else "Recommended courses: none yet")
    for course in courses:
        bits = [course.get("title") or "untitled"]
        if course.get("provider"):
            bits.append(f"from {course['provider']}")
        line = f"- {' '.join(bits)}"
        if course.get("why"):
            line += f" — {course['why']}"
        lines.append(line)

    return "\n".join(lines)


def verify_answer(text: str, fact_sheet: str) -> Optional[str]:
    """Why this answer must not be published, or None if it may be.

    Two checks, both of which the record can settle:

    1. **Every number must appear in the fact sheet.** A figure the model
       produced that nobody measured is the failure mode this project has hit in
       every agent that writes prose.
    2. **No internal vocabulary.** `gap_score` in particular is scaled so that
       0.0 is the best possible result, so a user shown that number reads it
       backwards.

    What is deliberately NOT checked: vague reassurance ("you're in good shape").
    The record cannot adjudicate it, and a check that pretends to would be
    theatre. Stated here so the limit is known rather than assumed away.
    """
    if not (text or "").strip():
        return "the answer was empty"

    lowered = (text or "").lower()
    for token in _FORBIDDEN:
        if token in lowered:
            return f"uses internal vocabulary: {token!r}"

    invented = _numbers(text) - _numbers(fact_sheet)
    if invented:
        return f"states figures not in the record: {', '.join(sorted(invented))}"

    return None


def deterministic_answer(fact_sheet: str, *, has_results: bool) -> str:
    """What is said when the model's answer is rejected, errors, or is absent.

    Not an apology and not an empty string: a real answer built from the record,
    pointing at what is actually there. `""` masquerading as "no answer" is the
    bug Agent E had to fix, so this always returns something a person can read.
    """
    if not has_results:
        return ("Your results are not ready yet — once your first match finishes, "
                "I can talk you through what it found.")
    return ("I can only go on what your results actually record, and I could not "
            "answer that from them. You can see the full picture on your dashboard.")
