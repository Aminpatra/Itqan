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

# A card handle that leaked into user-facing prose.
#
# Measured on the real model, 2026-08-18: asked "which jobs fit me right now?"
# it answered "These jobs matched you right now: [J1], [J2], [J3]..." — naming
# SEVEN while three were attached. The handles are an internal pointer, like
# `gap_score`; a person reading "[J1]" has been shown a token that means nothing
# to them, and a sentence promising seven cards above three cards is worse than
# meaningless.
#
# The prompt now says not to. This is the check that does not depend on the
# model reading it.
_HANDLE = re.compile(r"\[[JC]\d+\]")
# BRACKETED ONLY, and that is a deliberate limit rather than an oversight.
# A bare `C1` is a CEFR language level and `A2 English for Developers` is a
# real course in this corpus, so matching unbracketed tokens would reject
# honest answers about language courses. The observed failure was the
# bracketed form — the model echoes the notation it was shown — and the
# prompt covers the rest. A check that fires on true sentences is worse
# than one with a known edge.


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
    for i, job in enumerate(jobs, 1):
        # THE HANDLE. The model never writes a job into its prose — it names
        # `[J2]`, and code turns that into the real card. See `resolve_refs`.
        #
        # A short label rather than the row's `job_id`: that id is a sha, and
        # asking a model to reproduce one exactly invites a near-miss that
        # resolves to a DIFFERENT posting. `[J2]` either matches or it does not.
        bits = [job.get("title") or "untitled"]
        if job.get("employer"):
            bits.append(f"at {job['employer']}")
        if job.get("location"):
            bits.append(f"in {job['location']}")
        line = f"- [J{i}] {' '.join(bits)}"
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
    for i, course in enumerate(courses, 1):
        bits = [course.get("title") or "untitled"]
        if course.get("provider"):
            bits.append(f"from {course['provider']}")
        line = f"- [C{i}] {' '.join(bits)}"
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
       backwards. Card handles count: `[J1]` is a pointer for the code that
       resolves it, and it means nothing to the person reading the sentence.

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

    leaked = _HANDLE.search(text or "")
    if leaked:
        return f"names an internal card handle: {leaked.group(0)!r}"

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


_REF = re.compile(r"^\s*\[?([JC])(\d+)\]?\s*$", re.IGNORECASE)


def clean_suggestions(items: Any) -> list[str]:
    """Follow-up questions, minus any that leaked a handle.

    `verify_answer` guards the ANSWER; nothing guarded these. The same live run
    that caught "[J1], [J2]" in the prose also produced the suggestion chip
    "why did J1 match me?" — which a person would be invited to tap.

    Dropped rather than rewritten: a question with the handle stripped out
    ("why did match me?") is worse than one fewer chip.
    """
    out: list[str] = []
    for item in (items or []):
        text = str(item).strip()
        if text and not _HANDLE.search(text):
            out.append(text)
    return out[:3]


def resolve_refs(refs: Any, items: list[dict[str, Any]], *, kind: str) -> list[dict[str, Any]]:
    """Turn the model's handles into real cards, dropping anything invented.

    **This is the anti-fabrication check for attachments**, and it is the same
    discipline Agent C uses on a cited satisfying skill and Agent E on a claimed
    figure: the model points at evidence it was shown, and CODE decides whether
    the pointer resolves. `[J9]` against a sheet listing three jobs yields
    nothing — not a guess, not the nearest one, nothing.

    The card returned is the very dict `api/mapping` built for `/api/jobs`, so a
    posting attached to a chat turn and the same posting on the Jobs screen are
    byte-identical and cannot drift apart. That is what carries `why`, the live
    `source` and its `retrievedAt` onto this surface — the whole reason a
    posting may be ATTACHED here while the brand bars Hud from describing one.

    Order and duplicates are the model's to choose but not to abuse: the first
    mention wins and a repeat is ignored, so a reply naming `[J1]` three times
    renders one card.
    """
    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    for ref in (refs or []):
        match = _REF.match(str(ref))
        if not match or match.group(1).upper() != kind.upper():
            continue
        index = int(match.group(2))
        if index < 1 or index > len(items) or index in seen:
            continue
        seen.add(index)
        out.append(items[index - 1])
    return out
