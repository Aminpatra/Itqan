"""Whole-token skill matching, shared by the gap side and the course side.

Promoted from ``agents.agent_c_gap_analysis.nodes`` when a second consumer
appeared (``shared.course_market``), following the ``shared/grounding.py``
precedent — an agent may not import another agent, so anything two of them need
moves here.

The relation this expresses is **asymmetric**, which is exactly why cosine
similarity cannot stand in for it: "data analytics engineering" genuinely covers
a requirement for "data analytics", while symmetric similarity only reports that
the two phrases are 0.83 alike — the same number it gives pairs that do not cover
each other at all.

Matching is on WHOLE TOKENS, so "java" never matches inside "javascript". That
guard is load-bearing: it is the same class of bug as Agent A's grounding
matching "Java" inside a CV that only said "JavaScript".
"""

from __future__ import annotations

import re

# Kept in sync with the gap side deliberately: '+', '#' and '.' are part of skill
# names (c++, c#, node.js) and must not be split on.
_SPLIT = re.compile(r"[^a-z0-9+#.]+")


def tokens(text: str) -> list[str]:
    return [t for t in _SPLIT.split((text or "").lower()) if t]


def contains_tokens(container: str, needle: str) -> bool:
    """Does ``container`` contain ``needle`` as a whole-token subsequence?"""
    hay, need = tokens(container), tokens(needle)
    if not need or len(need) > len(hay):
        return False
    return any(hay[i:i + len(need)] == need for i in range(len(hay) - len(need) + 1))


# Single words that describe a FIELD rather than a skill. On their own they must
# never widen a search: a gap in "data" would otherwise pull in every "data
# analysis", "data science" and "data engineering" course going.
#
# An explicit list rather than a length threshold, because length does not encode
# genericness — "java" and "data" are both four characters and only one of them
# names a skill. These are the same terms the extraction prompts already tell the
# model to omit as filler, so a bare one here is a leak, not a real requirement.
# They remain matchable EXACTLY; only the widening is withheld.
_FIELD_WORDS = frozenset({
    "data", "business", "management", "development", "engineering", "analysis",
    "technology", "software", "computer", "digital", "systems", "science",
    "programming", "design", "marketing", "communication", "research",
})


def covers_skill(taught: str, required: str) -> bool:
    """Would a course teaching ``taught`` cover a gap in ``required``?

    True on an exact normalized match, or when either phrase contains the other
    as whole tokens — a "python programming" course covers "python", and a
    "python" course covers "python programming".

    A bare field word (see ``_FIELD_WORDS``) matches only exactly. That is the
    one place a loose match does real damage, and it is why Agent C's audit chose
    exact-only for unmapped skills; multi-word phrases carry enough tokens to
    mean something specific, so they widen safely.
    """
    a, b = tokens(taught), tokens(required)
    if not a or not b:
        return False
    if a == b:
        return True
    if len(b) == 1 and b[0] in _FIELD_WORDS:
        return False
    if len(a) == 1 and a[0] in _FIELD_WORDS:
        return False
    return contains_tokens(taught, required) or contains_tokens(required, taught)
