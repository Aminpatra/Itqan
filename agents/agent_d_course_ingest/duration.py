"""Turn a provider's stated duration into a range, or into nothing.

Coursera's `workload` is free text written by course authors, and the real
corpus contains at least these shapes:

    '1 hour 30 minutes'
    '4 weeks of study, 2-4 hours a week'
    '2 heures'                              (French — the catalogue is multilingual)
    'Approx. 12 hours'
    ''

**A range, never a midpoint.** '2-4 hours a week' over '4 weeks' is 8 to 16
hours; reporting it as 12 would publish a number no provider ever stated. That
is the reason durations were left unstored the first time rather than guessed
at, and it is the same rule that keeps `gap_score` null instead of 0.0 and a
course's price null instead of free.

**Unparseable is not zero.** Anything this module cannot read returns `(None,
None)` and the caller keeps the raw text, so a person still sees what the
provider wrote even when no arithmetic was possible.
"""

from __future__ import annotations

import re
from typing import Optional

# Units, in hours. Multilingual because the catalogue is: French appears in the
# live data, and Spanish costs one line to accept rather than one bug to find.
_UNITS: dict[str, float] = {
    "minute": 1 / 60, "minutes": 1 / 60, "min": 1 / 60, "mins": 1 / 60,
    "minuto": 1 / 60, "minutos": 1 / 60,
    "hour": 1.0, "hours": 1.0, "hr": 1.0, "hrs": 1.0,
    "heure": 1.0, "heures": 1.0, "hora": 1.0, "horas": 1.0,
}

# "2-4", "2 - 4", "2 to 4", "2–4" (en dash). The separator set is explicit
# because a hyphen inside a word must not read as a range.
_RANGE = r"(\d+(?:\.\d+)?)\s*(?:-|–|—|to|a|à)\s*(\d+(?:\.\d+)?)"

# Every unit, not just the hour-words. "30 min/week" matched none of them and
# fell through to the single-figure rule below, which read it as "30 minutes
# total" — half an hour for a course that runs for weeks. Found on live data,
# and the same mistake as reading "3-5 hours/week" as a total.
_PER_WEEK = re.compile(
    rf"(?:{_RANGE}|(\d+(?:\.\d+)?))\s*(?:\+)?\s*"
    rf"({'|'.join(_UNITS)})\s*(?:a|per|/|par|por)\s*week",
    re.I)
_WEEKS = re.compile(r"(\d+(?:\.\d+)?)\s*(weeks?|semaines?|semanas?)", re.I)
_RANGE_UNIT = re.compile(rf"{_RANGE}\s*({'|'.join(_UNITS)})\b", re.I)
_SINGLE_UNIT = re.compile(rf"(\d+(?:\.\d+)?)\s*({'|'.join(_UNITS)})\b", re.I)


def parse_workload(text: Optional[str]) -> tuple[Optional[float], Optional[float]]:
    """`(hours_min, hours_max)` for a stated duration, or `(None, None)`.

    Equal endpoints mean the provider stated ONE figure, not that a range was
    averaged. Nothing here ever returns a midpoint.
    """
    raw = (text or "").strip()
    if not raw:
        return None, None

    # "N weeks of study, X-Y hours a week" — a total, and the only shape here
    # that multiplies. Handled first because the string also contains a bare
    # "N weeks" that the single-unit rule below would otherwise misread.
    per_week = _PER_WEEK.search(raw)
    weeks = _WEEKS.search(raw)
    if per_week:
        if not weeks:
            # "3-5 hours/week" with no week count. The TOTAL is unknowable, and
            # falling through to the plain range rule below would answer "3-5
            # hours" — the effort for one week presented as the whole course.
            # Found on live data, where it understated a course by however many
            # weeks it runs. The words are kept and the numbers are not invented.
            return None, None
        low, high, single, unit = per_week.groups()
        # The unit is APPLIED, not assumed to be hours: "30 min/week for 6 weeks"
        # is 3 hours, and treating the 30 as hours would claim 180.
        factor = _UNITS[unit.lower()]
        span = float(weeks.group(1))
        if single is not None:
            total = float(single) * factor * span
            return _round(total), _round(total)
        return _round(float(low) * factor * span), _round(float(high) * factor * span)

    # "2-4 hours", "90-120 minutes"
    ranged = _RANGE_UNIT.search(raw)
    if ranged:
        low, high, unit = ranged.groups()
        factor = _UNITS[unit.lower()]
        lo, hi = float(low) * factor, float(high) * factor
        # A backwards range is the author's typo, not a fact. Ordering it is the
        # conservative reading; refusing it would lose a real duration.
        return _round(min(lo, hi)), _round(max(lo, hi))

    # "1 hour 30 minutes", "Approx. 12 hours", "2 heures" — every unit-bearing
    # figure in the string, summed. One value, so both ends are the same.
    total = 0.0
    found = False
    for value, unit in _SINGLE_UNIT.findall(raw):
        total += float(value) * _UNITS[unit.lower()]
        found = True
    if found:
        return _round(total), _round(total)

    return None, None


def _round(hours: float) -> float:
    """Two decimals: enough for '1 hour 30 minutes' to be exactly 1.5, and not
    so much that floating-point noise reaches a database column."""
    return round(hours, 2)


def parse_iso8601(value: Optional[str]) -> tuple[Optional[float], Optional[float]]:
    """`PT6H30M` / `P4W` — schema.org's duration format, which edX publishes.

    Weeks are DELIBERATELY not converted: `P4W` is elapsed calendar time, not
    study hours, and treating four weeks as 672 hours of effort would be a
    fabrication several orders of magnitude wrong.
    """
    text = (value or "").strip().upper()
    match = re.fullmatch(r"P(?:(\d+(?:\.\d+)?)D)?(?:T(?:(\d+(?:\.\d+)?)H)?"
                         r"(?:(\d+(?:\.\d+)?)M)?)?", text)
    if not match or not any(match.groups()):
        return None, None
    days, hours, minutes = (float(g) if g else 0.0 for g in match.groups())
    total = days * 24 + hours + minutes / 60
    return (_round(total), _round(total)) if total else (None, None)
