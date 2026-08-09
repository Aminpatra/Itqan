"""Keep only what the posting actually said.

MEASURED 2026-08-08, on the first live cycle that extracted these fields:
**19 of 19 postings came back `work_arrangement = 'onsite'`, and 0 of the 19
contained any arrangement phrase at all** — in either language. Not one said
"on-site", "حضوري", "عن بعد", "hybrid", anything. The model read "Muscat, Oman"
and answered the question it was asked.

The extraction prompt already forbids this in as many words — *"An office
address is NOT a statement that the role is onsite"* — and the instruction did
not survive contact with a real corpus. That is this repo's standing finding,
now on its third field: Agent E called an unpriced course "free" once in 25
guarded draws with the warning present; Agent A's model paraphrased quotes it
was told to copy. **Instructions are not a control. Verification is.**

So the fields go through the same discipline every other model output in this
system goes through: `shared/grounding.verify_quote` for Agent A's fields,
`verify_quote` for Agent B's legitimacy evidence, `verify_claims` for Agent E's
rationale — and this for a posting's stated facts. The check is deterministic,
bilingual, and cheap; a field the source cannot support becomes NULL, which is
what it should have been.

WHY A VOCABULARY LIST RATHER THAN AN LLM SECOND OPINION: the question "does this
text contain a word for remote work" has a right answer that a regex knows. A
second model call would cost money to be wrong in a new way, and could not be
audited by reading it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from shared.text_norm import normalize_ar

# ---------------------------------------------------------------------------
# The vocabulary. Arabic first for each concept, because the corpus is Omani and
# the systematic-Arabic-penalty bug (0.560 AR vs 0.700 EN on an identical
# posting) came from exactly the habit of writing the English forms and assuming
# the Arabic followed.
#
# Matched against text put through `normalize_ar`, so أ/إ/آ/ا, ة/ه, ى/ي and
# tatweel variants all collapse — the same normalisation `legitimacy.py` uses.
# ---------------------------------------------------------------------------
_ARRANGEMENT = {
    "remote": ("remote", "work from home", "telecommut", "wfh",
               "عن بعد", "من المنزل", "العمل عن بعد"),
    "hybrid": ("hybrid", "هجين", "مختلط", "عمل مختلط"),
    "onsite": ("on-site", "onsite", "on site", "in-office", "in office",
               "office-based", "office based", "حضوري", "في الموقع",
               "بالموقع", "من المكتب", "في المكتب"),
}

_EMPLOYMENT = {
    "full_time": ("full-time", "full time", "fulltime", "دوام كامل", "بدوام كامل"),
    "part_time": ("part-time", "part time", "parttime", "دوام جزئي", "بدوام جزئي"),
    "contract": ("contract", "contractor", "fixed-term", "fixed term",
                 "عقد", "بعقد", "تعاقد"),
    "internship": ("internship", "intern ", "trainee", "apprentice",
                   "تدريب", "متدرب", "تدريبية"),
    "temporary": ("temporary", "temp ", "seasonal", "مؤقت", "موسمي"),
}

_PERIOD = {
    "hour": ("hour", "hourly", "/hr", "ساعة", "بالساعة"),
    "day": ("day", "daily", "per day", "يوم", "يوميا"),
    "week": ("week", "weekly", "أسبوع", "اسبوع"),
    "month": ("month", "monthly", "per month", "pm", "شهر", "شهريا", "بالشهر"),
    "year": ("year", "annual", "annually", "per annum", "pa", "سنة", "سنوي", "سنويا"),
}


@dataclass
class StatedFacts:
    """What survived verification, and what did not."""

    work_arrangement: str | None = None
    employment_type: str | None = None
    salary_min: float | None = None
    salary_max: float | None = None
    salary_currency: str | None = None
    salary_period: str | None = None
    dropped: tuple[str, ...] = ()


def _norm(text: str) -> str:
    """Fold Arabic orthography and Arabic-Indic digits, then flatten whitespace.

    Digit folding is load-bearing for the salary check: a posting writing
    ٨٠٠ ريال and a model answering 800 agree, and without folding the figure
    would look fabricated and be discarded.
    """
    return re.sub(r"\s+", " ", normalize_ar(text or "")).lower()


def _mentions(blob: str, phrases: tuple[str, ...]) -> bool:
    return any(p in blob for p in phrases)


def _numbers_in(blob: str) -> set[str]:
    """Every number the text contains, thousands separators removed.

    `1,200` and `1200` are the same figure written two ways; a posting almost
    always writes the first and a model almost always answers the second.
    """
    return {n.replace(",", "").replace(".0", "")
            for n in re.findall(r"\d[\d,]*(?:\.\d+)?", blob.replace(",", ","))}


def verify_stated_facts(job, source_text: str) -> StatedFacts:
    """Null out every field the source does not support.

    Deliberately asymmetric, in the direction that costs us data rather than
    credibility: a posting that says "remote" while the model answered "onsite"
    loses the field entirely rather than being corrected to "remote". Being
    wrong about which is worse than being silent, and disagreement between the
    text and the model is not evidence for either reading.
    """
    blob = _norm(f"{getattr(job, 'title', '') or ''} {source_text}")
    dropped: list[str] = []

    arrangement = getattr(job, "work_arrangement", None)
    if arrangement:
        if not _mentions(blob, _ARRANGEMENT.get(arrangement, ())):
            dropped.append("work_arrangement")
            arrangement = None
        elif any(_mentions(blob, ph) for k, ph in _ARRANGEMENT.items()
                 if k != arrangement):
            # The text names a different arrangement too. Ambiguous, so silent.
            dropped.append("work_arrangement")
            arrangement = None

    employment = getattr(job, "employment_type", None)
    if employment and not _mentions(blob, _EMPLOYMENT.get(employment, ())):
        dropped.append("employment_type")
        employment = None

    # Pay: every figure must appear in the text, exactly as Agent E's
    # `verify_rationale` requires every number in a rationale to appear in the
    # fact sheet it was given. A salary nobody published is the most damaging
    # thing on this list to invent — it is what a person decides on.
    numbers = _numbers_in(blob)
    s_min = getattr(job, "salary_min", None)
    s_max = getattr(job, "salary_max", None)

    def _stated(value) -> bool:
        if value is None:
            return True
        text = f"{value:.0f}" if float(value).is_integer() else f"{value}"
        return text in numbers

    if not (_stated(s_min) and _stated(s_max)):
        dropped.append("salary")
        s_min = s_max = None

    currency = getattr(job, "salary_currency", None)
    period = getattr(job, "salary_period", None)
    if s_min is None and s_max is None:
        # A currency and a period with no amount describe nothing.
        currency = period = None
    else:
        if currency and _norm(currency) not in blob:
            dropped.append("salary_currency")
            currency = None
        if period and not _mentions(blob, _PERIOD.get(period, ())):
            dropped.append("salary_period")
            period = None

    return StatedFacts(
        work_arrangement=arrangement,
        employment_type=employment,
        salary_min=s_min,
        salary_max=s_max,
        salary_currency=currency,
        salary_period=period,
        dropped=tuple(dropped),
    )
