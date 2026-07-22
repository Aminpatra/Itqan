"""Structured outputs the LLM must fill — extraction and legitimacy adjudication.

The same convention Agent A established holds here and is load-bearing: an
optional field defaults to ``None``, and ``None`` means "the posting did not
state this." Never give one a non-null default. Once a fabricated value reaches
the store it is indistinguishable from a real one, and the whole point of Agent B
is that a number Agent C trusts is never invented.

The two enum-like fields — ``sector``, ``country`` — are constrained to closed
vocabularies so the model cannot quietly widen them. A sector outside ISCO-08 or
a country that is not two letters is a extraction error we want to see as a
validation failure, not as a novel row in ``skill_demand_stats``.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

# ISCO-08 major groups, the sector vocabulary. A string, not an int, because
# "unknown" has to be expressible and a bare 0 would look like a real group.
ISCO_MAJOR_GROUPS = {
    "1": "Managers",
    "2": "Professionals",
    "3": "Technicians and Associate Professionals",
    "4": "Clerical Support Workers",
    "5": "Services and Sales Workers",
    "6": "Skilled Agricultural, Forestry and Fishery Workers",
    "7": "Craft and Related Trades Workers",
    "8": "Plant and Machine Operators and Assemblers",
    "9": "Elementary Occupations",
    "0": "Armed Forces Occupations",
}

LISTING_INTENTS = ("vacancy", "seeking", "service", "unknown")
POSTER_TYPES = ("company", "individual", "unknown")


class JobExtraction(BaseModel):
    """What the LLM reads out of one posting's text.

    Everything here is derived ONLY from the posting body. The model is told, and
    it matters, that it must not import outside knowledge — "this is a Baker
    Hughes posting so the salary is probably X" is exactly the fabrication the
    downstream numbers cannot absorb.
    """

    # ISCO-08 major group as its single-digit code, or None when the posting
    # does not make the occupation clear. Never guessed into a default: a NULL
    # sector is excluded from aggregation, a wrong one silently miscounts.
    sector: Optional[str] = Field(
        default=None,
        description="ISCO-08 major group code, single digit '0'-'9', or null if unclear.",
    )
    # The employer, verbatim from the posting. None when no employer is named —
    # which is itself a legitimacy signal, so approximating it here would switch
    # that check off.
    company: Optional[str] = None
    # Free text, deduplicated downstream by lower(btrim()). No stemming or
    # synonym merging — that fabricates structure the data does not contain.
    required_skills: list[str] = Field(default_factory=list)
    seniority_level: Optional[str] = None
    location: Optional[str] = None
    # ISO-3166 alpha-2, uppercased, or None. Determined from the posting text,
    # NEVER from the source's own nationality — a regional board carries
    # postings for several countries, and scope is a separate axis from scam.
    country: Optional[str] = None

    # Read from the CONTENT, not from source metadata. "I am looking for a job"
    # is the poster stating their own intent; a named employer hiring is a
    # vacancy. Both stay 'unknown' when the text does not make them explicit, and
    # only ('vacancy', 'company') is eligible for aggregation.
    listing_intent: Literal["vacancy", "seeking", "service", "unknown"] = "unknown"
    poster_type: Literal["company", "individual", "unknown"] = "unknown"

    @field_validator("sector")
    @classmethod
    def _known_sector(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        if value not in ISCO_MAJOR_GROUPS:
            # Reject rather than coerce: a sector the model invented is a signal
            # the extraction is unreliable, not a value to salvage.
            raise ValueError(f"sector {value!r} is not an ISCO-08 major group")
        return value

    @field_validator("country")
    @classmethod
    def _iso_alpha2(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip().upper()
        if len(value) != 2 or not value.isalpha():
            raise ValueError(f"country {value!r} is not an ISO-3166 alpha-2 code")
        return value

    @field_validator("company", "seniority_level", "location")
    @classmethod
    def _empty_is_none(cls, value: Optional[str]) -> Optional[str]:
        # The model sometimes emits "" or "N/A" for absent fields; normalise them
        # to None so absence has one representation the grounder can rely on.
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned or cleaned.casefold() in {"n/a", "none", "null", "unknown", "not specified"}:
            # "null" the string included: seen in live output — the model wrote
            # the WORD null into an optional field instead of leaving it null.
            return None
        return cleaned

    @field_validator("required_skills")
    @classmethod
    def _clean_skills(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for skill in value:
            cleaned = (skill or "").strip()
            key = cleaned.casefold()
            if cleaned and key not in seen:
                seen.add(key)
                out.append(cleaned)
        return out


class LegitimacyVerdict(BaseModel):
    """The adjudicator's answer for a posting in the uncertain band.

    Only consulted for postings the deterministic rules could neither clear nor
    reject. Its ``evidence_quote`` is checked against the source with
    ``shared.grounding.verify_quote`` before the verdict is trusted — a model
    that cannot cite a real span does not get to overrule the rules. That check
    is why ``evidence_quote`` is required, not optional.
    """

    is_scam: bool = Field(description="True if this posting is likely fraudulent.")
    # Must appear verbatim in the posting. A fabricated quote discards the whole
    # verdict and the deterministic rule score stands.
    evidence_quote: str = Field(
        description="A short span copied EXACTLY from the posting that supports the verdict."
    )
    reasoning: str = Field(description="One sentence, grounded in the quoted span.")

    # 0.0 = certainly legitimate, 1.0 = certainly a scam. Combined with the rule
    # score by the node, not used raw.
    scam_confidence: float = Field(ge=0.0, le=1.0, default=0.5)
