"""Structured output for course extraction.

The one output that matters is ``taught_skills`` — the supply-side mirror of
Agent B's ``required_skills``, and the field the whole demand-vs-supply join
rests on. Same discipline: short canonical English skill names, derived from the
course text, never invented. An optional field is None when the course did not
state it.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

LEVELS = ("beginner", "intermediate", "advanced")


class CourseExtraction(BaseModel):
    # Skills the course TEACHES, as 1-4 word canonical English names — the form a
    # recruiter would type in a search box, so they aggregate with the demand
    # side. Condensing the course's own wording is allowed; adding skills the
    # course does not cover is not.
    taught_skills: list[str] = Field(default_factory=list)
    # beginner / intermediate / advanced, only if the course states or clearly
    # implies it. Null otherwise — never guessed to fill the field.
    level: Optional[Literal["beginner", "intermediate", "advanced"]] = None
    # A broad subject/domain label, if clear (e.g. "data science", "web
    # development"). One phrase, or null.
    subject: Optional[str] = None

    @field_validator("taught_skills")
    @classmethod
    def _clean(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for skill in value:
            cleaned = (skill or "").strip()
            key = cleaned.casefold()
            if cleaned and key not in seen:
                seen.add(key)
                out.append(cleaned)
        return out

    @field_validator("subject")
    @classmethod
    def _empty_none(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned or cleaned.casefold() in {"n/a", "none", "unknown", "null"}:
            return None
        return cleaned
