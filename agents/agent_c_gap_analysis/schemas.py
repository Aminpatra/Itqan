"""Structured output for the fenced skill-matching tier.

Agent C is otherwise arithmetic end to end. This is the one place a model is
asked anything, and the schema is the fence: the answer is a validated object
over a closed vocabulary, never prose, and every ``satisfied_by`` is checked
against the candidate's real skill list before it is believed.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class SkillVerdict(BaseModel):
    """One requirement, answered.

    ``uncertain`` is a first-class answer, not a failure. The deterministic
    verdict stands whenever the model is unsure, so guessing is never the
    lower-effort path — which is what stops the model resolving ambiguity in
    whichever direction sounds most helpful.
    """

    requirement: str = Field(
        description="The job requirement being judged, copied EXACTLY as given."
    )
    decision: Literal["satisfied", "not_satisfied", "uncertain"]
    # Required for `satisfied`, and verified against the candidate's real list.
    # A name that is not in that list voids the verdict.
    satisfied_by: Optional[str] = Field(
        default=None,
        description="The candidate skill that satisfies it, copied EXACTLY from "
        "the candidate's list. Null unless decision is 'satisfied'.",
    )
    reason: str = Field(description="One short clause. No more.")


class SkillMatchBatch(BaseModel):
    verdicts: list[SkillVerdict] = Field(default_factory=list)
