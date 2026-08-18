"""What the model is allowed to return.

Structured output, not prose to be parsed. `intent` in particular must be a
validated field: deciding whether someone wants a rerun by looking for the word
"rerun" in a sentence would make the trigger for spending a credit a string
match on model prose, which is exactly the kind of control this project does not
build.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class AssistantReply(BaseModel):
    """One turn from the model.

    Deliberately small. The model writes an answer and may raise a hand; it does
    not choose what happens next.
    """

    answer: str = Field(
        description=(
            "The reply, 2-4 sentences, in the SAME language as the user's "
            "question. Every figure must come from the facts provided — never "
            "estimate, never round, never fill a gap with a plausible number."
        )
    )

    intent: Literal["answer", "propose_rerun"] = Field(
        default="answer",
        description=(
            "'propose_rerun' ONLY when new results could genuinely change what "
            "the user is asking about — for example they ask whether new jobs "
            "have appeared since their last match. If the question can be "
            "answered from the facts already provided, this stays 'answer'. "
            "This is a suggestion; it does not start anything."
        ),
    )

    rerun_reason: Optional[str] = Field(
        default=None,
        description=(
            "One short sentence saying what a rerun would actually change. "
            "Required when intent is 'propose_rerun', so the user is deciding "
            "on a stated reason rather than on the offer alone."
        ),
    )

    job_refs: List[str] = Field(
        default_factory=list,
        description=(
            "Handles of the matched jobs worth showing, e.g. ['J1','J3'], taken "
            "ONLY from the FACTS block. The screen renders these as real cards "
            "with their own source and reasoning — so name the handle and do "
            "NOT write the employer, the title or any match figure into your "
            "answer. At most 3. A handle that is not in the FACTS is dropped."
        ),
    )

    course_refs: List[str] = Field(
        default_factory=list,
        description=(
            "The same, for recommended courses: ['C2']. At most 3."
        ),
    )

    suggestions: List[str] = Field(
        default_factory=list,
        description=(
            "Up to 3 follow-up QUESTIONS a person might ask next, phrased as "
            "they would say them, in their language. Questions, never commands, "
            "and never an offer to do something you cannot do."
        ),
    )

    out_of_scope: bool = Field(
        default=False,
        description=(
            "True when the question is not about this user's own results — "
            "another person's data, system-wide figures, or how this service is "
            "built. Answer with a brief decline in that case. This is a "
            "courtesy signal, NOT the access control: the facts you are given "
            "contain one user's results and nothing else, so there is nothing "
            "else to disclose."
        ),
    )
