"""What the model is allowed to return.

Structured output, not prose to be parsed. `intent` in particular must be a
validated field: deciding whether someone wants a rerun by looking for the word
"rerun" in a sentence would make the trigger for spending a credit a string
match on model prose, which is exactly the kind of control this project does not
build.
"""

from __future__ import annotations

from typing import Literal, Optional

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
