"""Agent S's nodes: ask the model, then check what it said.

Two nodes, and the second is the reason the first is allowed to exist. `answer`
is the only LLM call in this agent; `verify` decides whether its output reaches a
person or is replaced by a deterministic one.

Dependencies are injected exactly as in Agents A-E, so the whole suite runs
offline against a fake LLM and no test in this repo makes a network call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from shared.config import Config

from .facts import clean_suggestions, deterministic_answer, verify_answer
from .prompts import ASSISTANT_PROMPT
from .schemas import AssistantReply
from .state import AssistantState


@dataclass
class Deps:
    config: Config
    # None disables the model entirely and takes the deterministic path. Useful
    # in tests, and it means an outage degrades to a real sentence rather than an
    # error page.
    llm: Any = None


def _history_text(history: list[dict[str, Any]], *, limit: int) -> str:
    """The recent turns, flattened for the prompt.

    Truncated per message as well as by count: ten messages a day keeps this
    small, but a single pasted wall of text should not be able to push the fact
    sheet out of the model's attention.
    """
    lines = []
    for turn in (history or [])[-limit:]:
        who = "Them" if turn.get("role") == "user" else "You"
        lines.append(f"{who}: {(turn.get('content') or '')[:500]}")
    return "\n".join(lines)


def make_answer(deps: Deps) -> Callable[[AssistantState], dict[str, Any]]:
    def answer(state: AssistantState) -> dict[str, Any]:
        if deps.llm is None:
            return {"answer": "", "answer_source": "template", "model_failed": True,
                    "warnings": ["no model configured"]}

        try:
            # Composition is inside the try as well as the call: `prompt | llm`
            # raises TypeError for anything LCEL cannot adapt, and a
            # misconfigured model should degrade exactly like an unreachable one
            # rather than 500 the request.
            chain = ASSISTANT_PROMPT | deps.llm
            reply: AssistantReply = chain.invoke({
                "facts": state.get("fact_sheet", ""),
                "history": _history_text(
                    state.get("history") or [],
                    limit=deps.config.assistant_history_turns),
                "question": state.get("question", ""),
            })
        except Exception as exc:  # noqa: BLE001 - an outage is not a crash
            return {"answer": "", "answer_source": "template", "model_failed": True,
                    "warnings": [f"model call failed: {type(exc).__name__}: {exc}"]}

        return {
            "answer": reply.answer,
            "answer_source": "model",
            "model_failed": False,
            # Carried, not acted on. Nothing downstream of here starts anything.
            "proposed_rerun": reply.intent == "propose_rerun",
            "rerun_reason": reply.rerun_reason,
            "out_of_scope": reply.out_of_scope,
            # Handles, not cards. `resolve_refs` decides at the route whether
            # each one points at something the model was actually shown; an
            # invented [J9] becomes nothing rather than a fabricated posting.
            "job_refs": list(reply.job_refs or [])[:3],
            "course_refs": list(reply.course_refs or [])[:3],
            # Cleaned, not just trimmed: a suggestion is user-facing text that
            # never passes through `verify_answer`, and the same live run that
            # caught handles in the prose produced the chip "why did J1 match
            # me?" — a tappable question nobody can read.
            "suggestions": clean_suggestions(reply.suggestions),
        }

    return answer


def make_verify(deps: Deps) -> Callable[[AssistantState], dict[str, Any]]:
    def verify(state: AssistantState) -> dict[str, Any]:
        text = state.get("answer") or ""
        fact_sheet = state.get("fact_sheet", "")

        problem: Optional[str] = None
        if state.get("answer_source") == "model":
            problem = verify_answer(text, fact_sheet)

        if state.get("answer_source") == "template" or problem:
            return {
                "answer": deterministic_answer(
                    fact_sheet, has_results=bool(state.get("has_results"))),
                "answer_source": "template",
                # A rejected answer's suggestion goes with it. The model that
                # just said something unsupported does not get to keep proposing
                # that the user spend their one weekly credit.
                "proposed_rerun": False,
                "rerun_reason": None,
                # The cards go with the sentence they belonged to. The
                # deterministic reply is about a DIFFERENT thing — it says we
                # could not answer — and leaving two job cards under it would
                # attach evidence to a claim nobody made.
                "job_refs": [],
                "course_refs": [],
                "suggestions": [],
                "warnings": [f"answer replaced: {problem}"] if problem else [],
            }

        return {}

    return verify
