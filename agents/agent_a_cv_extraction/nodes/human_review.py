"""The interrupt node.

READ BEFORE EDITING. On resume, LangGraph re-executes this node from the top —
``interrupt()`` does not resume mid-function, it replays the whole node and
returns the resume value at the same call site. Anything above that line runs
twice.

So this node does exactly one thing: build a cheap, deterministic payload and
interrupt. No printing, no file writes, no LLM calls, no counters. Console
rendering lives in ``cli.py``; validating the answers lives in the next node.
That split is the entire reason HITL is two nodes rather than one.
"""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from ..state import AgentState

# Resuming with an empty dict does NOT work on langgraph 1.2.9: an empty mapping
# is read as a resume-map with no entries, so nothing is written, the node
# re-runs, and the graph interrupts again — an infinite loop with no error.
# (`Command(resume=None)` is worse: UnboundLocalError inside the Pregel loop.)
# So a caller with nothing to send must send this sentinel instead, and it is
# stripped here before anything downstream sees it.
SKIP_SENTINEL = {"__skipped__": "1"}


def _strip_sentinels(answers: Any) -> dict[str, str]:
    if not isinstance(answers, dict):
        return {}
    return {k: v for k, v in answers.items() if not str(k).startswith("__")}


def human_review(state: AgentState) -> dict[str, Any]:
    payload = {
        "kind": "gap_fill",
        "run_id": state.get("run_id"),
        "round": state.get("review_rounds", 0) + 1,
        "gaps": state.get("gaps", []),
    }
    answers = interrupt(payload)
    return {"human_input": _strip_sentinels(answers)}
