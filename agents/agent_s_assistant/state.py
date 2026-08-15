"""Graph state for Agent S.

Linear, no checkpointer, no interrupt: one question in, one answer out. The
graph exists for uniformity with Agents A-E — one runtime, one testing idiom —
rather than for control flow it strictly needs.

Note what is NOT in here: no user id, no quota counts, no session. Those live in
the API layer, which has already finished with them before this graph is built.
Agent S is handed a fact sheet and a question; it cannot look anything up.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict

STATE_VERSION = "itqan.agent_s_state/1.0"


class AssistantState(TypedDict, total=False):
    # ---- inputs ----------------------------------------------------------
    question: str
    # Already built by the caller from ONE user's mapped results. The graph
    # never assembles this itself, because assembling it is where a scoping
    # mistake would be made.
    fact_sheet: str
    history: list[dict[str, Any]]
    has_results: bool

    # ---- answer ----------------------------------------------------------
    answer: str
    # 'model' when the model's text was published, 'template' when verification
    # rejected it. Recorded on the row so a silent drift toward the fallback is
    # countable rather than invisible.
    answer_source: str
    # The model was absent or raised — an infrastructure failure, as distinct
    # from a model that answered and was refused by `verify`. The API layer
    # refunds the first and charges for the second, and the distinction is
    # deliberate: refunding a rejected answer would let someone chat for free by
    # steering the model into rejections, while charging for an outage takes one
    # of somebody's ten daily messages for a sentence we generated ourselves.
    model_failed: bool
    # The model's suggestion only. Nothing acts on this inside the graph.
    proposed_rerun: bool
    rerun_reason: Optional[str]
    out_of_scope: bool

    # Why an answer was rejected, when it was. Surfaced in the run log, not to
    # the user — they get the deterministic answer instead.
    warnings: Annotated[list[str], operator.add]
