"""Run a compiled graph and report each node as it finishes.

Why this exists: the web app's progress bar is driven by real phase completion and
never by a clock, which is the honest choice — a timer-driven bar reaches 90% and
stalls while the user waits, and a hung run looks identical to a slow one. The
cost of that honesty was granularity: with three checkpoints the bar jumped
0.15 -> 0.55 -> 0.80 and sat still in between.

The fix is more real checkpoints, not interpolation. Every agent is a LangGraph
`StateGraph`, and LangGraph will already tell us when each node finishes — so a
run reports 10-11 genuine steps instead of 2, and the bar moves because work
actually happened.

`stream_mode=["updates", "values"]` is what makes this uniform across agents:
`updates` carries the node names, `values` carries the full state after each step,
so the last one is the final state. That matters because Agent A compiles with a
checkpointer (it needs one for `interrupt()`) while Agents C and E deliberately do
not, so `get_state()` is available on one and not the others. This works on all
three.
"""

from __future__ import annotations

from typing import Any, Callable, Optional


def run_reporting(app: Any, initial: Any, *, config: Optional[dict[str, Any]] = None,
                  on_node: Optional[Callable[[str], None]] = None) -> dict[str, Any]:
    """Invoke `app`, calling `on_node(name)` as each node completes.

    Returns the final state, exactly as `app.invoke()` would. With no `on_node`
    this is `invoke` — the streaming path is not taken at all, so a caller that
    does not want progress pays nothing and behaves identically.

    A failing `on_node` must never sink the run: reporting progress is a courtesy
    to the UI, and losing an extraction because a status write failed would be a
    poor trade. Exceptions from the callback are swallowed deliberately.
    """
    if on_node is None:
        return app.invoke(initial, config=config) if config else app.invoke(initial)

    kwargs: dict[str, Any] = {"stream_mode": ["updates", "values"]}
    if config:
        kwargs["config"] = config

    final: dict[str, Any] = {}
    for mode, chunk in app.stream(initial, **kwargs):
        if mode == "values":
            # Each `values` chunk is the whole state after that step, so the last
            # one seen is the result. Kept rather than accumulated: merging deltas
            # by hand would have to reimplement every reducer in the state.
            if isinstance(chunk, dict):
                final = chunk
            continue
        for node in chunk:
            # `__interrupt__` and friends are control signals, not work.
            if node.startswith("__"):
                continue
            try:
                on_node(node)
            except Exception:            # noqa: BLE001 - see docstring
                pass
    return final
