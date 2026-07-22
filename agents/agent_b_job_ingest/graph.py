"""Graph assembly for the ingestion cycle.

    START
      -> plan_sources          validate config, choose which sources run
      -> Send(scrape, source)  ONE concurrent branch per source (fetch only)
      -> ingest                fan-in: age healthy inventory, then run the
                               pipeline per source batch in its own transaction
      -> staleness             mark stale (3 missed cycles), prune (60 days)
      -> aggregate             recompute skill_demand_stats
      -> runlog                record source health, write the cycle's JSON log
      -> END

No checkpointer, unlike Agent A: there is no ``interrupt()`` here, so state is
never serialized between supersteps and can carry live AdapterResult/RawPosting
objects. The concurrency that matters — fetching several sources at once — comes
from the ``Send`` fan-out, which LangGraph runs in parallel and barriers at the
fan-in before ``ingest``.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import (
    GraphDeps,
    fan_out_to_scrape,
    make_aggregate,
    make_ingest,
    make_plan_sources,
    make_runlog,
    make_scrape,
    make_staleness,
)
from .state import IngestState


def build_ingest_graph(deps: GraphDeps):
    builder = StateGraph(IngestState)

    builder.add_node("plan_sources", make_plan_sources(deps))
    builder.add_node("scrape", make_scrape(deps))
    builder.add_node("ingest", make_ingest(deps))
    builder.add_node("staleness", make_staleness(deps))
    builder.add_node("aggregate", make_aggregate(deps))
    builder.add_node("runlog", make_runlog(deps))

    builder.add_edge(START, "plan_sources")
    # Either a Send per source, or a direct hop to ingest when nothing is
    # planned (inventory still ages on an empty day).
    builder.add_conditional_edges("plan_sources", fan_out_to_scrape, ["scrape", "ingest"])
    builder.add_edge("scrape", "ingest")
    builder.add_edge("ingest", "staleness")
    builder.add_edge("staleness", "aggregate")
    builder.add_edge("aggregate", "runlog")
    builder.add_edge("runlog", END)

    return builder.compile()
