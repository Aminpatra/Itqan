"""Course ingestion cycle graph. Same shape as Agent B, no checkpointer."""

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
from .state import CourseIngestState


def build_course_ingest_graph(deps: GraphDeps):
    b = StateGraph(CourseIngestState)
    b.add_node("plan_sources", make_plan_sources(deps))
    b.add_node("scrape", make_scrape(deps))
    b.add_node("ingest", make_ingest(deps))
    b.add_node("staleness", make_staleness(deps))
    b.add_node("aggregate", make_aggregate(deps))
    b.add_node("runlog", make_runlog(deps))

    b.add_edge(START, "plan_sources")
    b.add_conditional_edges("plan_sources", fan_out_to_scrape, ["scrape", "ingest"])
    b.add_edge("scrape", "ingest")
    b.add_edge("ingest", "staleness")
    b.add_edge("staleness", "aggregate")
    b.add_edge("aggregate", "runlog")
    b.add_edge("runlog", END)
    return b.compile()
