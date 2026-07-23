"""Graph assembly for Agent C.

    START
      -> build_query_embedding    profile -> posting-shaped essence -> vector
      -> retrieve_postings        export_for_agent_c; <5 usable => fallback
      -> map_candidate_skills     read-only ESCO mapping, same tiers as Agent B
      -> gap_analysis             arithmetic: matched / missing / possible_match
      -> persist                  skill_gap.json
      -> END

Linear and deterministic: no interrupt, no checkpointer, no LLM. The graph
exists for uniformity with Agents A and B (one runtime, one testing idiom, one
place to add telemetry) rather than for control flow it strictly needs.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import (
    Deps,
    make_build_query_embedding,
    make_gap_analysis,
    make_map_candidate_skills,
    make_persist,
    make_retrieve_postings,
)
from .state import GapState


def build_gap_graph(deps: Deps):
    builder = StateGraph(GapState)

    builder.add_node("build_query_embedding", make_build_query_embedding(deps))
    builder.add_node("retrieve_postings", make_retrieve_postings(deps))
    builder.add_node("map_candidate_skills", make_map_candidate_skills(deps))
    builder.add_node("gap_analysis", make_gap_analysis(deps))
    builder.add_node("persist", make_persist(deps))

    builder.add_edge(START, "build_query_embedding")
    builder.add_edge("build_query_embedding", "retrieve_postings")
    builder.add_edge("retrieve_postings", "map_candidate_skills")
    builder.add_edge("map_candidate_skills", "gap_analysis")
    builder.add_edge("gap_analysis", "persist")
    builder.add_edge("persist", END)

    return builder.compile()
