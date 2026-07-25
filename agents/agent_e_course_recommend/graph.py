"""Graph assembly for Agent E.

    START
      -> load_missing_skills        read Agent C's aggregate.missing_skill_details
      -> retrieve_candidate_courses courses_for_skills (by esco, exact-key fallback)
      -> greedy_cover_assign        weighted greedy set-cover, deterministic
      -> attach_flags               priority_bucket + course_quality, in code
      -> generate_rationale         the ONLY LLM call, one per recommended course
      -> persist                    course_recommendations.json
      -> END

Linear and deterministic up to the fenced rationale step: no interrupt, no
checkpointer. The graph exists for uniformity with Agents A-D (one runtime, one
testing idiom) rather than for control flow it strictly needs.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import (
    Deps,
    make_attach_flags,
    make_generate_rationale,
    make_greedy_cover_assign,
    make_load_missing_skills,
    make_persist,
    make_retrieve_candidate_courses,
)
from .state import RecommendState


def build_recommend_graph(deps: Deps):
    builder = StateGraph(RecommendState)

    builder.add_node("load_missing_skills", make_load_missing_skills(deps))
    builder.add_node("retrieve_candidate_courses", make_retrieve_candidate_courses(deps))
    builder.add_node("greedy_cover_assign", make_greedy_cover_assign(deps))
    builder.add_node("attach_flags", make_attach_flags(deps))
    builder.add_node("generate_rationale", make_generate_rationale(deps))
    builder.add_node("persist", make_persist(deps))

    builder.add_edge(START, "load_missing_skills")
    builder.add_edge("load_missing_skills", "retrieve_candidate_courses")
    builder.add_edge("retrieve_candidate_courses", "greedy_cover_assign")
    builder.add_edge("greedy_cover_assign", "attach_flags")
    builder.add_edge("attach_flags", "generate_rationale")
    builder.add_edge("generate_rationale", "persist")
    builder.add_edge("persist", END)

    return builder.compile()
