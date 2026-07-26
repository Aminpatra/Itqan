"""Graph assembly.

    START
      -> ingest            classify both documents from magic bytes
      -> extract_text      PDF text layer or PaddleOCR; writes OCR JSON
      -> llm_extract_cv
           |--[transcript]--> llm_extract_transcript --|
           |--[none]-----------------------------------+--> verify_grounding
      -> assess_gaps
           |--[gaps and rounds left]--> human_review  <== interrupt() fires here
           |                              -> validate_human_input
           |                                   -> assess_gaps   (bounded loop)
           |--[no gaps / rounds spent]--> research_curriculum
      -> research_curriculum   what the courses/certs typically teach
      -> judge_skills          claims cross-checked against that evidence
      -> derive_coursework_skills  promote unclaimed skills the coursework teaches
      -> summarize             few-shot
      -> persist
      -> END

The checkpointer is required for ``interrupt()`` to work at all, and every
invocation must carry ``configurable.thread_id``. ``InMemorySaver`` is right for a
single-process CLI where the pause and the resume happen in one run; swap it for
``langgraph-checkpoint-sqlite`` if a run ever needs to survive process exit.
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from shared.config import Config

from .nodes import (
    human_review,
    make_assess_gaps_node,
    make_derive_coursework_skills_node,
    make_extract_text_node,
    make_ingest_node,
    make_judge_skills_node,
    make_llm_extract_cv_node,
    make_llm_extract_transcript_node,
    make_persist_node,
    make_research_curriculum_node,
    make_route_gaps,
    make_summarize_node,
    make_validate_human_node,
    make_verify_node,
    route_after_cv,
)
from .state import AgentState


def build_graph(llm: Any, config: Config | None = None, *, checkpointer: Any = None):
    """Wire and compile Agent A.

    Args:
        llm: any LangChain chat model supporting ``with_structured_output``.
        config: threshold overrides; defaults suit English CVs.
        checkpointer: defaults to ``InMemorySaver()``. Required for HITL.
    """
    config = config or Config()
    builder = StateGraph(AgentState)

    builder.add_node("ingest", make_ingest_node(config))
    builder.add_node("extract_text", make_extract_text_node(config))
    builder.add_node("llm_extract_cv", make_llm_extract_cv_node(llm, config))
    builder.add_node("llm_extract_transcript", make_llm_extract_transcript_node(llm, config))
    builder.add_node("verify_grounding", make_verify_node(llm, config))
    builder.add_node("assess_gaps", make_assess_gaps_node(config))
    builder.add_node("human_review", human_review)
    builder.add_node("validate_human_input", make_validate_human_node(llm, config))
    builder.add_node("research_curriculum", make_research_curriculum_node(llm, config))
    builder.add_node("judge_skills", make_judge_skills_node(llm, config))
    builder.add_node("derive_coursework_skills", make_derive_coursework_skills_node(llm, config))
    builder.add_node("summarize", make_summarize_node(llm, config))
    builder.add_node("persist", make_persist_node(config))

    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "extract_text")
    builder.add_edge("extract_text", "llm_extract_cv")

    builder.add_conditional_edges(
        "llm_extract_cv",
        route_after_cv,
        {
            "llm_extract_transcript": "llm_extract_transcript",
            "verify_grounding": "verify_grounding",
        },
    )
    builder.add_edge("llm_extract_transcript", "verify_grounding")
    builder.add_edge("verify_grounding", "assess_gaps")

    builder.add_conditional_edges(
        "assess_gaps",
        make_route_gaps(config),
        {
            "human_review": "human_review",
            # Curriculum research sits after the HITL loop deliberately: manual
            # entry can add courses or certificates, and researching before that
            # would miss them.
            "judge_skills": "research_curriculum",
        },
    )
    builder.add_edge("human_review", "validate_human_input")
    # Loop back: validate_human_input increments review_rounds, so route_gaps
    # eventually stops choosing human_review and the loop terminates.
    builder.add_edge("validate_human_input", "assess_gaps")

    builder.add_edge("research_curriculum", "judge_skills")
    # Derivation runs after judging so it can dedup against the accepted/rejected
    # claims, and reuses the same judge to filter its own candidates.
    builder.add_edge("judge_skills", "derive_coursework_skills")
    builder.add_edge("derive_coursework_skills", "summarize")
    builder.add_edge("summarize", "persist")
    builder.add_edge("persist", END)

    return builder.compile(checkpointer=checkpointer or InMemorySaver())
