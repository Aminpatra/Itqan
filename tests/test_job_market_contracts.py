"""Offline checks on the job-market read surface: contract shape, the no-LLM
guarantee, and the single-sourcing of the eligibility predicate."""

from __future__ import annotations

import ast
from pathlib import Path

from shared.contracts import JobPostingExport, SkillDemandStatRow

MODULE = Path(__file__).parent.parent / "shared" / "job_market.py"


def test_posting_export_round_trips_a_hand_built_row():
    posting = JobPostingExport(
        posting_id="abc", source="el7far", source_group="g", source_type="blogger_feed",
        source_url="https://e.test/a", title="Role", raw_description="desc",
        sector="2", required_skills=["python"], country="OM",
        first_seen_at="2026-07-22T00:00:00+00:00", last_seen_at="2026-07-22T00:00:00+00:00",
        similarity=0.91,
    )
    again = JobPostingExport.model_validate(posting.model_dump())
    assert again.similarity == 0.91 and again.listing_intent == "vacancy"


def test_stat_row_round_trips():
    row = SkillDemandStatRow(
        sector="2", skill="Python", skill_key="python", esco_code=None,
        window_start="2026-06-22", window_end="2026-07-22",
        frequency_count=5, computed_at="2026-07-22T00:00:00+00:00",
    )
    assert SkillDemandStatRow.model_validate(row.model_dump()).frequency_count == 5


def test_the_read_surface_imports_no_llm():
    """A read/serialize layer, deterministic like ingest and aggregate. The
    embedder used by map_skills_to_esco is INJECTED by the caller — this module
    must not be able to construct one."""
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        for name in names:
            assert "langchain" not in name and "openai" not in name, (
                f"job_market.py imports {name} — the read surface must stay LLM-free"
            )


def test_the_aggregation_embeds_the_shared_predicate():
    """Single-sourcing is real, not aspirational: the pipeline's eligibility CTE
    is built FROM the shared constant, so 'what we count' and 'what we serve'
    cannot drift apart."""
    from agents.agent_b_job_ingest.aggregate import _ELIGIBLE_CTE, _NULL_COUNTRY_SQL
    from shared.job_market import AGGREGABLE_POSTING_PREDICATE

    assert AGGREGABLE_POSTING_PREDICATE in _ELIGIBLE_CTE
    assert AGGREGABLE_POSTING_PREDICATE in _NULL_COUNTRY_SQL
