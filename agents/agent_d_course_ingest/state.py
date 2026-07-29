"""Graph state for the course ingestion cycle. No checkpointer (no interrupt),
so state carries live AdapterResult/RawCourse objects; only fan-in fields have
reducers."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

STATE_VERSION = "itqan.agent_d_state/1.0"


class CourseIngestState(TypedDict, total=False):
    run_id: str
    output_dir: str
    source_names: list[str]
    limit: int | None
    dry_run: bool

    planned_sources: list[str]
    scraped: Annotated[list[dict[str, Any]], operator.add]

    ingest_summary: dict[str, Any]
    # Per-source batch failures (one bad course no longer costs a source's
    # batch, and a lost batch is reported rather than silently absent).
    ingest_errors: list[str]
    # {"aged": [...], "not_aged": [...]} — which sources were trustworthy enough
    # a census to age their unseen inventory this cycle, and which were not.
    ageing: dict[str, Any]
    # {attempted, observed, failed, unparsed, remaining_before} — the
    # backlog a backfill leaves behind, draining a budget at a time.
    enrichment_backlog: dict[str, Any]
    staleness_summary: dict[str, Any]
    aggregation_summary: dict[str, Any]
    source_health: list[dict[str, Any]]
    run_log: dict[str, Any]
    run_log_path: str

    partial_cycle: bool
    warnings: Annotated[list[str], operator.add]
