"""Graph state for the ingestion cycle.

Unlike Agent A, this graph has no ``interrupt()`` and needs no checkpointer, so
state is never serialized between supersteps — it can hold live ``AdapterResult``
and ``RawPosting`` objects directly rather than dicts.

The only reducers are on the fan-in fields. ``plan_sources`` fans out one
``scrape`` branch per source with ``Send``; those branches run concurrently and
each appends its own result, so ``scraped`` and ``warnings`` accumulate with
``operator.add``. Everything after fan-in is linear and last-write-wins.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from .sources.base import AdapterResult

STATE_VERSION = "itqan.agent_b_state/1.0"


class ScrapeOutcome(TypedDict, total=False):
    """One source's fetch, carried whole from a scrape branch to the ingest node."""

    source: str
    result: AdapterResult


class IngestState(TypedDict, total=False):
    # ---- inputs ----------------------------------------------------------
    run_id: str
    output_dir: str
    source_names: list[str]        # which sources this cycle runs; empty = all enabled
    limit: int | None              # per-source cap, for a bounded test run
    dry_run: bool

    # ---- planning --------------------------------------------------------
    planned_sources: list[str]

    # ---- fan-in (concurrent scrape branches append here) -----------------
    scraped: Annotated[list[ScrapeOutcome], operator.add]

    # ---- linear tail -----------------------------------------------------
    ingest_summary: dict[str, Any]
    staleness_summary: dict[str, Any]
    aggregation_summary: dict[str, Any]
    source_health: list[dict[str, Any]]
    run_log: dict[str, Any]
    run_log_path: str

    # ---- cross-cutting ---------------------------------------------------
    # True when at least one source was blocked or partially fetched; the CLI
    # turns this into a non-zero exit so a scheduler notices.
    partial_cycle: bool
    warnings: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
