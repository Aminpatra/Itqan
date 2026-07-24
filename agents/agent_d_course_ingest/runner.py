"""Run one course ingestion cycle: build deps, take the lock, invoke the graph.

Separated from cli.py so a test can drive a whole cycle through the same path
the real CLI uses. The cycle lock (dedicated connection, Agent-D key) stops two
overlapping cycles double-incrementing missed_cycles.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from shared.config import Config

from .db import CourseStore, LockNotAcquired, apply_migrations
from .graph import build_course_ingest_graph
from .nodes import GraphDeps


@dataclass
class CycleResult:
    ran: bool
    exit_code: int
    state: dict[str, Any] | None = None


def _build_deps(config: Config, store: CourseStore, *, fake_llm: bool, no_embed: bool) -> GraphDeps:
    from .prompts.extraction import EXTRACTION_PROMPT
    from .schemas import CourseExtraction

    if fake_llm:
        from shared.llm import structured
        from tests.fake_embedder import FakeEmbedder
        from tests.fake_llm import FakeStructuredLLM

        llm = FakeStructuredLLM()
        embedder = None if no_embed else FakeEmbedder()
        model_name = "fake-llm"
    else:
        from shared.embeddings import build_embedder
        from shared.llm import build_llm, structured

        llm = build_llm(config)
        embedder = None if no_embed else build_embedder(config)
        model_name = config.model

    return GraphDeps(
        config=config, store=store,
        extractor=EXTRACTION_PROMPT | structured(llm, CourseExtraction),
        embedder=embedder, model_name=model_name,
    )


def run_cycle(
    config: Config,
    *,
    source_names: list[str] | None = None,
    limit: int | None = None,
    fake_llm: bool = False,
    no_embed: bool = False,
    run_id: str | None = None,
) -> CycleResult:
    if not fake_llm:
        config.require_api_key()
    config.require_identified_user_agent()
    dsn = config.require_database_url()
    apply_migrations(dsn)

    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:6]

    with CourseStore(dsn) as store:
        try:
            with store.cycle_lock():
                deps = _build_deps(config, store, fake_llm=fake_llm, no_embed=no_embed)
                state = build_course_ingest_graph(deps).invoke({
                    "run_id": run_id, "output_dir": str(config.output_dir),
                    "source_names": source_names or [], "limit": limit, "dry_run": False,
                })
        except LockNotAcquired:
            return CycleResult(ran=False, exit_code=0)

    exit_code = 1 if state.get("partial_cycle") else 0
    return CycleResult(ran=True, exit_code=exit_code, state=state)
