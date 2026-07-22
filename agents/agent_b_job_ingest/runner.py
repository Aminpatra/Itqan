"""Run one ingestion cycle: build the real dependencies, take the lock, invoke
the graph.

Separated from ``cli.py`` so the argument parsing and the orchestration are
testable apart from each other, and so a test can drive a whole cycle with fake
LLM/embedder through exactly the path the real CLI uses.

The cycle lock is the one piece of concurrency safety that lives here: two
overlapping cycles would double-increment ``missed_cycles`` and push every
posting to stale a full cycle early. It is held on a DEDICATED connection for the
whole cycle (see ``JobStore.cycle_lock``); failing to acquire it is not an error
— it means another cycle is already running, and the right response is to exit 0
and let it finish.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from shared.config import Config

from .db import JobStore, LockNotAcquired, apply_migrations
from .graph import build_ingest_graph
from .nodes import GraphDeps


@dataclass
class CycleResult:
    ran: bool                       # False when the lock was already held
    exit_code: int
    state: dict[str, Any] | None = None


def _build_deps(config: Config, store: JobStore, *, fake_llm: bool, no_embed: bool) -> GraphDeps:
    """Assemble the graph's dependencies.

    ``fake_llm`` swaps in the offline test doubles so a full cycle can run with
    no API key and no spend — the same fakes the test suite uses, reached through
    the real CLI path. Without it, the real ChatOpenAI and OpenAIEmbeddings are
    built, and the model id is recorded on every row as ``extraction_model``.
    """
    from .prompts.extraction import EXTRACTION_PROMPT
    from .prompts.legitimacy import LEGITIMACY_PROMPT
    from .schemas import JobExtraction, LegitimacyVerdict

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
        config=config,
        store=store,
        extractor=EXTRACTION_PROMPT | structured(llm, JobExtraction),
        adjudicator=LEGITIMACY_PROMPT | structured(llm, LegitimacyVerdict),
        embedder=embedder,
        model_name=model_name,
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
    """Apply migrations, take the cycle lock, run the graph once.

    Migrations are applied at the start: they are idempotent (an already-applied
    migration is skipped, an edited one is an error), so a scheduled job stays
    current across deploys without a separate manual step, while an accidental
    edit still fails loudly.
    """
    if not fake_llm:
        config.require_api_key()
    config.require_identified_user_agent()
    dsn = config.require_database_url()

    apply_migrations(dsn)

    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:6]

    with JobStore(dsn) as store:
        try:
            with store.cycle_lock():
                deps = _build_deps(config, store, fake_llm=fake_llm, no_embed=no_embed)
                graph = build_ingest_graph(deps)
                state = graph.invoke({
                    "run_id": run_id,
                    "output_dir": str(config.output_dir),
                    "source_names": source_names or [],
                    "limit": limit,
                    "dry_run": False,
                })
        except LockNotAcquired:
            return CycleResult(ran=False, exit_code=0)

    # A partial cycle exits non-zero so a scheduler's only signal — the exit
    # code — reflects that inventory was fetched incompletely.
    exit_code = 1 if state.get("partial_cycle") else 0
    return CycleResult(ran=True, exit_code=exit_code, state=state)
