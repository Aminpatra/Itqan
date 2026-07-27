"""Graph nodes — thin wrappers that sequence the work built in earlier phases.

The logic lives elsewhere (``pipeline.py``, ``aggregate.py``, the store); these
nodes only decide ORDER and TRANSACTION BOUNDARIES, which is where the cycle's
operational correctness lives:

  * scrape branches touch no database — they fan out concurrently and only fetch,
    so a scraping bug cannot reach a write and the branches cannot deadlock on
    the connection.
  * every write stage is its OWN transaction (per source batch, then staleness,
    aggregation, health each separately). A source blocked half-way commits what
    it got; nothing rolls back three healthy sources' work because a fourth 429'd.
  * ``age_missed`` runs only for HEALTHY sources and before the pipeline resets
    the seen ones — a partially-fetched source must never age its inventory, or a
    transient block would push its still-live jobs toward deletion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from langgraph.types import Send

from shared.config import Config

from .aggregate import recompute_stats
from .pipeline import IngestPipeline, IngestSummary
from .sources.base import AdapterResult, SourceAdapter
from .sources.config import DEFAULT_SOURCES, SourceConfig, select_sources
from .sources.factory import build_adapter
from .state import IngestState

# A source is aged/counted as healthy only if it fetched cleanly. `result.ok`
# is False for a partial fetch even with postings in hand — that is the whole
# reason the flag exists.
LONG_DEGRADED_DAYS = 14


@dataclass
class GraphDeps:
    """Everything the nodes need, injected once so the graph itself is pure wiring."""

    config: Config
    store: Any
    extractor: Any = None
    adjudicator: Any = None
    embedder: Any = None
    root_fetcher: Any = None
    model_name: str = "unknown"
    source_configs: tuple[SourceConfig, ...] = DEFAULT_SOURCES
    # Overridable so tests can supply fake adapters without a network. Real runs
    # use the factory, which builds the httpx-backed adapters.
    adapter_for: Callable[[SourceConfig, Config], SourceAdapter] | None = None

    def build_adapter(self, cfg: SourceConfig) -> SourceAdapter:
        if self.adapter_for is not None:
            return self.adapter_for(cfg, self.config)
        return build_adapter(cfg, config=self.config)

    def pipeline(self) -> IngestPipeline:
        return IngestPipeline(
            store=self.store,
            extractor=self.extractor,
            adjudicator=self.adjudicator,
            embedder=self.embedder,
            config=self.config,
            model_name=self.model_name,
            root_fetcher=self.root_fetcher,
        )

    def config_by_name(self, name: str) -> SourceConfig:
        for cfg in self.source_configs:
            if cfg.name == name:
                return cfg
        raise KeyError(name)


# ---------------------------------------------------------------------------
def make_plan_sources(deps: GraphDeps) -> Callable[[IngestState], dict]:
    def plan_sources(state: IngestState) -> dict:
        names = state.get("source_names") or None
        chosen = select_sources(
            ",".join(names) if names else None,
            sources=deps.source_configs,
            dry_run=state.get("dry_run", False),
        )
        return {"planned_sources": [c.name for c in chosen]}

    return plan_sources


def fan_out_to_scrape(state: IngestState):
    """One concurrent scrape branch per planned source.

    Returns node names, not just Sends: with no planned sources the cycle still
    has to run its staleness/aggregation tail (inventory ages even on a day when
    nothing was fetched), so it routes straight to ingest with an empty batch.
    """
    planned = state.get("planned_sources", [])
    if not planned:
        return ["ingest"]
    return [
        Send("scrape", {"scrape_source": name, "limit": state.get("limit"),
                        "dry_run": state.get("dry_run", False)})
        for name in planned
    ]


def make_scrape(deps: GraphDeps) -> Callable[[dict], dict]:
    def scrape(state: dict) -> dict:
        name = state["scrape_source"]
        cfg = deps.config_by_name(name)
        adapter = deps.build_adapter(cfg)
        result = adapter.fetch(limit=state.get("limit"))
        return {
            "scraped": [{"source": name, "source_type": cfg.source_type, "result": result}],
        }

    return scrape


def make_ingest(deps: GraphDeps) -> Callable[[IngestState], dict]:
    def ingest(state: IngestState) -> dict:
        outcomes = state.get("scraped", [])
        healthy = [o["source"] for o in outcomes if o["result"].ok]
        partial = any(not o["result"].ok for o in outcomes)
        store = deps.store

        # Age the inventory of cleanly-fetched sources BEFORE the pipeline resets
        # the ones it sees. A partial/blocked source is absent from `healthy`, so
        # its inventory is left untouched rather than aged toward deletion.
        with store.transaction():
            store.age_missed(healthy)

        # Link targets before linkers: a Telegram post resolves to a blog post by
        # URL, so the blog batch must commit first. Sorting telegram last makes
        # the deterministic link-dedup path find its target already persisted.
        ordered = sorted(outcomes, key=lambda o: o.get("source_type") == "telegram")

        pipe = deps.pipeline()
        total = IngestSummary()
        for outcome in ordered:
            postings = list(outcome["result"].postings)
            if not postings:
                continue
            # One transaction per source batch.
            with store.transaction():
                total.merge(pipe.run(postings))

        return {"ingest_summary": total.as_dict(), "partial_cycle": partial}

    return ingest


def make_staleness(deps: GraphDeps) -> Callable[[IngestState], dict]:
    def staleness(state: IngestState) -> dict:
        store, cfg = deps.store, deps.config
        with store.transaction():
            marked = store.mark_stale(threshold=cfg.stale_after_cycles)
            pruned = store.prune(older_than_days=cfg.prune_after_days)
        return {"staleness_summary": {"marked_stale": marked, "pruned": pruned}}

    return staleness


def make_aggregate(deps: GraphDeps) -> Callable[[IngestState], dict]:
    def aggregate(state: IngestState) -> dict:
        from .esco_map import map_new_skills

        # Mapping runs FIRST, in its own transaction, so the recompute's
        # LEFT JOIN sees this cycle's new skills already resolved. After warm-up
        # only genuinely new phrases pay an embedding call. A no-op when the
        # taxonomy has never been synced.
        with deps.store.transaction():
            esco = map_new_skills(deps.store, deps.embedder, deps.config)

        with deps.store.transaction():
            summary = recompute_stats(deps.store, deps.config)

        return {"aggregation_summary": {**summary.as_dict(), "esco": esco.as_dict()}}

    return aggregate


def make_runlog(deps: GraphDeps) -> Callable[[IngestState], dict]:
    def runlog(state: IngestState) -> dict:
        store, cfg = deps.store, deps.config
        outcomes = state.get("scraped", [])

        health_rows: list[dict[str, Any]] = []
        with store.transaction():
            for outcome in outcomes:
                result: AdapterResult = outcome["result"]
                # A partial fetch is a failure for health purposes: the source
                # did not deliver its inventory, even if it delivered some of it.
                success = result.error is None and not result.partial
                row = store.record_source_health(
                    outcome["source"],
                    success=success,
                    error=result.error,
                    degraded_after=cfg.degraded_after_cycles,
                )
                health_rows.append(row)

        warnings: list[str] = []
        now = datetime.now(timezone.utc)
        for row in health_rows:
            since = row.get("degraded_since")
            if since is not None and (now - since).days > LONG_DEGRADED_DAYS:
                warnings.append(
                    f"source {row['source']} has been degraded since {since.date()} "
                    f"({(now - since).days} days) — its inventory will never expire "
                    "while degraded; decommission with --purge-source if it is dead"
                )

        run_log = _assemble_run_log(state, outcomes, health_rows)
        path = _write_run_log(state, cfg, run_log)
        return {
            "source_health": health_rows,
            "run_log": run_log,
            "run_log_path": path,
            "warnings": warnings,
        }

    return runlog


# ---------------------------------------------------------------------------
def _assemble_run_log(state, outcomes, health_rows) -> dict[str, Any]:
    ingest = state.get("ingest_summary", {})
    agg = state.get("aggregation_summary", {})

    per_source = []
    for outcome in outcomes:
        r: AdapterResult = outcome["result"]
        per_source.append({
            "source": outcome["source"],
            "postings": len(r.postings),
            "skipped": r.skipped,
            "pages_fetched": r.pages_fetched,
            "bytes_fetched": r.bytes_fetched,
            "ok": r.ok,
            "partial": r.partial,
            "blocked_after_n": r.blocked_after_n,
            "error": r.error,
        })

    degraded = [h["source"] for h in health_rows if h.get("degraded_since") is not None]

    return {
        "run_id": state.get("run_id"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "planned_sources": state.get("planned_sources", []),
        "partial_cycle": state.get("partial_cycle", False),
        "sources": per_source,
        "ingest": ingest,
        "staleness": state.get("staleness_summary", {}),
        "aggregation": agg,
        # Operational signals the plan calls for, mapped from the counters we
        # actually produce. Named so a reader knows what each excluded item is.
        "signals": {
            "embeddings_skipped_link_dup": ingest.get("link_duplicates", 0),
            "embeddings_skipped_rejected": ingest.get("rejected", 0),
            "cross_group_duplicate_candidates": ingest.get("needs_review", 0),
            "sectors_with_zero_current_postings": agg.get("sectors_zeroed", 0),
            "postings_excluded_null_country": agg.get("postings_excluded_null_country", 0),
            "degraded_sources": degraded,
            # Deferred: label-vs-extraction agreement needs the baseline measured
            # first (labels are telemetry-only until then), so nothing is emitted
            # rather than a fabricated zero dressed up as a real measurement.
            "label_disagreements": None,
        },
    }


def _write_run_log(state, config: Config, run_log: dict) -> str:
    run_id = state.get("run_id") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    base = Path(state.get("output_dir") or config.output_dir) / run_id
    base.mkdir(parents=True, exist_ok=True)
    path = base / "ingest_cycle.json"
    # default=str so datetimes and any stray non-JSON value serialise rather than
    # crashing the run log — the log must always be written, even on a bad cycle.
    path.write_text(json.dumps(run_log, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    return str(path)
