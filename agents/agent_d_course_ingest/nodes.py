"""Graph nodes — order and transaction boundaries for the course cycle.

Mirrors Agent B's nodes exactly: scrape branches touch no database and fan out
concurrently; every write stage is its own transaction (per source batch, then
staleness, aggregation, health); ``age_missed`` runs only for HEALTHY sources
before the pipeline resets the seen ones, so a partial fetch never ages
inventory it did not re-fetch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from langgraph.types import Send

from shared.config import Config

from .aggregate import recompute_supply
from .esco_map import map_new_course_skills
from .pipeline import CoursePipeline, IngestSummary
from .sources.base import AdapterResult, CourseAdapter
from .sources.config import DEFAULT_SOURCES, CourseSourceConfig, select_sources
from .sources.factory import build_adapter
from .state import CourseIngestState

LONG_DEGRADED_DAYS = 14


@dataclass
class GraphDeps:
    config: Config
    store: Any
    extractor: Any = None
    embedder: Any = None
    model_name: str = "unknown"
    source_configs: tuple[CourseSourceConfig, ...] = DEFAULT_SOURCES
    adapter_for: Callable[[CourseSourceConfig, Config], CourseAdapter] | None = None

    def build_adapter(self, cfg: CourseSourceConfig) -> CourseAdapter:
        if self.adapter_for is not None:
            return self.adapter_for(cfg, self.config)
        return build_adapter(cfg, config=self.config)

    def pipeline(self) -> CoursePipeline:
        return CoursePipeline(store=self.store, extractor=self.extractor,
                              embedder=self.embedder, config=self.config,
                              model_name=self.model_name)

    def config_by_name(self, name: str) -> CourseSourceConfig:
        for cfg in self.source_configs:
            if cfg.name == name:
                return cfg
        raise KeyError(name)


def make_plan_sources(deps: GraphDeps) -> Callable[[CourseIngestState], dict]:
    def plan_sources(state: CourseIngestState) -> dict:
        names = state.get("source_names") or None
        chosen = select_sources(",".join(names) if names else None,
                                sources=deps.source_configs, dry_run=state.get("dry_run", False))
        return {"planned_sources": [c.name for c in chosen]}
    return plan_sources


def fan_out_to_scrape(state: CourseIngestState):
    planned = state.get("planned_sources", [])
    if not planned:
        return ["ingest"]
    return [Send("scrape", {"scrape_source": n, "limit": state.get("limit"),
                            "dry_run": state.get("dry_run", False)}) for n in planned]


def make_scrape(deps: GraphDeps) -> Callable[[dict], dict]:
    def scrape(state: dict) -> dict:
        name = state["scrape_source"]
        cfg = deps.config_by_name(name)
        adapter = deps.build_adapter(cfg)
        result = adapter.fetch(limit=state.get("limit"))
        return {"scraped": [{"source": name, "source_type": cfg.source_type, "result": result}]}
    return scrape


def make_ingest(deps: GraphDeps) -> Callable[[CourseIngestState], dict]:
    def ingest(state: CourseIngestState) -> dict:
        outcomes = state.get("scraped", [])
        healthy = [o["source"] for o in outcomes if o["result"].ok]
        partial = any(not o["result"].ok for o in outcomes)
        store = deps.store

        with store.transaction():
            store.age_missed(healthy)

        pipe = deps.pipeline()
        total = IngestSummary()
        for outcome in outcomes:
            courses = list(outcome["result"].courses)
            if not courses:
                continue
            with store.transaction():
                total.merge(pipe.run(courses))
        return {"ingest_summary": total.as_dict(), "partial_cycle": partial}
    return ingest


def make_staleness(deps: GraphDeps) -> Callable[[CourseIngestState], dict]:
    def staleness(state: CourseIngestState) -> dict:
        store, cfg = deps.store, deps.config
        with store.transaction():
            marked = store.mark_stale(threshold=cfg.course_stale_after_cycles)
            pruned = store.prune(older_than_days=cfg.course_prune_after_days)
        return {"staleness_summary": {"marked_stale": marked, "pruned": pruned}}
    return staleness


def make_aggregate(deps: GraphDeps) -> Callable[[CourseIngestState], dict]:
    def aggregate(state: CourseIngestState) -> dict:
        with deps.store.transaction():
            esco = map_new_course_skills(deps.store, deps.embedder, deps.config)
        with deps.store.transaction():
            summary = recompute_supply(deps.store, deps.config)
        return {"aggregation_summary": {**summary.as_dict(), "esco": esco.as_dict()}}
    return aggregate


def make_runlog(deps: GraphDeps) -> Callable[[CourseIngestState], dict]:
    def runlog(state: CourseIngestState) -> dict:
        store, cfg = deps.store, deps.config
        outcomes = state.get("scraped", [])
        health_rows: list[dict[str, Any]] = []
        with store.transaction():
            for o in outcomes:
                r: AdapterResult = o["result"]
                success = r.error is None and not r.partial
                health_rows.append(store.record_source_health(
                    o["source"], success=success, error=r.error,
                    degraded_after=cfg.degraded_after_cycles))

        warnings: list[str] = []
        now = datetime.now(timezone.utc)
        for row in health_rows:
            since = row.get("degraded_since")
            if since is not None and (now - since).days > LONG_DEGRADED_DAYS:
                warnings.append(f"source {row['source']} degraded since {since.date()}")

        run_log = _assemble(state, outcomes, health_rows)
        path = _write(state, cfg, run_log)
        return {"source_health": health_rows, "run_log": run_log,
                "run_log_path": path, "warnings": warnings}
    return runlog


def _assemble(state, outcomes, health_rows) -> dict[str, Any]:
    ingest = state.get("ingest_summary", {})
    agg = state.get("aggregation_summary", {})
    per_source = [{
        "source": o["source"], "courses": len(o["result"].courses),
        "skipped": o["result"].skipped, "pages_fetched": o["result"].pages_fetched,
        "ok": o["result"].ok, "partial": o["result"].partial, "error": o["result"].error,
    } for o in outcomes]
    return {
        "run_id": state.get("run_id"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "planned_sources": state.get("planned_sources", []),
        "partial_cycle": state.get("partial_cycle", False),
        "sources": per_source,
        "ingest": ingest,
        "staleness": state.get("staleness_summary", {}),
        "aggregation": agg,
        "signals": {
            "embeddings_skipped_rejected": ingest.get("rejected", 0),
            "cross_group_duplicate_candidates": ingest.get("needs_review", 0),
            "skills_mapped_this_cycle": (agg.get("esco") or {}).get("exact", 0)
                                        + (agg.get("esco") or {}).get("alt_label", 0)
                                        + (agg.get("esco") or {}).get("embedding", 0),
            "skills_unmapped": (agg.get("esco") or {}).get("unmapped", 0),
            "degraded_sources": [h["source"] for h in health_rows
                                 if h.get("degraded_since") is not None],
        },
    }


def _write(state, config: Config, run_log: dict) -> str:
    run_id = state.get("run_id") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    base = Path(state.get("output_dir") or config.output_dir) / run_id
    base.mkdir(parents=True, exist_ok=True)
    path = base / "course_cycle.json"
    path.write_text(json.dumps(run_log, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    return str(path)
