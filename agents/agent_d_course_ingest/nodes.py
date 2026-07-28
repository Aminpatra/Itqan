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
        partial = any(not o["result"].ok for o in outcomes)
        store = deps.store
        pipe = deps.pipeline()
        total = IngestSummary()
        aged: list[str] = []
        not_aged: list[str] = []
        failures: list[str] = []

        for outcome in outcomes:
            source = outcome["source"]
            result = outcome["result"]
            courses = list(result.courses)

            # TWO independent conditions, and both must hold before a course can
            # be aged toward deletion:
            #   census            — can one cycle enumerate this source at all?
            #   may_age_inventory — did THIS fetch actually read it all?
            # An unknown source defaults to census=False: never age what we
            # cannot account for.
            try:
                census = deps.config_by_name(source).census
            except KeyError:
                census = False
            ageable = census and result.may_age_inventory

            # Chunked so a large walk commits as it goes. One transaction for the
            # whole source is right for a 40-posting feed and wrong for a 2,000
            # course backfill: it holds a connection 'idle in transaction' for
            # twenty minutes of LLM calls, blocks vacuum, and loses every course
            # if the last one fails. Near-dup still works across the boundary —
            # committed chunks are found by the store query instead of the
            # in-batch scan.
            size = max(1, getattr(deps.config, "course_ingest_chunk_size", 200))
            chunks = [courses[i:i + size] for i in range(0, len(courses), size)] or [[]]

            for n, chunk in enumerate(chunks):
                try:
                    # Ageing and the batch that resets it share ONE transaction.
                    # Ageing used to commit first and separately, so an extraction
                    # failure left the whole source aged with nothing reset — the
                    # failure itself pushed live courses toward deletion.
                    with store.transaction():
                        if n == 0:
                            if ageable:
                                store.age_missed([source])
                                aged.append(source)
                            elif result.ok:
                                not_aged.append(source)
                        if chunk:
                            total.merge(pipe.run(chunk))
                except Exception as exc:  # noqa: BLE001 - one chunk must not sink the cycle
                    failures.append(f"{source}[{n}]: {type(exc).__name__}: {exc}")

        return {
            "ingest_summary": total.as_dict(),
            # An ingest failure is a partial cycle: the run reports it and the
            # CLI exits non-zero, rather than looking clean because the fetch
            # half succeeded.
            "partial_cycle": partial or bool(failures),
            "ingest_errors": failures,
            "ageing": {"aged": aged, "not_aged": not_aged},
        }
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

        # A page that loaded but yielded no rating is the signature of a layout
        # change. It used to be indistinguishable from "this course has no
        # rating", which is how a redesign could quietly blank the whole corpus.
        for o in outcomes:
            r = o["result"]
            attempted = r.enriched + r.enrich_failed + r.enrich_unparsed
            if attempted and r.enrich_unparsed / attempted > 0.5:
                warnings.append(
                    f"{o['source']}: {r.enrich_unparsed} of {attempted} quality-signal pages "
                    f"loaded but matched nothing — the page layout has probably changed. "
                    f"Stored ratings were preserved rather than overwritten.")

        for source in (state.get("ageing") or {}).get("not_aged", []):
            warnings.append(
                f"{source}: this fetch was a sample, not a census, so its unseen courses "
                f"were not aged. Absence here is not evidence a course was withdrawn.")

        for message in state.get("ingest_errors") or []:
            warnings.append(f"ingest batch lost: {message}")

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
        # Whether this fetch was a census, and the quality-signal telemetry that
        # makes a silent enrichment failure visible.
        "truncated": o["result"].truncated,
        "may_age_inventory": o["result"].may_age_inventory,
        "enriched": o["result"].enriched,
        "enrich_failed": o["result"].enrich_failed,
        "enrich_unparsed": o["result"].enrich_unparsed,
    } for o in outcomes]
    return {
        "run_id": state.get("run_id"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "planned_sources": state.get("planned_sources", []),
        "partial_cycle": state.get("partial_cycle", False),
        "sources": per_source,
        "ingest": ingest,
        "ingest_errors": state.get("ingest_errors", []),
        "ageing": state.get("ageing", {}),
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
