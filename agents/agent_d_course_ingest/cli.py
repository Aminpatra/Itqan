"""Agent D command-line frontend — scheduled course ingestion (3-day cycle).

Mirrors Agent B's CLI. Needs Postgres and an OpenAI key (extraction + embeddings)
and, at runtime, the shared ESCO taxonomy synced by ``agent-b --esco-sync``.
"""

from __future__ import annotations

import argparse
import sys

from shared.config import Config

PHASE = "complete (course ingestion, 3-day cycle)"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agent-d",
        description="Ingest online courses into the courses and skill_supply_stats tables "
        "(3-day cycle). Supply side of the skill market.",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="Run a single cycle and exit (normal mode)")
    mode.add_argument("--loop", action="store_true", help="Run cycles forever in-process (demos only)")
    p.add_argument("--interval-hours", type=int, default=72, help="Cycle interval for --loop (default 72)")
    p.add_argument("--sources", help="Comma-separated source names (default: all enabled)")
    p.add_argument("--limit", type=int, help="Max courses per source")
    p.add_argument("--dry-run", action="store_true", help="Fetch and parse; write nothing")
    p.add_argument("--no-embed", action="store_true", help="Skip embedding (near-dup disabled)")
    p.add_argument("--fake-llm", action="store_true", help="Canned offline LLM, for testing")
    p.add_argument("--run-id", help="Reuse a specific run id")

    ops = p.add_argument_group("operator commands")
    ops.add_argument("--migrate", action="store_true", help="Apply pending migrations and exit")
    ops.add_argument("--check", action="store_true", help="Report DB/schema/row counts and exit")
    ops.add_argument("--purge-source", metavar="SOURCE", help="Delete a decommissioned source's courses")
    return p


def _migrate(config: Config) -> int:
    from .db import CourseStore, MigrationError

    try:
        store = CourseStore.from_config(config)
    except RuntimeError as exc:
        print(f"  {exc}", file=sys.stderr); return 2
    try:
        outstanding = store.pending_migrations()
        if not outstanding:
            print("  schema is up to date; nothing to apply\n"); return 0
        print(f"  applying {len(outstanding)} migration(s):")
        for name in store.migrate():
            print(f"    + {name}")
        print(); return 0
    except MigrationError as exc:
        print(f"\n  migration failed: {exc}\n", file=sys.stderr); return 2
    finally:
        store.close()


def _check(config: Config) -> int:
    from .db import CourseStore

    try:
        store = CourseStore.from_config(config)
    except RuntimeError as exc:
        print(f"  {exc}", file=sys.stderr); return 2
    try:
        with store:
            h = store.health()
            print(f"  server        {h['server']}")
            print(f"  pgvector      {h['pgvector'] or 'NOT INSTALLED'}")
            print(f"  courses       {'yes' if h['courses_table'] else 'MISSING'}")
            print(f"  supply stats  {'yes' if h['supply_table'] else 'MISSING'}")
            print(f"  esco taxonomy {'synced' if h['esco_taxonomy'] else 'NOT SYNCED (run agent-b --esco-sync)'}")
            outstanding = store.pending_migrations()
            print(f"  migrations    {'up to date' if not outstanding else outstanding}")
            if h["courses_table"]:
                counts = store.counts()
                print(f"  rows          {', '.join(f'{k}={v}' for k, v in counts.items()) or 'empty'}")
            print()
        return 0
    except Exception as exc:
        print(f"\n  could not reach the database: {exc}\n", file=sys.stderr); return 2
    finally:
        store.close()


def _dry_run(config: Config, args) -> int:
    from shared.display import arabize

    from .sources.config import SourceConfigError, select_sources
    from .sources.factory import build_adapter

    try:
        sources = select_sources(args.sources, dry_run=True)
    except SourceConfigError as exc:
        print(f"  configuration error: {exc}\n", file=sys.stderr); return 2
    try:
        config.require_identified_user_agent()
    except RuntimeError as exc:
        print(f"  {exc}\n", file=sys.stderr); return 2

    print("  DRY RUN - no database writes, no LLM, no embeddings")
    print(f"  user agent: {config.user_agent}\n")
    exit_code = 0
    for source in sources:
        if source.source_type == "api" and not source.terms_reviewed:
            print(f"  ! {source.name}: terms_reviewed is False - inspect only\n")
        adapter = build_adapter(source, config=config)
        result = adapter.fetch(limit=args.limit)
        state = "ok" if result.ok else ("PARTIAL" if result.partial else "FAILED")
        print(f"  [{state}] {source.name} - {len(result.courses)} courses, "
              f"{result.skipped} skipped, {result.pages_fetched} pages, {result.bytes_fetched:,} bytes")
        if result.error:
            print(f"      error: {result.error}")
        if not result.ok:
            exit_code = 1
        for c in result.courses:
            print(f"      {arabize(c.name[:56])}  ({c.provider or '?'})")
            print(f"        {c.source_url[:90]}")
            if c.license:
                print(f"        license: {c.license}")
        print()
    return exit_code


def _run_once(config: Config, args) -> int:
    from shared.display import arabize

    from .runner import run_cycle

    names = [n.strip() for n in args.sources.split(",")] if args.sources else None
    try:
        result = run_cycle(config, source_names=names, limit=args.limit,
                           fake_llm=args.fake_llm, no_embed=args.no_embed, run_id=args.run_id)
    except RuntimeError as exc:
        print(f"  {exc}\n", file=sys.stderr); return 2
    if not result.ran:
        print("  another cycle is already running (advisory lock held); exiting cleanly.\n"); return 0

    st = result.state or {}
    ingest = st.get("ingest_summary", {})
    agg = st.get("aggregation_summary", {})
    print("  sources:")
    for s in st.get("run_log", {}).get("sources", []):
        flag = "ok" if s["ok"] else ("PARTIAL" if s["partial"] else "FAILED")
        print(f"    [{flag}] {s['source']:<14} {s['courses']} courses, {s['skipped']} skipped")
    print(f"  ingest:    new={ingest.get('new',0)} changed={ingest.get('changed',0)} "
          f"unchanged={ingest.get('unchanged',0)} rejected={ingest.get('rejected',0)} "
          f"embed_dups={ingest.get('embed_duplicates',0)}")
    print(f"  staleness: {st.get('staleness_summary', {})}")
    print(f"  supply:    {agg.get('rows_written',0)} skill rows over "
          f"{agg.get('skills_with_supply',0)} skills; esco {agg.get('esco', {})}")
    if st.get("run_log_path"):
        print(f"  run log -> {st['run_log_path']}")
    print("\n  cycle complete." if result.exit_code == 0 else
          "\n  cycle completed with a partial fetch (exit 1).")
    print()
    return result.exit_code


def _run_loop(config: Config, args) -> int:
    import time
    print("  --loop runs in-process, for demos only. Prefer Task Scheduler with --once.\n")
    interval = max(1, args.interval_hours) * 3600
    try:
        while True:
            _run_once(config, args)
            print(f"  sleeping {args.interval_hours}h (Ctrl-C to stop)\n")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n  stopped.\n"); return 0


def _purge_source(config: Config, args) -> int:
    from .db import CourseStore
    try:
        store = CourseStore.from_config(config)
    except RuntimeError as exc:
        print(f"  {exc}", file=sys.stderr); return 2
    try:
        with store:
            with store.transaction():
                deleted = store.purge_source(args.purge_source)
            print(f"  purged {deleted} course(s) from source '{args.purge_source}'.\n")
        return 0
    except Exception as exc:
        print(f"\n  purge failed: {exc}\n", file=sys.stderr); return 2


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = Config()
    print(f"\n  Itqan | Agent D - course ingestion\n  phase {PHASE}\n")

    if args.migrate:
        return _migrate(config)
    if args.check:
        return _check(config)
    if args.purge_source:
        return _purge_source(config, args)
    if args.dry_run:
        return _dry_run(config, args)
    if args.loop:
        return _run_loop(config, args)
    return _run_once(config, args)


if __name__ == "__main__":
    raise SystemExit(main())
