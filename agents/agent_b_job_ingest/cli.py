"""Agent B command-line frontend.

Dispatches to one of: a scheduled ingestion cycle (``--once`` / ``--loop``), or an
operator command (``--migrate``, ``--check``, ``--dry-run``, ``--label-sample``,
``--purge-source``). The cycle itself lives in ``runner.py``; this module only
parses arguments and formats output, so the two are testable apart.

``--once`` is the normal invocation and the one a scheduler runs. ``--dry-run``
fetches and scores but writes nothing, for inspecting a source before trusting
it. ``--fake-llm`` runs a real cycle with offline test doubles, so the whole path
can be exercised with no API key and no spend.
"""

from __future__ import annotations

import argparse
import sys

from shared.config import Config

PHASE = "7 (complete: scheduled ingestion cycle)"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-b",
        description=(
            "Ingest public job postings into the job_postings and "
            "skill_demand_stats tables. Runs on a 12-hour cycle."
        ),
        epilog=f"Currently at phase {PHASE}; ingestion is not yet implemented.",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--once", action="store_true", help="Run a single cycle and exit (the normal mode)"
    )
    mode.add_argument(
        "--loop",
        action="store_true",
        help="Run cycles forever in-process. For demos only — prefer cron/Task Scheduler "
        "with --once, which survives crashes and reboots and reports failures",
    )

    parser.add_argument(
        "--interval-hours", type=int, default=12, help="Cycle interval for --loop (default: 12)"
    )
    parser.add_argument(
        "--sources",
        help="Comma-separated source names to run (default: all enabled in config)",
    )
    parser.add_argument(
        "--limit", type=int, help="Max postings per source; caps a live run while testing"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and parse but write nothing — no database, no LLM, no embeddings",
    )
    parser.add_argument(
        "--no-embed", action="store_true", help="Skip embedding (near-dup detection is disabled)"
    )
    parser.add_argument("--fake-llm", action="store_true", help="Canned offline LLM, for testing")
    parser.add_argument("--run-id", help="Reuse a specific run id (advanced)")

    ops = parser.add_argument_group("operator commands")
    ops.add_argument(
        "--migrate",
        action="store_true",
        help="Apply pending database migrations and exit. Safe to re-run: already-applied "
        "migrations are skipped, and one edited after the fact is an error, not a silent no-op",
    )
    ops.add_argument(
        "--check",
        action="store_true",
        help="Report database connectivity, schema state and row counts, then exit",
    )
    ops.add_argument(
        "--label-sample",
        type=int,
        metavar="N",
        help="Export N ingested postings for a human to label, to measure legitimacy-filter "
        "precision. Output is gitignored: the labelled set is real data, not fixtures",
    )
    ops.add_argument(
        "--purge-source",
        metavar="SOURCE",
        help="Delete all postings from a decommissioned source. Never automatic — a "
        "scraper bug must not be able to trigger mass deletion",
    )

    return parser


def _migrate(config: Config) -> int:
    from .db import MigrationError, JobStore

    try:
        store = JobStore.from_config(config)
    except RuntimeError as exc:
        print(f"  {exc}", file=sys.stderr)
        return 2

    try:
        outstanding = store.pending_migrations()
        if not outstanding:
            print("  schema is up to date; nothing to apply\n")
            return 0
        print(f"  applying {len(outstanding)} migration(s):")
        for name in store.migrate():
            print(f"    + {name}")
        print()
        return 0
    except MigrationError as exc:
        print(f"\n  migration failed: {exc}\n", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"\n  could not reach the database: {exc}\n", file=sys.stderr)
        return 2
    finally:
        store.close()


def _check(config: Config) -> int:
    from .db import JobStore

    try:
        store = JobStore.from_config(config)
    except RuntimeError as exc:
        print(f"  {exc}", file=sys.stderr)
        return 2

    try:
        with store:
            health = store.health()
            print(f"  server     {health['server']}")
            print(f"  pgvector   {health['pgvector'] or 'NOT INSTALLED'}")
            print(f"  tables     {', '.join(health['tables']) or '(none)'}")
            print(f"  indexes    {len(health['indexes'])}")

            outstanding = store.pending_migrations()
            print(f"  migrations {'up to date' if not outstanding else outstanding}")

            if "job_postings" in health["tables"]:
                counts = store.counts()
                summary = ", ".join(f"{k}={v}" for k, v in counts.items()) or "empty"
                print(f"  rows       {summary}")
            print()
        return 0
    except Exception as exc:
        print(f"\n  could not reach the database: {exc}\n", file=sys.stderr)
        return 2
    finally:
        store.close()


def _dry_run(config: Config, args) -> int:
    """Fetch, parse and score — writing nothing anywhere.

    The phase-2b gate. It exists to be *watched*: the request cadence, the dates
    and links coming back, and the legitimacy band distribution are all things a
    human has to look at once before this is allowed to run unattended.
    """
    from .legitimacy import distribution, score_text
    from .sources.config import SourceConfigError, select_sources
    from .sources.factory import build_adapter

    try:
        sources = select_sources(args.sources, dry_run=True)
    except SourceConfigError as exc:
        print(f"  configuration error: {exc}\n", file=sys.stderr)
        return 2

    try:
        config.require_identified_user_agent()
    except RuntimeError as exc:
        print(f"  {exc}\n", file=sys.stderr)
        return 2

    print(f"  DRY RUN - no database writes, no LLM calls, no embeddings")
    print(f"  user agent: {config.user_agent}\n")

    assessments = []
    aggregable = 0
    aggregable_total = 0
    exit_code = 0

    for source in sources:
        if source.source_type == "telegram" and not source.terms_reviewed:
            print(f"  ! {source.name}: terms_reviewed is False — inspect-only\n")

        try:
            adapter = build_adapter(source, config=config)
        except NotImplementedError as exc:
            print(f"  - {source.name}: {exc}\n")
            continue

        result = adapter.fetch(limit=args.limit)
        state = "ok" if result.ok else ("PARTIAL" if result.partial else "FAILED")
        print(f"  [{state}] {source.name} - {len(result.postings)} postings, "
              f"{result.skipped} skipped, {result.pages_fetched} pages, "
              f"{result.bytes_fetched:,} bytes")
        if result.error:
            print(f"      error: {result.error}")
        # Partial or failed sources exit non-zero: a cron job whose only signal
        # is the exit code must not report success on a half-fetch.
        if not result.ok:
            exit_code = 1

        for posting in result.postings:
            assessment = score_text(
                posting.raw_description,
                company=posting.company,
                title=posting.title,
                # Extraction is phase 4. Adapters deliberately never set
                # `company`, so scoring no_employer_named here would fire on
                # every posting and make this gate's distribution a measure of
                # phase ordering rather than of the postings.
                employer_extracted=False,
            )
            assessments.append(assessment)
            aggregable_total += 1
            if (posting.listing_intent, posting.poster_type) == ("vacancy", "company"):
                aggregable += 1
            if posting.posted_date:
                date = posting.posted_date.date().isoformat()
            else:
                # Relative text, unresolved on purpose: the same string means a
                # different day on every fetch, so it is resolved once at
                # first-seen time rather than per cycle.
                date = f"~{posting.posted_date_text}" if posting.posted_date_text else "?"
            print(f"      {date}  {posting.title[:64]}")
            print(f"        url    {posting.source_url[:96]}")
            if (posting.listing_intent, posting.poster_type) != ("vacancy", "company"):
                print(f"        NOT AGGREGABLE  intent={posting.listing_intent} "
                      f"poster={posting.poster_type}")
            if posting.labels:
                print(f"        labels {', '.join(posting.labels[:6])}")
            if posting.outbound_links:
                print(f"        links  {len(posting.outbound_links)}: {posting.outbound_links[0]}")
            if assessment.signals:
                print(f"        risk   {assessment.risk:.2f} "
                      f"(score {assessment.score:.2f}, {assessment.band(config)}) "
                      f"<- {', '.join(assessment.codes)}")
        print()

    if aggregable_total:
        eligible = 100 * aggregable / aggregable_total
        print(f"  aggregable: {aggregable}/{aggregable_total} ({eligible:.0f}%) "
              "- only intent=vacancy AND poster=company reaches Agent C")
        if aggregable == 0:
            print("  none of these postings would contribute to skill_demand_stats.")
        print()

    if assessments:
        counts = distribution(assessments, config)
        total = len(assessments)
        band_pct = 100 * counts["adjudicate"] / total
        print(f"  legitimacy bands over {total} postings: "
              f"clean={counts['clean']} adjudicate={counts['adjudicate']} "
              f"reject={counts['reject']}")
        print(f"  adjudicate band: {band_pct:.0f}%", end="")
        # The gate metric. Above ~20% the rules are producing ambiguity rather
        # than resolving it, and the answer is to retune the weights — not to
        # send more postings to an LLM.
        print("  <- ABOVE 20%: retune the weights\n" if band_pct > 20 else "  (target: <=20%)\n")

    return exit_code


def _print_cycle(state: dict) -> None:
    ingest = state.get("ingest_summary", {})
    agg = state.get("aggregation_summary", {})
    stale = state.get("staleness_summary", {})
    print("  sources:")
    for s in state.get("run_log", {}).get("sources", []):
        flag = "ok" if s["ok"] else ("PARTIAL" if s["partial"] else "FAILED")
        print(f"    [{flag}] {s['source']:<14} {s['postings']} postings, "
              f"{s['skipped']} skipped, {s['pages_fetched']} pages")
        if s["error"]:
            print(f"          error: {s['error']}")
    print(f"  ingest:    new={ingest.get('new', 0)} changed={ingest.get('changed', 0)} "
          f"unchanged={ingest.get('unchanged', 0)} rejected={ingest.get('rejected', 0)} "
          f"needs_review={ingest.get('needs_review', 0)}")
    print(f"             link_dups={ingest.get('link_duplicates', 0)} "
          f"embed_dups={ingest.get('embed_duplicates', 0)} "
          f"extractions={ingest.get('extractions', 0)} embeddings={ingest.get('embeddings', 0)}")
    print(f"  staleness: marked_stale={stale.get('marked_stale', 0)} pruned={stale.get('pruned', 0)}")
    print(f"  demand:    {agg.get('rows_written', 0)} skill rows over "
          f"{agg.get('sectors_with_current_demand', 0)} sectors "
          f"(window {agg.get('window_start', '?')} .. {agg.get('window_end', '?')})")
    for warning in state.get("warnings", []):
        print(f"  ! {warning}")
    if state.get("run_log_path"):
        print(f"  run log -> {state['run_log_path']}")


def _run_once(config: Config, args) -> int:
    from .runner import run_cycle

    names = [n.strip() for n in args.sources.split(",")] if args.sources else None
    try:
        result = run_cycle(
            config,
            source_names=names,
            limit=args.limit,
            fake_llm=args.fake_llm,
            no_embed=args.no_embed,
            run_id=args.run_id,
        )
    except RuntimeError as exc:
        print(f"  {exc}\n", file=sys.stderr)
        return 2

    if not result.ran:
        print("  another cycle is already running (advisory lock held); exiting cleanly.\n")
        return 0

    _print_cycle(result.state or {})
    if result.exit_code != 0:
        print("\n  cycle completed with a partial fetch (exit 1).\n")
    else:
        print("\n  cycle complete.\n")
    return result.exit_code


def _run_loop(config: Config, args) -> int:
    import time

    print("  --loop runs cycles in-process, for demos only. For production prefer")
    print("  Task Scheduler/cron with --once, which survives crashes and reboots.\n")
    interval = max(1, args.interval_hours) * 3600
    try:
        while True:
            code = _run_once(config, args)
            print(f"  sleeping {args.interval_hours}h until the next cycle "
                  f"(Ctrl-C to stop). last exit={code}\n")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n  stopped.\n")
        return 0


def _purge_source(config: Config, args) -> int:
    from .db import JobStore

    try:
        store = JobStore.from_config(config)
    except RuntimeError as exc:
        print(f"  {exc}", file=sys.stderr)
        return 2
    try:
        with store:
            with store.transaction():
                deleted = store.purge_source(args.purge_source)
            print(f"  purged {deleted} posting(s) from source '{args.purge_source}'.\n")
        return 0
    except Exception as exc:
        print(f"\n  purge failed: {exc}\n", file=sys.stderr)
        return 2


def _label_sample(config: Config, args) -> int:
    import json

    from .db import JobStore

    try:
        store = JobStore.from_config(config)
    except RuntimeError as exc:
        print(f"  {exc}", file=sys.stderr)
        return 2
    try:
        with store:
            rows = store.sample_for_labelling(args.label_sample)
        out_dir = config.output_dir / "labels"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "legitimacy_labels_sample.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                # A blank 'human_label' for the reviewer to fill: "legit" / "scam".
                row["human_label"] = ""
                fh.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")
        print(f"  exported {len(rows)} postings to {path}")
        print("  label each 'human_label' as legit|scam, then measure reject-class precision.")
        print("  This file is gitignored: it is real data, not a fixture.\n")
        return 0
    except Exception as exc:
        print(f"\n  label export failed: {exc}\n", file=sys.stderr)
        return 2


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = Config()

    print(f"\n  Itqan | Agent B - job ingestion")
    print(f"  phase {PHASE}\n")

    # operator commands first — each exits without running a cycle
    if args.migrate:
        return _migrate(config)
    if args.check:
        return _check(config)
    if args.purge_source:
        return _purge_source(config, args)
    if args.label_sample:
        return _label_sample(config, args)
    if args.dry_run:
        return _dry_run(config, args)

    # the cycle
    if args.loop:
        return _run_loop(config, args)
    if args.once:
        return _run_once(config, args)

    # No mode given: run a single cycle, the normal invocation.
    return _run_once(config, args)


if __name__ == "__main__":
    raise SystemExit(main())
