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
        "--esco-sync",
        nargs="?",
        const="__default__",
        metavar="CSV",
        help="Load the ESCO skills taxonomy from a downloaded CSV (default: ESCO/skills_en.csv) "
        "and embed its labels (one-off, a few cents). Idempotent and resumable",
    )
    ops.add_argument(
        "--esco-version",
        default="1.2",
        help="Taxonomy version recorded with --esco-sync (default: 1.2). Bump it when "
        "loading a newer ESCO release; 'unmapped' skills are then retried",
    )
    ops.add_argument(
        "--label-sample",
        type=int,
        metavar="N",
        help="Export N ingested postings for a human to label, to measure legitimacy-filter "
        "precision. Output is gitignored: the labelled set is real data, not fixtures",
    )
    ops.add_argument(
        "--reslice-roundups",
        action="store_true",
        help="Give each vacancy split from a multi-job post its OWN text. Repairs rows that "
        "stored the whole article on every child. Re-extracts from the stored body — no "
        "network. Combine with --dry-run to report without writing",
    )
    ops.add_argument(
        "--backfill-destinations",
        action="store_true",
        help="Follow stored postings' outbound links to the employer's own page and enrich "
        "from it. For rows ingested before enrichment existed; re-fetches each article "
        "because outbound links were never stored",
    )
    ops.add_argument(
        "--prune-without-destination",
        action="store_true",
        help="Delete postings that reach no employer job page, for sources configured "
        "require_destination. Trace first with --backfill-destinations; only rows actually "
        "looked at are removed. ALWAYS run with --dry-run first",
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
            from shared.display import arabize

            print(f"      {date}  {arabize(posting.title[:64])}")
            print(f"        url    {posting.source_url[:96]}")
            if (posting.listing_intent, posting.poster_type) != ("vacancy", "company"):
                print(f"        NOT AGGREGABLE  intent={posting.listing_intent} "
                      f"poster={posting.poster_type}")
            if posting.labels:
                # Display-only reshaping; the stored labels stay logical-order.
                print(f"        labels {arabize(', '.join(posting.labels[:6]))}")
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
        held = s.get("already_held") or 0
        print(f"    [{flag}] {s['source']:<14} {s['postings']} postings, "
              f"{s['skipped']} skipped, {s['pages_fetched']} pages"
              + (f", {held} already held (not re-fetched)" if held else ""))
        if s["error"]:
            print(f"          error: {s['error']}")
    print(f"  ingest:    new={ingest.get('new', 0)} changed={ingest.get('changed', 0)} "
          f"unchanged={ingest.get('unchanged', 0)} rejected={ingest.get('rejected', 0)} "
          f"needs_review={ingest.get('needs_review', 0)}")
    print(f"             link_dups={ingest.get('link_duplicates', 0)} "
          f"dest_dups={ingest.get('destination_duplicates', 0)} "
          f"embed_dups={ingest.get('embed_duplicates', 0)} "
          f"extractions={ingest.get('extractions', 0)} embeddings={ingest.get('embeddings', 0)}"
          + (f" ({ingest['extraction_failures']} failed)"
             if ingest.get("extraction_failures") else ""))
    if ingest.get("skipped_no_destination"):
        print(f"             {ingest['skipped_no_destination']} posting(s) reached no employer "
              f"page and were NOT stored (source requires a destination)")
    # Printed every cycle, deliberately. This is the verifier working, not an
    # error rate — measured on the first live cycle without it, the model
    # asserted an arrangement on 19 of 19 postings that stated none. A run where
    # this reads 0 across hundreds of postings means the check has stopped
    # running, and that is the failure nobody would otherwise notice.
    print(f"             destinations={ingest.get('root_fetches', 0)} fetched, "
          f"{ingest.get('root_enrichments', 0)} enriched; "
          f"unstated_dropped={ingest.get('unstated_fields_dropped', 0)}")
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


def _esco_sync(config: Config, args) -> int:
    from pathlib import Path

    from shared.embeddings import build_embedder

    from .db import JobStore, apply_migrations
    from .esco_map import sync_taxonomy

    path = config.esco_csv_path if args.esco_sync == "__default__" else Path(args.esco_sync)
    if not path.exists():
        print(f"  {path} not found.", file=sys.stderr)
        print("  Download the ESCO skills CSV bundle (English) from", file=sys.stderr)
        print("  https://esco.ec.europa.eu/en/use-esco/download and place skills_en.csv there,",
              file=sys.stderr)
        print("  or pass the file path: --esco-sync path\\to\\skills_en.csv", file=sys.stderr)
        return 2

    try:
        dsn = config.require_database_url()
        apply_migrations(dsn)
        embedder = build_embedder(config)
    except RuntimeError as exc:
        print(f"  {exc}", file=sys.stderr)
        return 2

    with JobStore(dsn) as store:
        # No outer transaction: sync_taxonomy commits per batch, which is what
        # makes an interrupted run resumable instead of voided.
        summary = sync_taxonomy(store, embedder, path=path, version=args.esco_version)
        status = store.esco_status()

    print(f"  parsed   {summary.skills:,} concepts / {summary.labels:,} labels from {path.name}")
    print(f"  embedded {summary.embedded:,} labels this run")
    print(f"  taxonomy now: {status['skills']:,} skills, {status['labels']:,} labels "
          f"({status['embedded']:,} embedded), version {status['version']}\n")
    return 0


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


def _reslice_roundups(config: Config, args) -> int:
    """Narrow each split vacancy to its own text.

    No network: the stored description IS the article, which is precisely the
    defect being repaired, so the source text is already in hand.
    """
    from .backfill import reslice_roundups
    from .db import JobStore
    from .prompts.extraction import EXTRACTION_PROMPT
    from .schemas import JobExtractionBatch
    from shared.llm import build_llm, structured

    sources = [s.strip() for s in (args.sources or "").split(",") if s.strip()] or None
    llm = build_llm(config)

    if args.dry_run:
        print("  DRY RUN - reporting only, nothing is written\n")

    try:
        with JobStore.from_config(config) as store:
            with store.transaction():
                report = reslice_roundups(
                    store=store,
                    extractor=EXTRACTION_PROMPT | structured(llm, JobExtractionBatch),
                    sources=sources, limit=args.limit,
                    dry_run=bool(args.dry_run), config=config,
                )
    except Exception as exc:
        print(f"\n  backfill failed: {exc}\n", file=sys.stderr)
        return 2

    print(f"  posts considered : {report.posts_considered}")
    print(f"  vacancies narrowed: {report.rows_updated}")
    print(f"  left as they were : {report.rows_unchanged}")
    # Not an error: the row keeps the full body, exactly what it had before. It
    # is printed because a run where this is the whole population means the
    # model stopped quoting and the rows would stay clustered silently.
    print(f"  span unverified   : {report.unverified} (kept the full body)")
    print(f"  employers recorded: {report.employers_recorded}")
    for failure in report.failures[:5]:
        print(f"  ! {failure}")
    print()
    return 0


def _backfill_destinations(config: Config, args) -> int:
    """Follow outbound links on rows that never had enrichment run."""
    from .backfill import backfill_destinations
    from .db import JobStore
    from .prompts.extraction import EXTRACTION_PROMPT
    from .root_fetch import RootFetcher
    from .schemas import JobExtractionBatch
    from shared.llm import build_llm, structured

    try:
        config.require_identified_user_agent()
    except RuntimeError as exc:
        print(f"  {exc}\n", file=sys.stderr)
        return 2

    sources = [s.strip() for s in (args.sources or "").split(",") if s.strip()] or None
    llm = build_llm(config)
    fetcher = RootFetcher(config, interval_s=config.root_fetch_interval_s)

    if args.dry_run:
        print("  DRY RUN - reporting only, nothing is written\n")

    try:
        with JobStore.from_config(config) as store:
            with store.transaction():
                report = backfill_destinations(
                    store=store,
                    extractor=EXTRACTION_PROMPT | structured(llm, JobExtractionBatch),
                    root_fetcher=fetcher,
                    article_fetch=_article_links(config),
                    sources=sources, limit=args.limit,
                    dry_run=bool(args.dry_run), config=config,
                )
    except Exception as exc:
        print(f"\n  backfill failed: {exc}\n", file=sys.stderr)
        return 2
    finally:
        fetcher.close()

    print(f"  postings examined : {report.posts_considered}")
    print(f"  destinations found: {report.rows_updated}")
    print(f"  no link to follow : {report.rows_unchanged}")
    for failure in report.failures[:5]:
        print(f"  ! {failure}")
    print()
    return 0


def _article_links(config: Config):
    """Recover a posting's outbound links by re-fetching its article.

    They were never persisted, so a backfill has to go and look. Robots-checked
    through the same policy every adapter uses; a refusal yields no links rather
    than an exception, because one unreachable article must not end the run.
    """
    from selectolax.parser import HTMLParser
    from shared.scraping import build_client, build_robots
    from shared.scraping.http import SourcePolicy

    client = build_client(
        source="backfill",
        policy=SourcePolicy(min_interval_s=2.0, max_bytes=config.max_response_bytes),
        config=config,
    )
    robots = build_robots(source="backfill", config=config)

    seen: dict[str, tuple[str, ...]] = {}

    def links(url: str) -> tuple[str, ...]:
        # One fetch per ARTICLE, not per row. A roundup's 19 children all carry
        # `post_url#role`, and the fragment never reaches the server — without
        # this, 245 el7far children would re-request the same 40 pages.
        article = url.split("#", 1)[0]
        if article in seen:
            return seen[article]
        if not robots.can_fetch(article):
            seen[article] = ()
            return ()
        html = client.get_text(article)
        tree = HTMLParser(html)
        body = (tree.css_first(".post-body") or tree.css_first(".entry-content")
                or tree.css_first("article") or tree.body)
        if body is None:
            seen[article] = ()
            return ()
        found = [a.attributes.get("href") or "" for a in body.css("a[href]")]
        result = tuple(dict.fromkeys(h for h in found if h.startswith("http")))
        seen[article] = result
        return result

    return links


def _prune_without_destination(config: Config, args) -> int:
    """Delete postings that lead nowhere — the user's "only final destinations".

    Two safeguards it will not run without, both because 481 rows is past the
    point where a number alone is enough to approve a deletion:

      * it reports the full breakdown by source and reason FIRST, every time,
        dry run or not;
      * it only deletes rows that were actually traced. A row nobody looked at
        is not evidence of anything, and survives.

    Sources come from the registry's `require_destination` flag, so a source
    that never opted in — GulfTalent — cannot be caught by a widening mistake
    here.
    """
    from .db import JobStore
    from .sources.config import DEFAULT_SOURCES

    named = [c.name for c in DEFAULT_SOURCES if c.require_destination]
    if args.sources:
        wanted = {s.strip() for s in args.sources.split(",") if s.strip()}
        named = [n for n in named if n in wanted]
    if not named:
        print("  no source is configured require_destination; nothing to prune.\n")
        return 0

    print(f"  sources in scope: {', '.join(named)}")
    print("  (a source without require_destination is untouchable here by construction)\n")

    try:
        with JobStore.from_config(config) as store:
            survey = store.destination_survey(named)
            if not survey:
                print("  these sources have no rows.\n")
                return 0

            keep = doomed = untraced = 0
            print(f"  {'source':<14} {'reason':<22} {'rows':>6}   verdict")
            for row in survey:
                if row["has_destination"]:
                    verdict, n = "KEEP (reaches a job page)", 0
                    keep += row["rows"]
                elif row["status"] == "never traced":
                    verdict, n = "KEEP (never traced — trace it first)", 0
                    keep += row["rows"]
                    untraced += row["rows"]
                else:
                    verdict = "DELETE"
                    doomed += row["rows"]
                print(f"  {row['source']:<14} {row['status']:<22} {row['rows']:>6}   {verdict}")

            print(f"\n  keep {keep}, delete {doomed}")
            if untraced:
                print(f"  ! {untraced} row(s) were never traced and are being KEPT. Run "
                      f"--backfill-destinations first, or they will survive a prune "
                      f"that was meant to remove them.")

            if args.dry_run:
                print("\n  DRY RUN - nothing was deleted.\n")
                return 0
            if not doomed:
                print()
                return 0

            with store.transaction():
                removed = store.prune_without_destination(named)
            print(f"  deleted {removed} posting(s).")

            # `skill_demand_stats.sample_postings` carries posting ids. Leaving
            # them would make every demand figure cite rows that no longer
            # exist, which is a worse failure than the clutter just removed.
            from .aggregate import recompute_stats

            with store.transaction():
                stats = recompute_stats(store, config=config)
            print(f"  recomputed demand stats: {stats.get('rows_written', 0)} rows over "
                  f"{stats.get('sectors_with_current_demand', 0)} sectors.\n")
        return 0
    except Exception as exc:
        print(f"\n  prune failed: {exc}\n", file=sys.stderr)
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
    if args.reslice_roundups:
        return _reslice_roundups(config, args)
    if args.backfill_destinations:
        return _backfill_destinations(config, args)
    if args.prune_without_destination:
        return _prune_without_destination(config, args)
    if args.esco_sync:
        return _esco_sync(config, args)
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
