"""One read-only view of the whole system.

Five agents, four tables and two cycles, and until this existed the only way to
answer "is any of this healthy?" was to run ``agent-b --check``, ``agent-d
--check``, and then write SQL by hand. Every number here is already computed
somewhere; the value is having them side by side, where a mismatch between two
of them is visible.

Read-only by construction: this module opens one connection, runs SELECTs, and
writes nothing. Like the orchestrator it lives beside the agents rather than
inside one, because it belongs to none of them.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from typing import Any

from shared.config import Config


def _age(value: Any) -> str:
    """How long ago, in plain words. The point of this view is spotting a stage
    that quietly stopped running, and a raw timestamp makes you do that
    subtraction in your head."""
    if value is None:
        return "never"
    if isinstance(value, datetime):
        moment = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - moment
    elif isinstance(value, date):
        delta = datetime.now(timezone.utc).date() - value
        days = delta.days
        return "today" if days == 0 else f"{days}d ago"
    else:
        return str(value)
    hours = delta.total_seconds() / 3600
    if hours < 1:
        return f"{int(delta.total_seconds() // 60)}m ago"
    if hours < 48:
        return f"{int(hours)}h ago"
    return f"{int(hours // 24)}d ago"


_QUERIES: dict[str, str] = {
    "postings": """
        SELECT status, count(*) AS n FROM job_postings GROUP BY status ORDER BY status
    """,
    "courses": """
        SELECT status, count(*) AS n FROM courses GROUP BY status ORDER BY status
    """,
    "demand_windows": """
        SELECT count(DISTINCT window_end) AS windows, count(*) AS rows,
               max(window_end) AS latest
          FROM skill_demand_stats
    """,
    "supply_windows": """
        SELECT count(DISTINCT window_end) AS windows, count(*) AS rows,
               max(window_end) AS latest
          FROM skill_supply_stats
    """,
    "concept_windows": """
        SELECT count(DISTINCT window_end) AS windows, count(*) AS rows,
               max(window_end) AS latest
          FROM concept_supply_stats
    """,
    "esco_demand": """
        SELECT count(*) FILTER (WHERE esco_code IS NOT NULL) AS mapped, count(*) AS total
          FROM skill_demand_stats
         WHERE window_end = (SELECT max(window_end) FROM skill_demand_stats)
    """,
    "esco_supply": """
        SELECT count(*) FILTER (WHERE esco_code IS NOT NULL) AS mapped, count(*) AS total
          FROM skill_supply_stats
         WHERE window_end = (SELECT max(window_end) FROM skill_supply_stats)
    """,
    "enrich_backlog": """
        -- TWO different numbers, because conflating them is misleading.
        -- `unrated` is a standing property of the corpus that will never reach
        -- zero: measured 2026-07-29, most of these courses carry no
        -- AggregateRating markup at all because nobody has rated them yet, and
        -- no amount of re-fetching changes that. `unchecked` is the actual work
        -- queue — courses the enrichment path has never looked at — and that
        -- one does drain.
        SELECT count(*) FILTER (WHERE rating IS NULL AND review_count IS NULL
                                  AND enrollment_count IS NULL) AS unrated,
               count(*) FILTER (WHERE price_observed_at IS NULL) AS unchecked,
               min(price_observed_at) AS oldest_check,
               count(*) AS total
          FROM courses
         WHERE status = 'active' AND duplicate_of IS NULL AND source = 'coursera'
    """,
    "job_health": """
        SELECT source, consecutive_failures, degraded_since, last_success_at
          FROM source_health ORDER BY source
    """,
    "course_health": """
        SELECT source, consecutive_failures, degraded_since, last_success_at
          FROM course_source_health ORDER BY source
    """,
}


def collect(config: Config) -> dict[str, Any]:
    """Run every query, tolerating tables that do not exist yet.

    A partially-migrated database is a normal state (Agent D's tables are absent
    until `agent-d --migrate`), and a status view that crashes on it is useless
    exactly when it would be most wanted.
    """
    import psycopg
    from psycopg.rows import dict_row

    out: dict[str, Any] = {}
    conn = psycopg.connect(config.require_database_url(), row_factory=dict_row)
    try:
        for name, sql in _QUERIES.items():
            try:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    rows = cur.fetchall()
                out[name] = [dict(r) for r in rows]
            except Exception as exc:  # noqa: BLE001 - a missing table is not fatal here
                conn.rollback()
                out[name] = {"unavailable": str(exc).splitlines()[0]}
    finally:
        conn.close()
    return out


def _counts(rows: Any) -> str:
    if isinstance(rows, dict):
        return "unavailable"
    return ", ".join(f"{r['status']}={r['n']}" for r in rows) or "empty"


def _window_line(rows: Any, keep: int) -> str:
    if isinstance(rows, dict) or not rows:
        return "unavailable"
    r = rows[0]
    if not r["rows"]:
        return "empty"
    # Retention prunes before the cycle writes the new snapshot, so keep..keep+1
    # is the steady state, not a bug.
    over = "  OVER RETENTION" if (r["windows"] or 0) > keep + 1 else ""
    return (f"{r['rows']} rows across {r['windows']} window(s), "
            f"latest {r['latest']} ({_age(r['latest'])}){over}")


def _ratio(rows: Any, label: str) -> str:
    if isinstance(rows, dict) or not rows:
        return f"{label}: unavailable"
    r = rows[0]
    total = r["total"] or 0
    if not total:
        return f"{label}: no rows"
    pct = 100.0 * (r["mapped"] or 0) / total
    return f"{label}: {r['mapped']}/{total} ({pct:.0f}%)"


def render(data: dict[str, Any], config: Config) -> list[str]:
    keep = config.stats_windows_to_keep()
    lines = [
        "  CORPUS",
        f"    job postings     {_counts(data['postings'])}",
        f"    courses          {_counts(data['courses'])}",
        "",
        "  AGGREGATION  (consumers read the latest window only)",
        f"    demand           {_window_line(data['demand_windows'], keep)}",
        f"    supply (skill)   {_window_line(data['supply_windows'], keep)}",
        f"    supply (concept) {_window_line(data['concept_windows'], keep)}",
        f"    retention        keeping {keep} window(s)",
        "",
        "  SHARED VOCABULARY  (the demand<->supply join runs on this)",
        f"    {_ratio(data['esco_demand'], 'demand mapped to ESCO')}",
        f"    {_ratio(data['esco_supply'], 'supply mapped to ESCO')}",
    ]

    backlog = data["enrich_backlog"]
    if not isinstance(backlog, dict) and backlog:
        r = backlog[0]
        total = r["total"] or 0
        lines += ["", "  QUALITY SIGNALS  (coursera; freeCodeCamp publishes none)",
                  f"    without a rating     {r['unrated'] or 0} of {total}"
                  f"  (mostly genuinely unrated, not a backlog)",
                  f"    never checked        {r['unchecked'] or 0}"
                  f"  (the work queue; drains a budget per cycle)",
                  f"    oldest check         {_age(r['oldest_check'])}"]

    for label, key in (("job sources", "job_health"), ("course sources", "course_health")):
        rows = data[key]
        if isinstance(rows, dict) or not rows:
            continue
        lines += ["", f"  {label.upper()}"]
        for r in rows:
            flag = "DEGRADED" if r["degraded_since"] else "ok"
            lines.append(
                f"    [{flag:>8}] {r['source']:<16} last success {_age(r['last_success_at'])}"
                + (f", {r['consecutive_failures']} consecutive failure(s)"
                   if r["consecutive_failures"] else ""))
    return lines


def main(argv: list[str] | None = None) -> int:
    config = Config()
    print("\n  Itqan | system status\n")
    try:
        config.require_database_url()
        data = collect(config)
    except Exception as exc:  # noqa: BLE001 - report, never raise, from a status view
        print(f"  could not read the database: {exc}\n", file=sys.stderr)
        return 2

    for line in render(data, config):
        print(line)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
