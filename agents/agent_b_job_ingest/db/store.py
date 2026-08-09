"""Database access for Agent B.

**Every SQL string in this project lives in this module.** Nodes call methods;
they never build SQL. That keeps the queries reviewable in one place, makes the
partial-index predicates (which callers must repeat verbatim) auditable, and
means a node cannot quietly grow an unindexed query.

PHASE 2 SKELETON. Connection handling, migrations and the advisory lock are
real. Ingestion methods arrive with the phases that need them and are declared
below rather than invented ahead of their callers.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import psycopg
from psycopg.rows import dict_row

from shared.config import Config

from .migrate import apply_migrations, pending

# Namespaced so it cannot collide with another application's advisory lock on a
# shared server.
CYCLE_LOCK_KEY = "itqan_agent_b_cycle"


def to_float_list(value: Any) -> list[float] | None:
    """Normalize whatever the driver returns for a vector into plain floats.

    pgvector hands back a ``Vector`` object, not a list: it has ``.to_list()``
    and ``.dimensions()`` but no ``len()``, and ``np.asarray`` on it yields
    object dtype. Every read path goes through here so that a library type never
    reaches a node — which is what keeps the vector backend swappable without
    every consumer learning a new API.
    """
    if value is None:
        return None
    if isinstance(value, list):
        return value
    to_list = getattr(value, "to_list", None)
    if callable(to_list):
        return list(to_list())
    return [float(x) for x in value]


@dataclass
class LockNotAcquired(RuntimeError):
    """Another cycle holds the lock. Not an error condition — exit 0."""

    key: str = CYCLE_LOCK_KEY


class JobStore:
    """Owns the connection to Postgres.

    Not a pool. Agent B is a batch job: one process, one cycle, then exit. A
    pool would add a moving part whose main benefit — reusing connections across
    concurrent short requests — is a shape this workload does not have. The
    per-source scrape branches run concurrently but do not touch the database;
    all writes happen in the linear tail after fan-in.
    """

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._conn: psycopg.Connection | None = None
        # A SEPARATE connection for reads issued during the scrape phase — see
        # `exists_during_scrape` for why sharing the main one silently discarded
        # a whole cycle's writes.
        self._scrape_conn: psycopg.Connection | None = None
        self._scrape_lock = threading.Lock()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: Config) -> "JobStore":
        return cls(config.require_database_url())

    def connect(self) -> psycopg.Connection:
        """The cycle's connection. AUTOCOMMIT, and that is load-bearing.

        psycopg opens an IMPLICIT transaction on a connection's first statement.
        Any read taken before the first `transaction()` block therefore leaves
        one open, every later `with conn.transaction()` nests inside it as a
        SAVEPOINT, and releasing a savepoint commits nothing — so the work is
        rolled back when the connection closes, with no error anywhere.

        This has now cost two silent data losses:

          * a scrape-phase `existing_ids` lookup — 25 postings extracted,
            embedded, logged `written=25`, and zero rows in the table;
          * a `destination_survey` read before a prune — "deleted 432
            posting(s)" printed, 432 rows still present.

        Both were fixed locally, twice, which is the signal that the connection
        was the problem rather than either caller. With autocommit a bare read
        holds nothing, and `transaction()` opens a REAL transaction that commits
        on exit. Every writer already wraps its work in `transaction()`, so
        batch rollback-on-error is unchanged.
        """
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self.dsn, row_factory=dict_row, autocommit=True)
            self._register_vector(self._conn)
        return self._conn

    @staticmethod
    def _register_vector(conn: psycopg.Connection) -> None:
        """Teach psycopg the `vector` type.

        Without this a list[float] round-trip goes through string formatting,
        which is a classic source of silent precision and format bugs. Tolerated
        as a no-op when the extension is not yet installed, so `--migrate` can
        run against a virgin database.
        """
        try:
            from pgvector.psycopg import register_vector

            register_vector(conn)
        except Exception:
            pass

    def exists_during_scrape(self, posting_ids: list[str]) -> set[str]:
        """`existing_ids`, but safe to call from a scrape branch.

        Two reasons this cannot share the main connection, and the first one
        cost a silent data loss:

        **1. It would poison the cycle's transactions.** psycopg opens an
        implicit transaction on the first statement. Scraping happens BEFORE any
        `transaction()` block, so a read here leaves one open; every later
        `with conn.transaction()` then nests as a SAVEPOINT rather than being the
        transaction, and releasing a savepoint commits nothing. The cycle reports
        `written=25`, raises nothing, and the whole thing rolls back when the
        connection closes. Measured exactly that: 25 postings extracted,
        embedded, "written", and zero rows in the table.

        **2. Scrape branches run concurrently.** LangGraph fans them out with
        `Send`, and a psycopg connection is not safe for concurrent use.

        So: a dedicated autocommit connection, opened on first use and closed
        with the store. Autocommit because these are point reads that must never
        hold a transaction open behind a crawl that takes minutes.
        """
        if not posting_ids:
            return set()
        with self._scrape_lock:
            if self._scrape_conn is None or self._scrape_conn.closed:
                self._scrape_conn = psycopg.connect(
                    self.dsn, row_factory=dict_row, autocommit=True)
            with self._scrape_conn.cursor() as cur:
                cur.execute(
                    "SELECT posting_id FROM job_postings WHERE posting_id = ANY(%s)",
                    (posting_ids,),
                )
                return {r["posting_id"] for r in cur.fetchall()}

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
        self._conn = None
        if self._scrape_conn is not None and not self._scrape_conn.closed:
            self._scrape_conn.close()
        self._scrape_conn = None

    def __enter__(self) -> "JobStore":
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[psycopg.Connection]:
        """One transaction.

        Scoped per source batch, not per cycle: a source blocked partway must
        commit what it already ingested rather than discarding three healthy
        sources' work alongside it.
        """
        conn = self.connect()
        with conn.transaction():
            yield conn


    # ------------------------------------------------------------------
    # operator backfills (agents/agent_b_job_ingest/backfill.py)
    # ------------------------------------------------------------------
    def clustered_posts(self, *, sources: list[str] | None = None,
                        limit: int | None = None) -> list[dict[str, Any]]:
        """Multi-vacancy posts whose children still share one body.

        `needs_reslice` is computed rather than flagged: a post needs work when
        more than one of its children holds the SAME description. That makes the
        backfill idempotent by observation — re-running it finds nothing to do —
        without a column that could disagree with the rows it describes.
        """
        conn = self.connect()
        clauses = ["source_post_url IS NOT NULL", "source_post_url <> source_url"]
        params: dict[str, Any] = {}
        if sources:
            clauses.append("source = ANY(%(sources)s)")
            params["sources"] = list(sources)
        params["limit"] = limit

        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT source_post_url,
                       min(source)      AS source,
                       min(source_type) AS source_type,
                       -- Deliberately NOT min(title): a child's title is its
                       -- ROLE ("Accounts Payable In-Charge"), and the roundup's
                       -- own headline is stored nowhere. Passing a role title
                       -- back to the extractor told it the article was about
                       -- that one job, and it returned exactly one vacancy for
                       -- a 19-vacancy post. The headline is the body's first
                       -- line; the caller reads it from there.
                       array_agg(json_build_object('posting_id', posting_id,
                                                   'title', title,
                                                   'company', company)) AS children_rows,
                       count(*)                        AS children,
                       count(DISTINCT raw_description) AS distinct_bodies,
                       -- The longest body is the un-narrowed article: a child
                       -- that has already been sliced is strictly shorter.
                       (array_agg(raw_description ORDER BY length(raw_description) DESC))[1] AS body,
                       array_agg(posting_id) AS posting_ids
                  FROM job_postings
                 WHERE {' AND '.join(clauses)}
                 GROUP BY source_post_url
                HAVING count(*) > 1
                 ORDER BY count(*) DESC
                 LIMIT %(limit)s
                """,
                params,
            )
            posts = [dict(r) for r in cur.fetchall()]

        for post in posts:
            # Every child holding one identical body is the defect. Once narrowed,
            # each child differs and there is nothing left to do.
            post["needs_reslice"] = post["distinct_bodies"] < post["children"]
        return posts

    def narrow_descriptions(self, updates: dict[str, str]) -> int:
        """Replace raw_description on specific rows.

        Deliberately the ONLY write this backfill can make. It cannot insert,
        delete, or touch identity — a repair command that could remove postings
        is a different and much more dangerous tool.
        """
        if not updates:
            return 0
        conn = self.connect()
        with conn.cursor() as cur:
            cur.executemany(
                "UPDATE job_postings SET raw_description = %s WHERE posting_id = %s",
                [(text, pid) for pid, text in updates.items()],
            )
        return len(updates)

    def rows_without_destination(self, *, sources: list[str] | None = None,
                                 limit: int | None = None,
                                 single_vacancy_only: bool = False) -> list[dict[str, Any]]:
        """Single-vacancy postings that have never had a destination resolved.

        DUPLICATES ARE INCLUDED, deliberately. `duplicate_of` is ON DELETE SET
        NULL, so deleting a destination-less canonical PROMOTES its duplicates
        to standalone rows — which would re-create exactly what the prune
        removed. Measured: 45 telegram rows duplicate el7far canonicals. Every
        row needs its own verdict, or the rule leaks through the FK.

        `single_vacancy_only` excludes split children. Their post's links belong
        to the whole roundup, so there is rarely a per-role page — verified on
        the three largest el7far roundups, which link to a careers hub, an ATS
        listing with no job id, and an image. But when postings are about to be
        DELETED for lacking a destination, every row deserves its one attempt
        first, so the prune's tracing pass sets this False.
        """
        conn = self.connect()
        clauses = [
            "final_url IS NULL",
            "status <> 'rejected'",
        ]
        if single_vacancy_only:
            clauses.append("(source_post_url IS NULL OR source_post_url = source_url)")
        params: dict[str, Any] = {"limit": limit}
        if sources:
            clauses.append("source = ANY(%(sources)s)")
            params["sources"] = list(sources)

        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT posting_id, source, source_type, source_url, title,
                       (source_post_url IS NOT NULL AND source_post_url <> source_url)
                           AS is_split
                  FROM job_postings
                 WHERE {' AND '.join(clauses)}
                 -- posting_id breaks the tie, and it is load-bearing. Every
                 -- row touched by one cycle shares a `last_seen_at`, so
                 -- ordering on it alone is arbitrary: two capped runs sample
                 -- different rows and neither makes progress through the
                 -- backlog. Measured — a --limit 20 run found 0 destinations
                 -- while the same query in a probe found 2 in its first 8.
                 ORDER BY last_seen_at DESC, posting_id
                 LIMIT %(limit)s
                """,
                params,
            )
            return [dict(r) for r in cur.fetchall()]

    def update_posting(self, posting_id: str, values: dict[str, Any]) -> None:
        """Set named columns on one row. Column names come from this module's own
        allowlist, never from a caller's keys — a backfill must not be able to
        name a column."""
        allowed = {"final_url", "required_skills", "work_arrangement", "employment_type",
                   "salary_min", "salary_max", "salary_currency", "salary_period",
                   "seniority_level", "location", "country", "company",
                   "destination_status"}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"update_posting refuses unknown columns: {sorted(unknown)}")
        if not values:
            return
        assignments = ", ".join(f"{name} = %({name})s" for name in values)
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE job_postings SET {assignments} WHERE posting_id = %(posting_id)s",
                {**values, "posting_id": posting_id},
            )

    # ------------------------------------------------------------------
    # migrations
    # ------------------------------------------------------------------
    def migrate(self) -> list[str]:
        return apply_migrations(self.dsn)

    def pending_migrations(self) -> list[str]:
        return pending(self.dsn)

    # ------------------------------------------------------------------
    # cycle lock
    # ------------------------------------------------------------------
    @contextmanager
    def cycle_lock(self) -> Iterator[None]:
        """Prevent two cycles running at once.

        Held on a DEDICATED connection for the whole cycle. `pg_try_advisory_lock`
        is session-scoped, so taking it on a connection that is later returned to
        a pool — or closed between transactions — releases it silently and the
        protection stops working with no error.

        Two overlapping cycles would double-increment `missed_cycles` and push
        every posting to stale a full cycle early.
        """
        conn = psycopg.connect(self.dsn)
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT pg_try_advisory_lock(hashtext(%s))", (CYCLE_LOCK_KEY,))
                row = cur.fetchone()
                acquired = bool(row[0]) if row else False
            if not acquired:
                raise LockNotAcquired()
            try:
                yield
            finally:
                with conn.cursor() as cur:
                    cur.execute("SELECT pg_advisory_unlock(hashtext(%s))", (CYCLE_LOCK_KEY,))
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # introspection — used by the phase gate and `--check`
    # ------------------------------------------------------------------
    def health(self) -> dict[str, Any]:
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute("SELECT version() AS server")
            server = (cur.fetchone() or {}).get("server", "")
            cur.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            )
            row = cur.fetchone()
            cur.execute(
                """
                SELECT table_name FROM information_schema.tables
                 WHERE table_schema = 'public' ORDER BY table_name
                """
            )
            tables = [r["table_name"] for r in cur.fetchall()]
            cur.execute(
                """
                SELECT indexname FROM pg_indexes
                 WHERE schemaname = 'public' ORDER BY indexname
                """
            )
            indexes = [r["indexname"] for r in cur.fetchall()]
        return {
            "server": server.split(" on ")[0],
            "pgvector": row["extversion"] if row else None,
            "tables": tables,
            "indexes": indexes,
        }

    def counts(self) -> dict[str, int]:
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT status, count(*) AS n
                  FROM job_postings GROUP BY status ORDER BY status
                """
            )
            by_status = {r["status"]: r["n"] for r in cur.fetchall()}
            cur.execute("SELECT count(*) AS n FROM skill_demand_stats")
            stats = (cur.fetchone() or {}).get("n", 0)
        return {**by_status, "skill_demand_stats": stats}

    # ------------------------------------------------------------------
    # ingestion (phase 4)
    # ------------------------------------------------------------------
    def lookup_hashes(self, posting_ids: list[str]) -> dict[str, str]:
        """Return the content_hash already stored for each id present.

        This is the whole basis of the "second run does nothing" property: a
        posting whose stored hash equals its freshly computed one is unchanged,
        so extraction and embedding are skipped. An id absent from the result is
        new. Empty input returns empty rather than issuing a `= ANY('{}')` that
        scans nothing usefully.
        """
        if not posting_ids:
            return {}
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT posting_id, content_hash FROM job_postings WHERE posting_id = ANY(%s)",
                (posting_ids,),
            )
            return {r["posting_id"]: r["content_hash"] for r in cur.fetchall()}

    def lookup_posts(self, source_post_urls: list[str]) -> dict[str, dict[str, Any]]:
        """Post-level change detection for split roundups.

        Maps each source_post_url to ``{"content_hash": <the post's hash>,
        "posting_ids": [child ids]}``. All rows of one post share a content_hash
        (the POST is the unit of change), so any child's hash answers "has this
        post changed?" without re-extracting the vacancies it split into; the ids
        are what ``touch_seen`` marks when it has not. Absent url -> new post.
        """
        if not source_post_urls:
            return {}
        conn = self.connect()
        out: dict[str, dict[str, Any]] = {}
        with conn.cursor() as cur:
            cur.execute(
                "SELECT source_post_url, content_hash, posting_id "
                "FROM job_postings WHERE source_post_url = ANY(%s)",
                (source_post_urls,),
            )
            for r in cur.fetchall():
                entry = out.setdefault(
                    r["source_post_url"], {"content_hash": r["content_hash"], "posting_ids": []}
                )
                entry["posting_ids"].append(r["posting_id"])
        return out

    def existing_ids(self, posting_ids: list[str]) -> set[str]:
        """Which of these ids already exist — used by link dedup to tell an
        in-cycle canonical from one already persisted."""
        if not posting_ids:
            return set()
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT posting_id FROM job_postings WHERE posting_id = ANY(%s)",
                (posting_ids,),
            )
            return {r["posting_id"] for r in cur.fetchall()}

    def find_by_source_urls(self, urls: list[str]) -> dict[str, str]:
        """Map canonical source_url -> posting_id for any that already exist.

        This is what makes link-based dedup deterministic: a Telegram post links
        to a blog URL, and because source_url is unique the target resolves
        without knowing which source it came from. No embedding involved — the
        link IS the evidence.
        """
        if not urls:
            return {}
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT source_url, posting_id FROM job_postings WHERE source_url = ANY(%s)",
                (urls,),
            )
            return {r["source_url"]: r["posting_id"] for r in cur.fetchall()}

    def find_by_final_urls(self, urls: list[str]) -> dict[str, str]:
        """Map final_url -> posting_id for vacancies we already hold.

        The employer's own URL is a stronger duplicate key than anything we can
        infer: two aggregators writing two different summaries of one Siemens
        vacancy produce two dissimilar texts, so embedding near-dup misses them —
        but they point at the same page, and that is not a coincidence that needs
        a threshold.

        Only canonical rows are eligible as targets; pointing a duplicate at
        another duplicate is what `_flatten_duplicate_chains` exists to undo.
        """
        if not urls:
            return {}
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT final_url, posting_id FROM job_postings "
                "WHERE final_url = ANY(%s) AND duplicate_of IS NULL",
                (urls,),
            )
            return {r["final_url"]: r["posting_id"] for r in cur.fetchall()}

    def touch_seen(self, posting_ids: list[str]) -> int:
        """Mark unchanged postings as seen this cycle without rewriting them.

        Resets the missed-cycle counter and revives a posting that had gone
        stale — being seen again is exactly the evidence that it is still live.
        A rejected or needs_review posting keeps its status: re-seeing a scam
        unchanged is not a reason to trust it.
        """
        if not posting_ids:
            return 0
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE job_postings
                   SET last_seen_at = now(),
                       missed_cycles = 0,
                       status = CASE WHEN status = 'stale' THEN 'active' ELSE status END,
                       stale_since = CASE WHEN status = 'stale' THEN NULL ELSE stale_since END
                 WHERE posting_id = ANY(%s)
                """,
                (posting_ids,),
            )
            return cur.rowcount

    def upsert_batch(self, rows: list[Any]) -> None:
        """Insert or update a batch of postings in ONE transaction.

        Rows are sorted so canonicals (duplicate_of IS NULL) go first. A
        duplicate's FK references its canonical, and when that canonical is also
        new this cycle it must exist before the duplicate inserts — otherwise
        the FK fails part-way and, because it is one transaction, takes the whole
        batch down. Sorting here is cheaper and clearer than deferring the
        constraint.

        On conflict the content, extraction and embedding are overwritten and
        the posting is marked seen; first_seen_at is preserved. embedding is
        written as-is, INCLUDING NULL — a caller must not pass a row with a
        null embedding for a posting it did not intend to clear (the pipeline
        routes unchanged postings through touch_seen, never here).
        """
        if not rows:
            return
        ordered = sorted(rows, key=lambda r: r.duplicate_of is not None)
        conn = self.connect()
        with conn.cursor() as cur:
            for row in ordered:
                cur.execute(_UPSERT_SQL, _row_params(row))

    def find_neardup_candidates(
        self,
        embedding: list[float],
        *,
        recent_days: int,
        limit: int,
        exclude_id: str,
    ) -> list[dict[str, Any]]:
        """Nearest already-stored postings by cosine similarity.

        Three things here are the whole correctness of the feature:

        * ``<=>`` is cosine DISTANCE; similarity is ``1 - distance``. A 0.93
          distance is 0.07 similarity — inverting this is the likeliest bug in
          near-dup detection, so the conversion happens here, once.
        * the two terminal predicates are written VERBATIM to match the partial
          HNSW index; drift and the planner silently stops using it.
        * ``SET LOCAL hnsw.ef_search`` — without raising it, a filtered pgvector
          search can return fewer than ``limit`` rows and miss a real duplicate
          with no error at all. Requires a transaction, which the pipeline holds
          open around the batch.
        """
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute("SET LOCAL hnsw.ef_search = 100")
            cur.execute(
                """
                SELECT posting_id,
                       source_group,
                       1 - (embedding <=> %(emb)s::vector) AS similarity
                  FROM job_postings
                 WHERE embedding IS NOT NULL
                   AND duplicate_of IS NULL
                   AND status <> 'rejected'
                   AND first_seen_at >= now() - make_interval(days => %(recent)s)
                   AND posting_id <> %(self)s
                 ORDER BY embedding <=> %(emb)s::vector
                 LIMIT %(k)s
                """,
                {"emb": embedding, "recent": recent_days, "self": exclude_id, "k": limit},
            )
            return list(cur.fetchall())

    def get_status(self, posting_id: str) -> str | None:
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute("SELECT status FROM job_postings WHERE posting_id = %s", (posting_id,))
            row = cur.fetchone()
            return row["status"] if row else None

    # ------------------------------------------------------------------
    # staleness (phase 5)
    #
    # The staleness rule counts CYCLES, not elapsed time, so a skipped cycle
    # (machine asleep, source blocked for a day) does not age postings that were
    # never actually checked. `missed_cycles` is the explicit counter that makes
    # that possible.
    # ------------------------------------------------------------------
    def age_missed(self, sources: list[str]) -> int:
        """Increment the missed-cycle counter for a source's active inventory.

        MUST run before the pipeline's touch/upsert for the same cycle: this
        blanket-increments every active posting from ``sources``, and the
        pipeline then resets the ones actually seen back to 0. The net effect is
        "+1 for everything the source no longer lists, 0 for everything it still
        does" — without needing to diff two large id sets.

        Scoped to ``sources`` on purpose: only sources that fetched cleanly this
        cycle may age their inventory. A blocked or partially-fetched source must
        NOT be passed here, or a 429 would push its real, still-live jobs toward
        deletion. That gate (healthy_sources) lives in the run-log node.
        """
        if not sources:
            return 0
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE job_postings SET missed_cycles = missed_cycles + 1 "
                "WHERE status = 'active' AND source = ANY(%s)",
                (sources,),
            )
            return cur.rowcount

    def mark_stale(self, *, threshold: int) -> int:
        """Move postings past the missed-cycle threshold to 'stale'.

        ``stale_since`` is set once and left alone on re-runs (COALESCE), so the
        60-day prune clock measures continuous staleness, not the last time this
        ran.
        """
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE job_postings
                   SET status = 'stale',
                       stale_since = COALESCE(stale_since, now())
                 WHERE status = 'active' AND missed_cycles >= %s
                """,
                (threshold,),
            )
            return cur.rowcount

    def prune(self, *, older_than_days: int) -> int:
        """Hard-delete postings that have been stale longer than the limit.

        ``WHERE status = 'stale'`` ONLY. A rejected posting is retained forever
        as audit evidence and must never be swept up here — asserted in a test.
        Deleting a canonical sets its duplicates' ``duplicate_of`` to NULL
        (ON DELETE SET NULL), promoting them back to standalone, which is exactly
        true once the canonical is gone.
        """
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM job_postings "
                "WHERE status = 'stale' AND stale_since < now() - make_interval(days => %s)",
                (older_than_days,),
            )
            return cur.rowcount

    # ------------------------------------------------------------------
    # aggregation (phase 5) — see recompute_stats in aggregate.py
    # ------------------------------------------------------------------
    def replace_stats_window(self, *, sql: str, params: dict[str, Any]) -> int:
        """REPLACE one window's stats: write this run's rows, drop what it did not
        produce. Returns rows written.

        The SQL itself lives in ``aggregate.py`` beside the reasoning about what
        it counts, rather than as one more opaque string in this module; this
        method exists only so that aggregate.py does not open its own cursor and
        the "all SQL executes through the store" boundary holds.

        The delete is why this is called *replace*. The upsert alone left behind
        any ``(sector, skill_key)`` a previous run of the same window had produced
        and this one no longer supports — measured on the live corpus: **114 rows
        (117 demand units) whose sample_postings cited URLs that no longer
        resolve**, still being weighted by Agent C. A window is a statement about
        the corpus as it stands now, so recomputing it has to be able to say
        "that skill is no longer in evidence", and only a delete can say that.

        Clear-then-insert, rather than stamping rows and deleting the ones with
        an older ``computed_at``. The stamped version was correct, but only just:
        inside a transaction Postgres' ``now()`` is the TRANSACTION START time, so
        the rows this run writes carry the same timestamp it compares against and
        the whole thing rests on the comparison being a strict ``<``. Agent D
        made exactly that mistake and deleted the rows it had just written. There
        is no version of the ordering below with a subtlety to get wrong.
        """
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM skill_demand_stats WHERE window_end = %(w_end)s",
                {"w_end": params["w_end"]},
            )
            cur.execute(sql, params)
            return cur.rowcount

    def prune_stats_windows(self, *, keep: int) -> int:
        """Drop aggregation snapshots older than the newest ``keep`` windows.

        Replacement within a window was already handled; nothing pruned ACROSS
        them, so every `window_end = (SELECT max(window_end) …)` a consumer runs
        scanned an ever-growing pile of superseded history — measured at 4
        windows / 4,120 rows serving 1,139 current.

        The caller passes ``Config.stats_windows_to_keep()``, which floors this
        at 2; dropping to 1 would leave `prior_frequency_count` with no prior
        window to read and silently restore the fabricated-trend bug.
        """
        if keep < 2:
            raise ValueError(
                f"stats retention of {keep} would leave no prior window for the trend "
                "calculation; the floor is 2 (see Config.stats_windows_to_keep)")
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM skill_demand_stats
                 WHERE window_end < (
                    SELECT min(window_end) FROM (
                        SELECT DISTINCT window_end FROM skill_demand_stats
                         ORDER BY window_end DESC LIMIT %(keep)s
                    ) recent
                 )
                """,
                {"keep": keep},
            )
            return cur.rowcount

    def scalar(self, sql: str, params: dict[str, Any] | None = None) -> Any:
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(sql, params or {})
            row = cur.fetchone()
            return next(iter(row.values())) if row else None

    # ------------------------------------------------------------------
    # source health (phase 6)
    #
    # Read and written ONLY here and in the runlog node that calls it. Nothing
    # under sources/ imports this table, so no code path exists that could react
    # to a recorded failure by retrying harder — degradation is a fact reported
    # to a human, never an input to the scraper's own behaviour. A test asserts
    # the import never appears.
    # ------------------------------------------------------------------
    def record_source_health(
        self,
        source: str,
        *,
        success: bool,
        error: str | None,
        degraded_after: int,
    ) -> dict[str, Any]:
        """Update one source's cycle history and return the resulting row.

        ``degraded_since`` is stamped when ``consecutive_failures`` first reaches
        ``degraded_after`` and left in place across further failures — it marks
        WHEN degradation began, so "degraded for more than N days" is answerable.
        A success clears it: recovery is recovery.
        """
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO source_health (
                    source, consecutive_failures, last_success_at, last_attempt_at,
                    last_error, degraded_since, total_cycles
                )
                VALUES (
                    %(source)s,
                    CASE WHEN %(success)s THEN 0 ELSE 1 END,
                    CASE WHEN %(success)s THEN now() ELSE NULL END,
                    now(),
                    %(error)s,
                    NULL,
                    1
                )
                ON CONFLICT (source) DO UPDATE SET
                    consecutive_failures =
                        CASE WHEN %(success)s THEN 0
                             ELSE source_health.consecutive_failures + 1 END,
                    last_success_at =
                        CASE WHEN %(success)s THEN now()
                             ELSE source_health.last_success_at END,
                    last_attempt_at = now(),
                    last_error = CASE WHEN %(success)s THEN NULL ELSE %(error)s END,
                    degraded_since =
                        CASE
                            WHEN %(success)s THEN NULL
                            WHEN source_health.consecutive_failures + 1 >= %(degraded_after)s
                                 THEN COALESCE(source_health.degraded_since, now())
                            ELSE source_health.degraded_since
                        END,
                    total_cycles = source_health.total_cycles + 1
                RETURNING source, consecutive_failures, last_success_at, last_attempt_at,
                          last_error, degraded_since, total_cycles
                """,
                {
                    "source": source,
                    "success": success,
                    "error": error,
                    "degraded_after": degraded_after,
                },
            )
            return dict(cur.fetchone())

    def get_source_health(self, source: str) -> dict[str, Any] | None:
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM source_health WHERE source = %s", (source,))
            row = cur.fetchone()
            return dict(row) if row else None

    # ------------------------------------------------------------------
    # ESCO mapping layer
    # ------------------------------------------------------------------
    def upsert_esco_skills(self, rows: list[dict[str, Any]], *, version: str) -> int:
        conn = self.connect()
        with conn.cursor() as cur:
            for row in rows:
                cur.execute(
                    """
                    INSERT INTO esco_skills (esco_uri, preferred_label, skill_type,
                                             description, esco_version)
                    VALUES (%(esco_uri)s, %(preferred_label)s, %(skill_type)s,
                            %(description)s, %(version)s)
                    ON CONFLICT (esco_uri) DO UPDATE SET
                        preferred_label = EXCLUDED.preferred_label,
                        skill_type = EXCLUDED.skill_type,
                        description = EXCLUDED.description,
                        esco_version = EXCLUDED.esco_version,
                        loaded_at = now()
                    """,
                    {**row, "version": version},
                )
        return len(rows)

    def insert_esco_labels(self, rows: list[dict[str, Any]]) -> int:
        """Labels are inserted WITHOUT embeddings; the sync's embedding pass
        fills the NULL ones afterwards. That split is what makes an interrupted
        sync resumable instead of restarted."""
        conn = self.connect()
        inserted = 0
        with conn.cursor() as cur:
            for row in rows:
                cur.execute(
                    """
                    INSERT INTO esco_labels (label_key, label, esco_uri, is_preferred)
                    VALUES (%(label_key)s, %(label)s, %(esco_uri)s, %(is_preferred)s)
                    ON CONFLICT (label_key, esco_uri) DO NOTHING
                    """,
                    row,
                )
                inserted += cur.rowcount
        return inserted

    def labels_missing_embeddings(self, *, limit: int) -> list[dict[str, Any]]:
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT label_key, esco_uri, label FROM esco_labels "
                "WHERE embedding IS NULL ORDER BY label_key LIMIT %s",
                (limit,),
            )
            return [dict(r) for r in cur.fetchall()]

    def set_label_embeddings(self, entries: list[tuple[str, str, list[float]]]) -> None:
        conn = self.connect()
        with conn.cursor() as cur:
            for label_key, esco_uri, embedding in entries:
                cur.execute(
                    "UPDATE esco_labels SET embedding = %s "
                    "WHERE label_key = %s AND esco_uri = %s",
                    (embedding, label_key, esco_uri),
                )

    def esco_status(self) -> dict[str, Any]:
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT (SELECT count(*) FROM esco_skills) AS skills,
                       (SELECT count(*) FROM esco_labels) AS labels,
                       (SELECT count(*) FROM esco_labels WHERE embedding IS NOT NULL) AS embedded,
                       (SELECT max(esco_version) FROM esco_skills) AS version,
                       (SELECT count(*) FROM skill_esco_map) AS mapped_keys
                """
            )
            return dict(cur.fetchone())

    def get_esco_version(self) -> str | None:
        return self.scalar("SELECT max(esco_version) FROM esco_skills")

    def find_esco_by_labels(self, label_keys: list[str]) -> dict[str, dict[str, Any]]:
        """Exact lexical lookup. When one key matches several labels, a
        preferred label beats an alternative — DISTINCT ON with that ordering,
        so the choice is deterministic rather than row-order luck."""
        if not label_keys:
            return {}
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT ON (label_key)
                       label_key, esco_uri, is_preferred
                  FROM esco_labels
                 WHERE label_key = ANY(%s)
                 ORDER BY label_key, is_preferred DESC, esco_uri
                """,
                (label_keys,),
            )
            return {r["label_key"]: dict(r) for r in cur.fetchall()}

    def nearest_esco_label(self, embedding: list[float], *, limit: int = 3) -> list[dict[str, Any]]:
        """Nearest labels by cosine. Same two rules as the near-dup search:
        ``1 - <=>`` is the similarity, and ef_search is raised so a filtered
        HNSW scan cannot silently return fewer candidates than asked. Requires
        the surrounding transaction the mapper runs in."""
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute("SET LOCAL hnsw.ef_search = 100")
            cur.execute(
                """
                SELECT esco_uri, label,
                       1 - (embedding <=> %(emb)s::vector) AS similarity
                  FROM esco_labels
                 WHERE embedding IS NOT NULL
                 ORDER BY embedding <=> %(emb)s::vector
                 LIMIT %(k)s
                """,
                {"emb": embedding, "k": limit},
            )
            return [dict(r) for r in cur.fetchall()]

    def upsert_skill_map(self, entries: list[dict[str, Any]]) -> None:
        conn = self.connect()
        with conn.cursor() as cur:
            for entry in entries:
                cur.execute(
                    """
                    INSERT INTO skill_esco_map (skill_key, esco_uri, method,
                                                similarity, esco_version)
                    VALUES (%(skill_key)s, %(esco_uri)s, %(method)s,
                            %(similarity)s, %(esco_version)s)
                    ON CONFLICT (skill_key) DO UPDATE SET
                        esco_uri = EXCLUDED.esco_uri,
                        method = EXCLUDED.method,
                        similarity = EXCLUDED.similarity,
                        esco_version = EXCLUDED.esco_version,
                        mapped_at = now()
                    """,
                    entry,
                )

    def pending_skill_keys(self, *, version: str) -> list[str]:
        """Skill keys that need mapping this cycle.

        The eligibility predicate matches the aggregation's (minus the country
        clause — mapping an out-of-scope skill is harmless, missing an in-scope
        one is not), so everything the stats table will contain is covered.
        A key already mapped stays mapped; a key recorded 'unmapped' is retried
        only when the taxonomy version has changed — the one event that can turn
        a miss into a hit.
        """
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH keys AS (
                    SELECT DISTINCT lower(btrim(s)) AS skill_key
                      FROM job_postings, unnest(required_skills) AS s
                     WHERE status = 'active'
                       AND duplicate_of IS NULL
                       AND sector IS NOT NULL
                       AND listing_intent = 'vacancy'
                       AND poster_type = 'company'
                       AND btrim(s) <> ''
                )
                SELECT k.skill_key
                  FROM keys k
                  LEFT JOIN skill_esco_map m ON m.skill_key = k.skill_key
                 WHERE m.skill_key IS NULL
                    OR (m.method = 'unmapped' AND m.esco_version <> %(version)s)
                 ORDER BY k.skill_key
                """,
                {"version": version},
            )
            return [r["skill_key"] for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # operator commands (phase 7)
    # ------------------------------------------------------------------
    def purge_source(self, source: str) -> int:
        """Delete every posting from a decommissioned source.

        Never called automatically — a scraping bug must not be able to trigger
        mass deletion, so this is only reachable through the explicit
        ``--purge-source`` flag. Duplicates in OTHER sources that pointed at this
        source's canonicals are promoted to standalone by ON DELETE SET NULL,
        not deleted with it.
        """
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM job_postings WHERE source = %s", (source,))
            return cur.rowcount

    def destination_survey(self, sources: list[str]) -> list[dict[str, Any]]:
        """What the prune would remove, and why, before it removes anything.

        481 rows is far past the point where a bare number is enough to approve
        a deletion, so this reports the breakdown by `destination_status` and by
        source. Read-only.
        """
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT source,
                       COALESCE(destination_status, 'never traced') AS status,
                       final_url IS NOT NULL AS has_destination,
                       count(*) AS rows
                  FROM job_postings
                 WHERE source = ANY(%s)
                 GROUP BY 1, 2, 3
                 ORDER BY 1, 4 DESC
                """,
                (list(sources),),
            )
            return [dict(r) for r in cur.fetchall()]

    def prune_without_destination(self, sources: list[str]) -> int:
        """Delete postings from these sources that reach no employer page.

        Scoped to NAMED sources, never "all". A deletion this size must say
        whose rows it is taking, and a source omitted from the list cannot be
        caught by a widening bug — which is why GulfTalent is safe here by
        construction rather than by an exclusion clause someone could edit.

        Deletes only rows we actually LOOKED AT: `destination_status IS NOT
        NULL` and not one of the resolved values. A row nobody traced is not
        evidence of anything, and must survive to be traced later.

        `duplicate_of` is ON DELETE SET NULL, so a duplicate of a deleted
        canonical is promoted to standalone rather than orphaned or removed with
        it. Callers MUST recompute `skill_demand_stats` afterwards: its
        `sample_postings` carry posting ids and would otherwise cite rows that
        no longer exist.
        """
        if not sources:
            return 0
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM job_postings
                 WHERE source = ANY(%s)
                   AND final_url IS NULL
                   AND destination_status IS NOT NULL
                   AND destination_status NOT IN ('resolved', 'source_is_destination')
                """,
                (list(sources),),
            )
            return cur.rowcount

    def sample_for_labelling(self, n: int) -> list[dict[str, Any]]:
        """A random sample of ingested postings for a human to label.

        Used to MEASURE the legitimacy filter's precision against reality — the
        one number the rules cannot self-report, because a wrongly rejected
        posting is excluded from the very stats you would notice it in. The
        export is gitignored: it is real ingested data, not a fixture, and a
        realistic scam posting must never live in the repo.
        """
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT posting_id, source, title, raw_description,
                       legitimacy_score, status, review_reason
                  FROM job_postings
                 ORDER BY random()
                 LIMIT %s
                """,
                (n,),
            )
            return [dict(r) for r in cur.fetchall()]


# The one write statement, kept beside the store so every column the pipeline
# populates is visible in one place. ON CONFLICT preserves first_seen_at and the
# staleness lifecycle is reset because a posting present in the batch was, by
# definition, seen this cycle.
_UPSERT_SQL = """
INSERT INTO job_postings (
    posting_id, source, source_group, source_type, source_url, source_post_url,
    title, raw_description, content_hash,
    status, sector, required_skills, company, seniority_level, location, country, posted_date,
    final_url, work_arrangement, employment_type,
    salary_min, salary_max, salary_currency, salary_period,
    attribution, terms_url, expires_at, destination_status,
    legitimacy_score, review_reason, duplicate_of,
    listing_intent, poster_type, extraction_model, embedding,
    first_seen_at, last_seen_at, missed_cycles, stale_since
) VALUES (
    %(posting_id)s, %(source)s, %(source_group)s, %(source_type)s, %(source_url)s,
    %(source_post_url)s,
    %(title)s, %(raw_description)s, %(content_hash)s,
    %(status)s, %(sector)s, %(required_skills)s, %(company)s, %(seniority_level)s, %(location)s,
    %(country)s, %(posted_date)s,
    %(final_url)s, %(work_arrangement)s, %(employment_type)s,
    %(salary_min)s, %(salary_max)s, %(salary_currency)s, %(salary_period)s,
    %(attribution)s, %(terms_url)s, %(expires_at)s, %(destination_status)s,
    %(legitimacy_score)s, %(review_reason)s, %(duplicate_of)s,
    %(listing_intent)s, %(poster_type)s, %(extraction_model)s, %(embedding)s,
    now(), now(), 0, NULL
)
ON CONFLICT (posting_id) DO UPDATE SET
    source = EXCLUDED.source,
    source_group = EXCLUDED.source_group,
    source_type = EXCLUDED.source_type,
    source_url = EXCLUDED.source_url,
    source_post_url = EXCLUDED.source_post_url,
    title = EXCLUDED.title,
    raw_description = EXCLUDED.raw_description,
    content_hash = EXCLUDED.content_hash,
    status = EXCLUDED.status,
    sector = EXCLUDED.sector,
    required_skills = EXCLUDED.required_skills,
    company = EXCLUDED.company,
    seniority_level = EXCLUDED.seniority_level,
    location = EXCLUDED.location,
    country = EXCLUDED.country,
    posted_date = EXCLUDED.posted_date,
    -- COALESCE, not EXCLUDED: a later cycle that could not reach the employer
    -- page must not erase what an earlier one learned from it. Same rule the
    -- course refresh follows for volatile columns — absence of an observation is
    -- not an observation of absence.
    final_url = COALESCE(EXCLUDED.final_url, job_postings.final_url),
    work_arrangement = COALESCE(EXCLUDED.work_arrangement, job_postings.work_arrangement),
    employment_type = COALESCE(EXCLUDED.employment_type, job_postings.employment_type),
    salary_min = COALESCE(EXCLUDED.salary_min, job_postings.salary_min),
    salary_max = COALESCE(EXCLUDED.salary_max, job_postings.salary_max),
    salary_currency = COALESCE(EXCLUDED.salary_currency, job_postings.salary_currency),
    salary_period = COALESCE(EXCLUDED.salary_period, job_postings.salary_period),
    -- Attribution is EXCLUDED, not COALESCE: it is a fact about the SOURCE, not
    -- an observation about the posting, so the current cycle's value is always
    -- the right one. If we rename how a publisher is credited, every row it owns
    -- must follow on the next cycle — stale attribution is the one kind this
    -- table must not carry.
    attribution = EXCLUDED.attribution,
    terms_url = EXCLUDED.terms_url,
    expires_at = COALESCE(EXCLUDED.expires_at, job_postings.expires_at),
    -- COALESCE: a cycle that did not trace must not erase what a trace learned.
    destination_status = COALESCE(EXCLUDED.destination_status,
                                  job_postings.destination_status),
    legitimacy_score = EXCLUDED.legitimacy_score,
    review_reason = EXCLUDED.review_reason,
    duplicate_of = EXCLUDED.duplicate_of,
    listing_intent = EXCLUDED.listing_intent,
    poster_type = EXCLUDED.poster_type,
    extraction_model = EXCLUDED.extraction_model,
    embedding = EXCLUDED.embedding,
    last_seen_at = now(),
    missed_cycles = 0,
    stale_since = NULL
"""


def _row_params(row: Any) -> dict[str, Any]:
    return {
        "posting_id": row.posting_id,
        "source": row.source,
        "source_group": row.source_group,
        "source_type": row.source_type,
        "source_url": row.source_url,
        # Empty means "an ordinary single-vacancy posting" -> the post IS the row,
        # so its post-url is its own source_url.
        "source_post_url": row.source_post_url or row.source_url,
        "title": row.title,
        "raw_description": row.raw_description,
        "content_hash": row.content_hash,
        "status": row.status,
        "sector": row.sector,
        "required_skills": row.required_skills,
        "company": row.company,
        "final_url": row.final_url,
        "work_arrangement": row.work_arrangement,
        "employment_type": row.employment_type,
        "salary_min": row.salary_min,
        "salary_max": row.salary_max,
        "salary_currency": row.salary_currency,
        "salary_period": row.salary_period,
        "attribution": row.attribution,
        "terms_url": row.terms_url,
        "expires_at": row.expires_at,
        "destination_status": row.destination_status,
        "seniority_level": row.seniority_level,
        "location": row.location,
        "country": row.country,
        "posted_date": row.posted_date,
        "legitimacy_score": row.legitimacy_score,
        "review_reason": row.review_reason,
        "duplicate_of": row.duplicate_of,
        "listing_intent": row.listing_intent,
        "poster_type": row.poster_type,
        "extraction_model": row.extraction_model,
        "embedding": row.embedding,
    }
