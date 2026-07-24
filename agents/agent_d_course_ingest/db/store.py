"""Database access for Agent D.

**Every SQL string in this project lives in this module.** Nodes call methods;
they never build SQL — the same rule Agent B follows. This is the supply-side
mirror of ``JobStore``, adapted to the ``courses`` / ``skill_supply_stats`` /
``course_esco_map`` / ``course_source_health`` tables.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import psycopg
from psycopg.rows import dict_row

from shared.config import Config

from .migrate import apply_migrations, pending

# Distinct from Agent B's lock key so the two agents' cycles never block each
# other — they touch different tables and can run concurrently.
CYCLE_LOCK_KEY = "itqan_agent_d_cycle"


def to_float_list(value: Any) -> list[float] | None:
    """Normalize pgvector's ``Vector`` (no len(), object dtype under np.asarray)
    into plain floats, so a library type never reaches a node."""
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
    key: str = CYCLE_LOCK_KEY


class CourseStore:
    """Owns the connection to Postgres. Single connection, not a pool: Agent D
    is a batch job — one process, one cycle, then exit."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._conn: psycopg.Connection | None = None

    @classmethod
    def from_config(cls, config: Config) -> "CourseStore":
        return cls(config.require_database_url())

    def connect(self) -> psycopg.Connection:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self.dsn, row_factory=dict_row)
            self._register_vector(self._conn)
        return self._conn

    @staticmethod
    def _register_vector(conn: psycopg.Connection) -> None:
        try:
            from pgvector.psycopg import register_vector

            register_vector(conn)
        except Exception:
            pass

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
        self._conn = None

    def __enter__(self) -> "CourseStore":
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[psycopg.Connection]:
        conn = self.connect()
        with conn.transaction():
            yield conn

    # ------------------------------------------------------------------
    # migrations
    # ------------------------------------------------------------------
    def migrate(self) -> list[str]:
        return apply_migrations(self.dsn)

    def pending_migrations(self) -> list[str]:
        return pending(self.dsn)

    # ------------------------------------------------------------------
    # cycle lock (dedicated connection, held for the whole cycle)
    # ------------------------------------------------------------------
    @contextmanager
    def cycle_lock(self) -> Iterator[None]:
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
    # introspection
    # ------------------------------------------------------------------
    def health(self) -> dict[str, Any]:
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute("SELECT version() AS server")
            server = (cur.fetchone() or {}).get("server", "")
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            row = cur.fetchone()
            cur.execute(
                "SELECT to_regclass('public.courses') AS c, "
                "to_regclass('public.skill_supply_stats') AS s, "
                "to_regclass('public.esco_labels') AS esco"
            )
            tables = cur.fetchone() or {}
        return {
            "server": server.split(" on ")[0],
            "pgvector": row["extversion"] if row else None,
            "courses_table": tables.get("c") is not None,
            "supply_table": tables.get("s") is not None,
            "esco_taxonomy": tables.get("esco") is not None,
        }

    def counts(self) -> dict[str, int]:
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute("SELECT status, count(*) AS n FROM courses GROUP BY status ORDER BY status")
            by_status = {r["status"]: r["n"] for r in cur.fetchall()}
            cur.execute("SELECT count(*) AS n FROM skill_supply_stats")
            stats = (cur.fetchone() or {}).get("n", 0)
        return {**by_status, "skill_supply_stats": stats}

    # ------------------------------------------------------------------
    # ingestion
    # ------------------------------------------------------------------
    def lookup_hashes(self, course_ids: list[str]) -> dict[str, str]:
        if not course_ids:
            return {}
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT course_id, content_hash FROM courses WHERE course_id = ANY(%s)",
                (course_ids,),
            )
            return {r["course_id"]: r["content_hash"] for r in cur.fetchall()}

    def find_by_source_urls(self, urls: list[str]) -> dict[str, str]:
        if not urls:
            return {}
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT source_url, course_id FROM courses WHERE source_url = ANY(%s)",
                (urls,),
            )
            return {r["source_url"]: r["course_id"] for r in cur.fetchall()}

    def touch_seen(self, course_ids: list[str]) -> int:
        if not course_ids:
            return 0
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE courses
                   SET last_seen_at = now(), missed_cycles = 0,
                       status = CASE WHEN status = 'stale' THEN 'active' ELSE status END,
                       stale_since = CASE WHEN status = 'stale' THEN NULL ELSE stale_since END
                 WHERE course_id = ANY(%s)
                """,
                (course_ids,),
            )
            return cur.rowcount

    def upsert_batch(self, rows: list[Any]) -> None:
        if not rows:
            return
        ordered = sorted(rows, key=lambda r: r.duplicate_of is not None)
        conn = self.connect()
        with conn.cursor() as cur:
            for row in ordered:
                cur.execute(_UPSERT_SQL, _row_params(row))

    def find_neardup_candidates(
        self, embedding: list[float], *, recent_days: int, limit: int, exclude_id: str
    ) -> list[dict[str, Any]]:
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute("SET LOCAL hnsw.ef_search = 100")
            cur.execute(
                """
                SELECT course_id, source_group,
                       1 - (embedding <=> %(emb)s::vector) AS similarity
                  FROM courses
                 WHERE embedding IS NOT NULL
                   AND duplicate_of IS NULL
                   AND status <> 'rejected'
                   AND first_seen_at >= now() - make_interval(days => %(recent)s)
                   AND course_id <> %(self)s
                 ORDER BY embedding <=> %(emb)s::vector
                 LIMIT %(k)s
                """,
                {"emb": embedding, "recent": recent_days, "self": exclude_id, "k": limit},
            )
            return list(cur.fetchall())

    def get_status(self, course_id: str) -> str | None:
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute("SELECT status FROM courses WHERE course_id = %s", (course_id,))
            row = cur.fetchone()
            return row["status"] if row else None

    # ------------------------------------------------------------------
    # staleness
    # ------------------------------------------------------------------
    def age_missed(self, sources: list[str]) -> int:
        if not sources:
            return 0
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE courses SET missed_cycles = missed_cycles + 1 "
                "WHERE status = 'active' AND source = ANY(%s)",
                (sources,),
            )
            return cur.rowcount

    def mark_stale(self, *, threshold: int) -> int:
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE courses SET status = 'stale', stale_since = COALESCE(stale_since, now()) "
                "WHERE status = 'active' AND missed_cycles >= %s",
                (threshold,),
            )
            return cur.rowcount

    def prune(self, *, older_than_days: int) -> int:
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM courses "
                "WHERE status = 'stale' AND stale_since < now() - make_interval(days => %s)",
                (older_than_days,),
            )
            return cur.rowcount

    # ------------------------------------------------------------------
    # ESCO mapping cache (course_esco_map — Agent D's own, never skill_esco_map)
    # ------------------------------------------------------------------
    def pending_skill_keys(self, *, version: str) -> list[str]:
        """Course taught-skill keys needing a mapping this cycle: eligible
        (active, canonical) courses' skills not yet mapped, or previously
        'unmapped' but the taxonomy version has since changed."""
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH keys AS (
                    SELECT DISTINCT lower(btrim(s)) AS skill_key
                      FROM courses, unnest(taught_skills) AS s
                     WHERE status = 'active' AND duplicate_of IS NULL AND btrim(s) <> ''
                )
                SELECT k.skill_key FROM keys k
                  LEFT JOIN course_esco_map m ON m.skill_key = k.skill_key
                 WHERE m.skill_key IS NULL
                    OR (m.method = 'unmapped' AND m.esco_version <> %(version)s)
                 ORDER BY k.skill_key
                """,
                {"version": version},
            )
            return [r["skill_key"] for r in cur.fetchall()]

    def upsert_course_map(self, entries: list[dict[str, Any]]) -> None:
        conn = self.connect()
        with conn.cursor() as cur:
            for e in entries:
                cur.execute(
                    """
                    INSERT INTO course_esco_map (skill_key, esco_uri, method, similarity, esco_version)
                    VALUES (%(skill_key)s, %(esco_uri)s, %(method)s, %(similarity)s, %(esco_version)s)
                    ON CONFLICT (skill_key) DO UPDATE SET
                        esco_uri = EXCLUDED.esco_uri, method = EXCLUDED.method,
                        similarity = EXCLUDED.similarity, esco_version = EXCLUDED.esco_version,
                        mapped_at = now()
                    """,
                    e,
                )

    def get_esco_version(self) -> str | None:
        return self.scalar("SELECT max(esco_version) FROM esco_skills")

    # ------------------------------------------------------------------
    # aggregation (SQL lives in aggregate.py; executed through here)
    # ------------------------------------------------------------------
    def replace_stats_window(self, *, sql: str, params: dict[str, Any]) -> int:
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.rowcount

    def scalar(self, sql: str, params: dict[str, Any] | None = None) -> Any:
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(sql, params or {})
            row = cur.fetchone()
            return next(iter(row.values())) if row else None

    # ------------------------------------------------------------------
    # source health
    # ------------------------------------------------------------------
    def record_source_health(
        self, source: str, *, success: bool, error: str | None, degraded_after: int
    ) -> dict[str, Any]:
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO course_source_health (
                    source, consecutive_failures, last_success_at, last_attempt_at,
                    last_error, degraded_since, total_cycles
                )
                VALUES (%(source)s, CASE WHEN %(success)s THEN 0 ELSE 1 END,
                        CASE WHEN %(success)s THEN now() ELSE NULL END, now(),
                        %(error)s, NULL, 1)
                ON CONFLICT (source) DO UPDATE SET
                    consecutive_failures = CASE WHEN %(success)s THEN 0
                        ELSE course_source_health.consecutive_failures + 1 END,
                    last_success_at = CASE WHEN %(success)s THEN now()
                        ELSE course_source_health.last_success_at END,
                    last_attempt_at = now(),
                    last_error = CASE WHEN %(success)s THEN NULL ELSE %(error)s END,
                    degraded_since = CASE
                        WHEN %(success)s THEN NULL
                        WHEN course_source_health.consecutive_failures + 1 >= %(degraded_after)s
                            THEN COALESCE(course_source_health.degraded_since, now())
                        ELSE course_source_health.degraded_since END,
                    total_cycles = course_source_health.total_cycles + 1
                RETURNING source, consecutive_failures, last_success_at, last_attempt_at,
                          last_error, degraded_since, total_cycles
                """,
                {"source": source, "success": success, "error": error,
                 "degraded_after": degraded_after},
            )
            return dict(cur.fetchone())

    def get_source_health(self, source: str) -> dict[str, Any] | None:
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM course_source_health WHERE source = %s", (source,))
            row = cur.fetchone()
            return dict(row) if row else None

    # ------------------------------------------------------------------
    # operator commands
    # ------------------------------------------------------------------
    def purge_source(self, source: str) -> int:
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM courses WHERE source = %s", (source,))
            return cur.rowcount


_UPSERT_SQL = """
INSERT INTO courses (
    course_id, source, source_group, source_type, source_url,
    name, raw_description, content_hash, status,
    taught_skills, provider, level, primary_language, subject, country,
    review_reason, duplicate_of, attribution, license, extraction_model, embedding,
    first_seen_at, last_seen_at, missed_cycles, stale_since
) VALUES (
    %(course_id)s, %(source)s, %(source_group)s, %(source_type)s, %(source_url)s,
    %(name)s, %(raw_description)s, %(content_hash)s, %(status)s,
    %(taught_skills)s, %(provider)s, %(level)s, %(primary_language)s, %(subject)s, %(country)s,
    %(review_reason)s, %(duplicate_of)s, %(attribution)s, %(license)s, %(extraction_model)s,
    %(embedding)s, now(), now(), 0, NULL
)
ON CONFLICT (course_id) DO UPDATE SET
    source = EXCLUDED.source, source_group = EXCLUDED.source_group,
    source_type = EXCLUDED.source_type, source_url = EXCLUDED.source_url,
    name = EXCLUDED.name, raw_description = EXCLUDED.raw_description,
    content_hash = EXCLUDED.content_hash, status = EXCLUDED.status,
    taught_skills = EXCLUDED.taught_skills, provider = EXCLUDED.provider,
    level = EXCLUDED.level, primary_language = EXCLUDED.primary_language,
    subject = EXCLUDED.subject, country = EXCLUDED.country,
    review_reason = EXCLUDED.review_reason, duplicate_of = EXCLUDED.duplicate_of,
    attribution = EXCLUDED.attribution, license = EXCLUDED.license,
    extraction_model = EXCLUDED.extraction_model, embedding = EXCLUDED.embedding,
    last_seen_at = now(), missed_cycles = 0, stale_since = NULL
"""


def _row_params(row: Any) -> dict[str, Any]:
    return {
        "course_id": row.course_id, "source": row.source, "source_group": row.source_group,
        "source_type": row.source_type, "source_url": row.source_url,
        "name": row.name, "raw_description": row.raw_description, "content_hash": row.content_hash,
        "status": row.status, "taught_skills": row.taught_skills, "provider": row.provider,
        "level": row.level, "primary_language": row.primary_language, "subject": row.subject,
        "country": row.country, "review_reason": row.review_reason,
        "duplicate_of": row.duplicate_of, "attribution": row.attribution, "license": row.license,
        "extraction_model": row.extraction_model, "embedding": row.embedding,
    }
