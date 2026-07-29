"""The course-market read surface — how Agent E sees Agent D's tables.

Read-only by construction (nothing here writes a row, nothing calls an LLM), the
supply-side mirror of ``shared.job_market``. It exists because the architecture
forbids Agent E from importing ``agents.agent_d_course_ingest.*`` (agents share
nothing but ``shared/``), so the ``courses`` table plus its ESCO map need a
published surface here.

As on the job side, the eligibility predicate is defined HERE and imported back
by Agent D's ``aggregate.py`` — ``agent -> shared`` is the only import direction
the architecture allows, and single-sourcing it means retrieval and aggregation
can never drift on what "a real, recommendable course" means.

Retrieval is by ESCO code (the canonical vocabulary Agents B/C/D all share) with
an exact normalized-``skill_key`` fallback for the missing skills that never
mapped to a concept — an EXACT string match only, never a fuzzy/embedding guess,
because an unmapped skill is precisely the case where guessing does harm.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Optional

from .config import Config
from .contracts import CourseCandidate, CoursePrice
from .skill_match import covers_skill, tokens

# ---------------------------------------------------------------------------
# The eligibility predicate — single source of truth (imported by aggregate.py).
#
# A course is recommendable if it is active and not a duplicate of another. That
# is deliberately simpler than the job side (no country/intent/poster clauses):
# a course has no geography or hiring intent, and the vetted-source ingestion
# already did the fraud filtering the job board could not assume.
# ---------------------------------------------------------------------------
AGGREGABLE_COURSE_PREDICATE = "status = 'active' AND duplicate_of IS NULL"


# ---------------------------------------------------------------------------
# connection plumbing (local: shared/ cannot import Agent D's CourseStore)
# ---------------------------------------------------------------------------
@contextmanager
def _connection(config: Config) -> Iterator[Any]:
    import psycopg
    from psycopg.rows import dict_row

    conn = psycopg.connect(config.require_database_url(), row_factory=dict_row)
    try:
        yield conn
    finally:
        conn.close()


# The two selects share a column list and a row->CourseCandidate mapping; only
# the join/filter differ (esco concept vs exact normalized skill_key).
_COURSE_COLUMNS = """
    c.course_id, c.name, c.provider, c.source_url, c.taught_skills,
    c.rating, c.review_count, c.enrollment_count, c.level, c.last_updated,
    c.price_amount, c.price_currency, c.price_is_free
"""

_BY_ESCO_SQL = f"""
SELECT DISTINCT m.esco_uri AS matched, {_COURSE_COLUMNS}
  FROM courses c
  JOIN LATERAL unnest(c.taught_skills) AS ts(skill) ON true
  JOIN course_esco_map m ON m.skill_key = lower(btrim(ts.skill)) AND m.esco_uri IS NOT NULL
 WHERE {AGGREGABLE_COURSE_PREDICATE}
   AND m.esco_uri = ANY(%(codes)s)
 ORDER BY matched, c.course_id
"""

# The unmapped fallback. Postgres narrows to courses sharing at least one token
# with a requested key; Python then applies the whole-token containment rule, so
# the loosening is auditable in one place rather than buried in SQL. `matched` is
# resolved caller-side because one course can answer several requested keys.
_BY_KEY_SQL = f"""
SELECT DISTINCT lower(btrim(ts.skill)) AS matched, {_COURSE_COLUMNS}
  FROM courses c
  JOIN LATERAL unnest(c.taught_skills) AS ts(skill) ON true
 WHERE {AGGREGABLE_COURSE_PREDICATE}
   AND (lower(btrim(ts.skill)) = ANY(%(keys)s)
        OR lower(btrim(ts.skill)) LIKE ANY(%(patterns)s))
 ORDER BY matched, c.course_id
"""


def courses_for_skills(
    esco_codes: list[str],
    skill_keys: list[str],
    *,
    config: Config | None = None,
) -> dict[str, dict[str, list[CourseCandidate]]]:
    """Retrieve recommendable courses for a set of missing skills.

    ``esco_codes`` are the concept URIs for skills that mapped to ESCO;
    ``skill_keys`` are the normalized keys of skills that did NOT map (the exact
    fallback). Returns ``{"by_esco": {esco_uri: [CourseCandidate...]},
    "by_key": {skill_key: [CourseCandidate...]}}`` — every requested code/key is
    a present key (empty list when nothing teaches it, so a caller can tell "no
    courses" from "never asked").

    A course teaching several requested skills appears under each of them; which
    of the caller's skills it covers is the caller's set-relative concern.
    """
    config = config or Config()
    by_esco: dict[str, list[CourseCandidate]] = {code: [] for code in esco_codes}
    by_key: dict[str, list[CourseCandidate]] = {key: [] for key in skill_keys}

    with _connection(config) as conn:
        if esco_codes:
            with conn.cursor() as cur:
                cur.execute(_BY_ESCO_SQL, {"codes": list(esco_codes)})
                for row in cur.fetchall():
                    by_esco.setdefault(row["matched"], []).append(_candidate(row))
        if skill_keys:
            with conn.cursor() as cur:
                cur.execute(_BY_KEY_SQL, {
                    "keys": list(skill_keys),
                    "patterns": [f"%{_longest_token(k)}%" for k in skill_keys],
                })
                rows = cur.fetchall()
            # 84% of course skills never map to an ESCO concept (measured on the
            # live corpus), so this fallback carries most of the retrieval — and
            # exact-string-only meant a gap in "problem-solving skills" found
            # nothing while a "problem solving" course sat unread.
            #
            # CONTAINMENT IS A FALLBACK WITHIN THE FALLBACK: it runs only for a
            # skill with no exact match at all. Measured on the real corpus, a gap
            # in "project management" matches exactly AND drags in "environmental
            # project management" and "project management office" — neither wrong
            # exactly, but Agent E's tiebreak scores candidates on rating, so a
            # loosely-related course can outrank the exact one. Widening only
            # where the shelf is otherwise empty means it can add coverage and
            # never dilute it.
            seen: dict[str, set[str]] = {k: set() for k in skill_keys}
            for row in rows:
                if row["matched"] in by_key:
                    key = row["matched"]
                    if row["course_id"] not in seen[key]:
                        seen[key].add(row["course_id"])
                        by_key[key].append(_candidate(row))

            for key in skill_keys:
                if by_key[key]:
                    continue        # an exact match exists; do not widen
                for row in rows:
                    if covers_skill(row["matched"], key) and row["course_id"] not in seen[key]:
                        seen[key].add(row["course_id"])
                        by_key[key].append(_candidate(row))

    # Deterministic order regardless of which taught-skill phrasing matched.
    for bucket in by_key.values():
        bucket.sort(key=lambda c: c.course_id)

    # A safety valve, not a filter. Set high enough that it does not bind on a
    # realistic corpus (no single skill is taught by 200 distinct courses here),
    # so that a catalog-scale ingest cannot turn one recommendation into tens of
    # thousands of materialized rows. NOTE: if it ever does bind, Agent E's
    # `courses_available` saturates at this number rather than being exact —
    # which is why it is set well above where a real answer lives.
    cap = config.agent_e_max_candidates_per_skill
    for bucket in (by_esco, by_key):
        for key, courses in bucket.items():
            if len(courses) > cap:
                bucket[key] = courses[:cap]
    return {"by_esco": by_esco, "by_key": by_key}


def _longest_token(key: str) -> str:
    """The token Postgres pre-filters on. Longest = most selective; the real
    decision is `covers_skill`, this only avoids scanning every course."""
    parts = tokens(key)
    return max(parts, key=len) if parts else key


def _candidate(row: dict[str, Any]) -> CourseCandidate:
    return CourseCandidate(
        course_id=row["course_id"],
        title=row["name"],
        provider=row.get("provider"),
        url=row["source_url"],
        taught_skills=list(row.get("taught_skills") or []),
        rating=float(row["rating"]) if row.get("rating") is not None else None,
        review_count=row.get("review_count"),
        enrollment_count=row.get("enrollment_count"),
        level=row.get("level"),
        last_updated=row["last_updated"].isoformat() if row.get("last_updated") else None,
        price=_price(row),
    )


def _price(row: dict[str, Any]) -> Optional[CoursePrice]:
    # None only when the provider reported no price at all (all three columns
    # null). A free course is amount 0.0 / is_free True — a real, kept value.
    amount = row.get("price_amount")
    is_free = row.get("price_is_free")
    currency = row.get("price_currency")
    if amount is None and is_free is None and currency is None:
        return None
    return CoursePrice(
        amount=float(amount) if amount is not None else None,
        currency=currency,
        is_free=bool(is_free),
    )
