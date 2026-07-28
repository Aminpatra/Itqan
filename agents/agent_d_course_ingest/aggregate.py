"""Recompute the supply tables — how many courses teach each skill.

The supply-side mirror of Agent B's aggregation, with one deliberate departure
that the first version got wrong by copying too faithfully:

**Supply is a STOCK, not a flow.** A job posting is a flow — a vacancy open
during a window, and one from 90 days ago is not current demand. A course does
not expire: a course ingested last quarter still teaches its skills today. The
original SQL filtered the population by ``first_seen_at`` inside a 90-day window,
so every course would silently leave the table 90 days after we first saw it, and
``course_count`` would decay toward zero as the corpus aged while nothing about
the actual supply changed. The window survives as the SNAPSHOT DATE — the history
axis the table is keyed on, which is what the docstring always claimed it was —
but it no longer decides who is counted.

Two grains are written, because they answer different questions and only one of
them can be computed correctly from the other:

* ``skill_supply_stats`` — per raw skill_key. The audit grain.
* ``concept_supply_stats`` — per ESCO concept, counted from courses directly.
  Several skill_keys map to one concept, so summing the skill grain double-counts
  a course teaching two phrasings and taking the max undercounts. Only
  ``count(DISTINCT course_id)`` over the join is right, and that is what joins to
  ``skill_demand_stats.esco_code``.

Eligibility stays simpler than the job side (no country/sector/intent): a course
counts if it is ``active`` and not a duplicate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

# Single source of truth for "a recommendable course", shared with the read
# surface Agent E queries — imported so retrieval and aggregation never drift.
from shared.course_market import AGGREGABLE_COURSE_PREDICATE

# The population, once. Note there is NO date predicate: every eligible course is
# current supply regardless of when we first saw it.
_ELIGIBLE_CTE = f"""
eligible AS (
    SELECT c.course_id, c.name, c.source_url, c.provider, c.level,
           ts.skill AS raw_skill, lower(btrim(ts.skill)) AS skill_key
      FROM courses c,
           LATERAL unnest(c.taught_skills) AS ts(skill)
     WHERE {AGGREGABLE_COURSE_PREDICATE}
       AND btrim(ts.skill) <> ''
),
totals AS (
    SELECT count(DISTINCT course_id) AS total_courses FROM eligible
)
"""

_AGGREGATE_SQL = f"""
INSERT INTO skill_supply_stats (
    skill, skill_key, esco_code, window_start, window_end,
    course_count, provider_count, total_courses, courses_without_provider,
    levels, sample_courses, low_confidence, computed_at
)
WITH {_ELIGIBLE_CTE},
per_skill AS (
    SELECT skill_key,
           mode() WITHIN GROUP (ORDER BY raw_skill) AS skill,
           count(DISTINCT course_id) AS course_count,
           count(DISTINCT provider) AS provider_count,
           count(DISTINCT course_id) FILTER (WHERE provider IS NULL)
               AS courses_without_provider
      FROM eligible
     GROUP BY skill_key
)
SELECT
    ps.skill,
    ps.skill_key,
    m.esco_uri,
    %(w_start)s::date,
    %(w_end)s::date,
    ps.course_count,
    ps.provider_count,
    t.total_courses,
    ps.courses_without_provider,
    COALESCE((
        SELECT jsonb_agg(jsonb_build_object('level', lv.level, 'count', lv.n)
                         ORDER BY lv.n DESC, lv.level)
          FROM (
            SELECT COALESCE(e.level, 'unspecified') AS level, count(DISTINCT e.course_id) AS n
              FROM eligible e WHERE e.skill_key = ps.skill_key
             GROUP BY COALESCE(e.level, 'unspecified')
          ) lv
    ), '[]'::jsonb),
    COALESCE((
        SELECT jsonb_agg(jsonb_build_object('name', sc.name, 'source_url', sc.source_url))
          FROM (
            SELECT DISTINCT e.course_id, e.name, e.source_url
              FROM eligible e WHERE e.skill_key = ps.skill_key
             ORDER BY e.course_id LIMIT 5
          ) sc
    ), '[]'::jsonb),
    (ps.course_count < %(low_conf_min)s),
    now()
FROM per_skill ps
LEFT JOIN course_esco_map m ON m.skill_key = ps.skill_key AND m.esco_uri IS NOT NULL
CROSS JOIN totals t
ON CONFLICT (skill_key, window_end) DO UPDATE SET
    skill = EXCLUDED.skill,
    esco_code = EXCLUDED.esco_code,
    course_count = EXCLUDED.course_count,
    provider_count = EXCLUDED.provider_count,
    total_courses = EXCLUDED.total_courses,
    courses_without_provider = EXCLUDED.courses_without_provider,
    levels = EXCLUDED.levels,
    sample_courses = EXCLUDED.sample_courses,
    low_confidence = EXCLUDED.low_confidence,
    computed_at = now()
"""

# The concept grain. `count(DISTINCT course_id)` across every phrasing that maps
# to the concept — a course teaching "python" and "python programming" counts
# once, which no rollup of the skill grain can achieve.
_CONCEPT_SQL = f"""
INSERT INTO concept_supply_stats (
    esco_code, label, window_start, window_end,
    course_count, provider_count, total_courses,
    variant_keys, levels, sample_courses, low_confidence, computed_at
)
WITH {_ELIGIBLE_CTE},
mapped AS (
    SELECT m.esco_uri AS esco_code, e.*
      FROM eligible e
      JOIN course_esco_map m
        ON m.skill_key = e.skill_key AND m.esco_uri IS NOT NULL
),
per_concept AS (
    SELECT esco_code,
           count(DISTINCT course_id) AS course_count,
           count(DISTINCT provider) AS provider_count,
           mode() WITHIN GROUP (ORDER BY raw_skill) AS modal_phrase
      FROM mapped
     GROUP BY esco_code
)
SELECT
    pc.esco_code,
    COALESCE(s.preferred_label, pc.modal_phrase),
    %(w_start)s::date,
    %(w_end)s::date,
    pc.course_count,
    pc.provider_count,
    t.total_courses,
    COALESCE((
        SELECT jsonb_agg(DISTINCT mm.skill_key)
          FROM mapped mm WHERE mm.esco_code = pc.esco_code
    ), '[]'::jsonb),
    COALESCE((
        SELECT jsonb_agg(jsonb_build_object('level', lv.level, 'count', lv.n)
                         ORDER BY lv.n DESC, lv.level)
          FROM (
            SELECT COALESCE(mm.level, 'unspecified') AS level,
                   count(DISTINCT mm.course_id) AS n
              FROM mapped mm WHERE mm.esco_code = pc.esco_code
             GROUP BY COALESCE(mm.level, 'unspecified')
          ) lv
    ), '[]'::jsonb),
    COALESCE((
        SELECT jsonb_agg(jsonb_build_object('name', sc.name, 'source_url', sc.source_url))
          FROM (
            SELECT DISTINCT mm.course_id, mm.name, mm.source_url
              FROM mapped mm WHERE mm.esco_code = pc.esco_code
             ORDER BY mm.course_id LIMIT 5
          ) sc
    ), '[]'::jsonb),
    (pc.course_count < %(low_conf_min)s),
    now()
FROM per_concept pc
LEFT JOIN esco_skills s ON s.esco_uri = pc.esco_code
CROSS JOIN totals t
ON CONFLICT (esco_code, window_end) DO UPDATE SET
    label = EXCLUDED.label,
    course_count = EXCLUDED.course_count,
    provider_count = EXCLUDED.provider_count,
    total_courses = EXCLUDED.total_courses,
    variant_keys = EXCLUDED.variant_keys,
    levels = EXCLUDED.levels,
    sample_courses = EXCLUDED.sample_courses,
    low_confidence = EXCLUDED.low_confidence,
    computed_at = now()
"""

# Clear the snapshot before rebuilding it, so a skill whose last course was
# removed cannot keep a row from an earlier run of the same day and be served
# forever as current supply — the phantom-row bug Agent B's audit measured at 114
# rows before its own fix.
#
# Delete-then-insert rather than a computed_at watermark: inside a transaction
# Postgres' now() is the TRANSACTION START time, so "delete rows older than the
# moment this run began" deletes the rows the run has just written. That is a
# genuinely easy mistake — it was made here first — and the ordering below has no
# such subtlety. Both statements share the caller's transaction, so a failed
# rebuild rolls the delete back with it.
_CLEAR_SKILL_SQL = "DELETE FROM skill_supply_stats WHERE window_end = %(w_end)s::date"
_CLEAR_CONCEPT_SQL = "DELETE FROM concept_supply_stats WHERE window_end = %(w_end)s::date"

_SKILL_COUNT_SQL = f"""
SELECT count(DISTINCT lower(btrim(s))) FROM courses, unnest(taught_skills) AS s
 WHERE {AGGREGABLE_COURSE_PREDICATE} AND btrim(s) <> ''
"""


@dataclass
class SupplySummary:
    window_start: date
    window_end: date
    rows_written: int
    skills_with_supply: int
    concepts_written: int = 0
    rows_cleared: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "rows_written": self.rows_written,
            "skills_with_supply": self.skills_with_supply,
            "concepts_written": self.concepts_written,
            "rows_cleared": self.rows_cleared,
        }


def recompute_supply(store: Any, config: Any, *, as_of: date | None = None) -> SupplySummary:
    w_end = as_of or date.today()
    # The snapshot's nominal span. Retained for the table's history axis and for
    # continuity with the demand side's shape; it does NOT filter the population.
    w_start = w_end - timedelta(days=config.course_window_days)
    params = {
        "w_start": w_start,
        "w_end": w_end,
        "low_conf_min": config.course_low_confidence_min_courses,
    }
    clear = {"w_end": w_end}

    cleared = (store.replace_stats_window(sql=_CLEAR_SKILL_SQL, params=clear)
               + store.replace_stats_window(sql=_CLEAR_CONCEPT_SQL, params=clear))
    rows = store.replace_stats_window(sql=_AGGREGATE_SQL, params=params)
    concepts = store.replace_stats_window(sql=_CONCEPT_SQL, params=params)
    skills = store.scalar(_SKILL_COUNT_SQL, {}) or 0
    return SupplySummary(
        window_start=w_start, window_end=w_end,
        rows_written=rows, skills_with_supply=int(skills),
        concepts_written=concepts, rows_cleared=cleared,
    )
