"""Recompute ``skill_supply_stats`` — how many courses teach each skill.

The supply-side mirror of Agent B's aggregation. Per skill (by normalized
skill_key, with esco_code filled from course_esco_map): how many active,
non-duplicate courses teach it, from how many distinct providers, at what levels.
Joined with skill_demand_stats ON esco_code, this is the payoff — in-demand
skills with thin course supply.

Eligibility is deliberately simpler than the job side (no country/sector/intent):
a course counts if it is ``active`` and not a duplicate. There is no window on
posted-date because courses have no posting date; the "window" is first_seen_at,
kept only so the table carries history and Agent B's latest-window consumer
contract has an analog here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

# Single source of truth for "a recommendable course", shared with the read
# surface Agent E queries — imported so retrieval and aggregation never drift.
from shared.course_market import AGGREGABLE_COURSE_PREDICATE

_AGGREGATE_SQL = f"""
INSERT INTO skill_supply_stats (
    skill, skill_key, esco_code, window_start, window_end,
    course_count, provider_count, levels, sample_courses, low_confidence, computed_at
)
WITH eligible AS (
    SELECT course_id, name, source_url, provider, level, required_skills.skill AS raw_skill,
           lower(btrim(required_skills.skill)) AS skill_key
      FROM courses,
           LATERAL unnest(taught_skills) AS required_skills(skill)
     WHERE {AGGREGABLE_COURSE_PREDICATE}
       AND btrim(required_skills.skill) <> ''
       AND first_seen_at::date > %(w_start)s
       AND first_seen_at::date <= %(w_end)s
),
per_skill AS (
    SELECT skill_key,
           mode() WITHIN GROUP (ORDER BY raw_skill) AS skill,
           count(DISTINCT course_id) AS course_count,
           count(DISTINCT provider) AS provider_count
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
ON CONFLICT (skill_key, window_end) DO UPDATE SET
    skill = EXCLUDED.skill,
    esco_code = EXCLUDED.esco_code,
    course_count = EXCLUDED.course_count,
    provider_count = EXCLUDED.provider_count,
    levels = EXCLUDED.levels,
    sample_courses = EXCLUDED.sample_courses,
    low_confidence = EXCLUDED.low_confidence,
    computed_at = now()
"""

_SKILL_COUNT_SQL = f"""
SELECT count(DISTINCT lower(btrim(s))) FROM courses, unnest(taught_skills) AS s
 WHERE {AGGREGABLE_COURSE_PREDICATE} AND btrim(s) <> ''
   AND first_seen_at::date > %(w_start)s AND first_seen_at::date <= %(w_end)s
"""


@dataclass
class SupplySummary:
    window_start: date
    window_end: date
    rows_written: int
    skills_with_supply: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "rows_written": self.rows_written,
            "skills_with_supply": self.skills_with_supply,
        }


def recompute_supply(store: Any, config: Any, *, as_of: date | None = None) -> SupplySummary:
    w_end = as_of or date.today()
    w_start = w_end - timedelta(days=config.course_window_days)
    params = {
        "w_start": w_start,
        "w_end": w_end,
        "low_conf_min": config.course_low_confidence_min_courses,
    }
    rows = store.replace_stats_window(sql=_AGGREGATE_SQL, params=params)
    skills = store.scalar(_SKILL_COUNT_SQL, params) or 0
    return SupplySummary(
        window_start=w_start, window_end=w_end,
        rows_written=rows, skills_with_supply=int(skills),
    )
