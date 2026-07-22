"""Recompute ``skill_demand_stats`` — the aggregated table Agent C falls back to.

This is the most fabrication-prone code in Agent B: every number it writes is a
claim about the labour market that someone may plan a career around. So the query
is written to count only what is real and to make its own filters legible.

The eligibility gate is the whole ethical core. A posting contributes to a demand
statistic ONLY if it is:

  * ``status = 'active'``         — not stale, not rejected, not under review
  * ``duplicate_of IS NULL``      — a duplicate would count one job as several
  * ``sector IS NOT NULL``        — never guessed into a default bucket
  * ``country = ANY(in_scope)``   — scope is a separate axis from scam; out-of-
                                     scope postings are stored, just not counted
  * ``listing_intent = 'vacancy'``— a job SEEKER advertising themselves is supply,
    ``AND poster_type = 'company'`` not demand, and counting it would invert the
                                     signal (migration 0006's index predicate,
                                     repeated here verbatim)

Two windows are counted — the current one and the one immediately before it — so
the trend label can be derived from the data rather than asserted. The trend is
deliberately conservative: below ``trend_min_volume`` a skill is always 'stable',
because 1 → 2 postings is +100% noise, not a rising trend, and publishing it as
one would be exactly the kind of fabricated finding this table must not contain.

History is kept, not overwritten: ``window_end`` is in the primary key, so each
run appends and a bad run cannot destroy the prior state. Re-running on the same
day updates that day's rows (ON CONFLICT) rather than duplicating them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any


@dataclass
class AggregationSummary:
    window_start: date
    window_end: date
    rows_written: int
    sectors_with_current_demand: int
    # Sectors that had postings in the prior window and none now — demand did not
    # error, it went to zero. A normal state, reported so it is visible rather
    # than looking like a dropped sector.
    sectors_zeroed: int
    # Postings that were eligible in every way EXCEPT that country could not be
    # determined. Excluded from aggregation, but counted separately: an
    # extraction failure and an out-of-scope posting are different facts, and a
    # rising count here means extraction is losing the country, not that demand
    # fell.
    postings_excluded_null_country: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "rows_written": self.rows_written,
            "sectors_with_current_demand": self.sectors_with_current_demand,
            "sectors_zeroed": self.sectors_zeroed,
            "postings_excluded_null_country": self.postings_excluded_null_country,
        }


# The eligibility CTE, defined once and reused for both windows so the two counts
# cannot drift apart. ``eff_date`` falls back to first_seen_at when a posting has
# no stated posted_date (a relative date the adapter left unresolved), because
# the day we first observed it is the honest stand-in — never "now".
_ELIGIBLE_CTE = """
eligible AS (
    SELECT posting_id, sector, title, source_url, required_skills,
           COALESCE(posted_date, first_seen_at::date) AS eff_date
      FROM job_postings
     WHERE status = 'active'
       AND duplicate_of IS NULL
       AND sector IS NOT NULL
       AND country = ANY(%(countries)s)
       AND listing_intent = 'vacancy'
       AND poster_type = 'company'
)
"""

_AGGREGATE_SQL = f"""
INSERT INTO skill_demand_stats (
    sector, skill, skill_key, window_start, window_end,
    frequency_count, prior_frequency_count, trend,
    co_occurring_skills, sample_postings, low_confidence, computed_at
)
WITH {_ELIGIBLE_CTE},
cur_postings AS (
    SELECT * FROM eligible WHERE eff_date > %(w_start)s AND eff_date <= %(w_end)s
),
prior_postings AS (
    SELECT * FROM eligible WHERE eff_date > %(p_start)s AND eff_date <= %(w_start)s
),
sector_volume AS (
    SELECT sector, count(*) AS n FROM cur_postings GROUP BY sector
),
cur_skills AS (
    SELECT c.sector, c.posting_id, c.title, c.source_url,
           btrim(s) AS skill, lower(btrim(s)) AS skill_key
      FROM cur_postings c, unnest(c.required_skills) AS s
     WHERE btrim(s) <> ''
),
prior_counts AS (
    SELECT p.sector, lower(btrim(s)) AS skill_key,
           count(DISTINCT p.posting_id) AS freq
      FROM prior_postings p, unnest(p.required_skills) AS s
     WHERE btrim(s) <> ''
     GROUP BY p.sector, lower(btrim(s))
),
cur_counts AS (
    SELECT sector, skill_key,
           mode() WITHIN GROUP (ORDER BY skill) AS skill,
           count(DISTINCT posting_id) AS freq
      FROM cur_skills
     GROUP BY sector, skill_key
)
SELECT
    cc.sector,
    cc.skill,
    cc.skill_key,
    %(w_start)s::date,
    %(w_end)s::date,
    cc.freq,
    COALESCE(pc.freq, 0),
    CASE
        WHEN cc.freq < %(trend_min)s THEN 'stable'
        WHEN COALESCE(pc.freq, 0) = 0 THEN 'rising'
        WHEN cc.freq::numeric / pc.freq >= %(rising)s THEN 'rising'
        WHEN cc.freq::numeric / pc.freq <= %(falling)s THEN 'falling'
        ELSE 'stable'
    END,
    COALESCE((
        SELECT jsonb_agg(jsonb_build_object('skill', co.skill, 'count', co.n)
                         ORDER BY co.n DESC, co.skill)
          FROM (
            SELECT mode() WITHIN GROUP (ORDER BY cs2.skill) AS skill,
                   count(DISTINCT cs2.posting_id) AS n
              FROM cur_skills cs1
              JOIN cur_skills cs2
                ON cs2.posting_id = cs1.posting_id
               AND cs2.skill_key <> cs1.skill_key
             WHERE cs1.sector = cc.sector AND cs1.skill_key = cc.skill_key
             GROUP BY cs2.skill_key
             ORDER BY count(DISTINCT cs2.posting_id) DESC,
                      mode() WITHIN GROUP (ORDER BY cs2.skill)
             LIMIT 5
          ) co
    ), '[]'::jsonb),
    COALESCE((
        SELECT jsonb_agg(jsonb_build_object('title', sp.title, 'source_url', sp.source_url))
          FROM (
            SELECT DISTINCT cs3.posting_id, cs3.title, cs3.source_url
              FROM cur_skills cs3
             WHERE cs3.sector = cc.sector AND cs3.skill_key = cc.skill_key
             ORDER BY cs3.posting_id
             LIMIT 5
          ) sp
    ), '[]'::jsonb),
    (sv.n < %(low_conf_min)s),
    now()
FROM cur_counts cc
JOIN sector_volume sv ON sv.sector = cc.sector
LEFT JOIN prior_counts pc ON pc.sector = cc.sector AND pc.skill_key = cc.skill_key
ON CONFLICT (sector, skill_key, window_end) DO UPDATE SET
    skill = EXCLUDED.skill,
    window_start = EXCLUDED.window_start,
    frequency_count = EXCLUDED.frequency_count,
    prior_frequency_count = EXCLUDED.prior_frequency_count,
    trend = EXCLUDED.trend,
    co_occurring_skills = EXCLUDED.co_occurring_skills,
    sample_postings = EXCLUDED.sample_postings,
    low_confidence = EXCLUDED.low_confidence,
    computed_at = now()
"""

# Sectors that had demand in the prior window but none in the current one.
_ZEROED_SQL = f"""
WITH {_ELIGIBLE_CTE},
cur_sectors AS (
    SELECT DISTINCT sector FROM eligible
     WHERE eff_date > %(w_start)s AND eff_date <= %(w_end)s
),
prior_sectors AS (
    SELECT DISTINCT sector FROM eligible
     WHERE eff_date > %(p_start)s AND eff_date <= %(w_start)s
)
SELECT count(*) FROM prior_sectors
 WHERE sector NOT IN (SELECT sector FROM cur_sectors)
"""

_CURRENT_SECTORS_SQL = f"""
WITH {_ELIGIBLE_CTE}
SELECT count(DISTINCT sector) FROM eligible
 WHERE eff_date > %(w_start)s AND eff_date <= %(w_end)s
"""

# Eligible in every way EXCEPT country — same filters as the CTE minus the
# country clause and the sector requirement, restricted to null country in the
# current window.
_NULL_COUNTRY_SQL = """
SELECT count(*) FROM job_postings
 WHERE status = 'active'
   AND duplicate_of IS NULL
   AND listing_intent = 'vacancy'
   AND poster_type = 'company'
   AND country IS NULL
   AND COALESCE(posted_date, first_seen_at::date) > %(w_start)s
   AND COALESCE(posted_date, first_seen_at::date) <= %(w_end)s
"""


def recompute_stats(store: Any, config: Any, *, as_of: date | None = None) -> AggregationSummary:
    """Recompute the current window's stats and return what the run log needs.

    Empty is not an error. A sector — or the whole corpus — with no eligible
    current postings simply writes no rows; the node must not raise or skip on
    that, because "no current demand data for this sector" is a real answer Agent
    C is told to expect (it filters to the latest window).
    """
    w_end = as_of or date.today()
    w_start = w_end - timedelta(days=config.window_days)
    p_start = w_start - timedelta(days=config.window_days)

    params = {
        "countries": list(config.in_scope_countries),
        "w_start": w_start,
        "w_end": w_end,
        "p_start": p_start,
        "trend_min": config.trend_min_volume,
        "rising": config.trend_rising_ratio,
        "falling": config.trend_falling_ratio,
        "low_conf_min": config.low_confidence_min_postings,
    }

    rows_written = store.replace_stats_window(sql=_AGGREGATE_SQL, params=params)
    current = store.scalar(_CURRENT_SECTORS_SQL, params) or 0
    zeroed = store.scalar(_ZEROED_SQL, params) or 0
    null_country = store.scalar(_NULL_COUNTRY_SQL, params) or 0

    return AggregationSummary(
        window_start=w_start,
        window_end=w_end,
        rows_written=rows_written,
        sectors_with_current_demand=int(current),
        sectors_zeroed=int(zeroed),
        postings_excluded_null_country=int(null_country),
    )
