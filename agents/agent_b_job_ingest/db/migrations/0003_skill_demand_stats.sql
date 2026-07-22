-- Aggregated skill demand — the second table Agent C consumes, and its fallback
-- when no live posting matches a user's sector.
--
-- Recomputed by a plain SQL aggregate each cycle. There is deliberately no
-- vector column here: this is counting, not retrieval, and a second embedding
-- store would be a way to make simple arithmetic expensive and approximate.

CREATE TABLE IF NOT EXISTS skill_demand_stats (
    sector                text        NOT NULL,

    -- The display label, in whatever casing the posting used.
    skill                 text        NOT NULL,
    -- The grouping key: lower(btrim(skill)). Free-text skills fragment on case
    -- and stray whitespace, and "Python" + "python" as two rows means BOTH
    -- counts are wrong. Deliberately no stemming or synonym merging — that is
    -- the future ESCO layer's job, and doing it heuristically here would
    -- fabricate structure that isn't in the data.
    skill_key             text        NOT NULL,
    -- Reserved. Always NULL until a mapping layer exists. A single scalar per
    -- row, so unlike a parallel array on job_postings there is no alignment
    -- hazard in reserving it now.
    esco_code             text,

    window_start          date        NOT NULL,
    window_end            date        NOT NULL,

    frequency_count       integer     NOT NULL,
    -- Published so a consumer can audit the trend label instead of trusting it
    -- — the same instinct as Agent A publishing rejected skills with rationales
    -- rather than silently dropping them.
    prior_frequency_count integer     NOT NULL DEFAULT 0,
    trend                 text        NOT NULL,

    -- [{skill, count}], top 5
    co_occurring_skills   jsonb       NOT NULL DEFAULT '[]'::jsonb,
    -- [{title, source_url}], max 5. Evidence, so a consumer can check the
    -- number against real postings rather than taking it on faith.
    sample_postings       jsonb       NOT NULL DEFAULT '[]'::jsonb,

    -- Sector-level, set on every row of a sector whose window held fewer than
    -- config.low_confidence_min_postings deduped postings. Distinct from the
    -- per-skill trend volume floor: that one governs only the trend label,
    -- while this says no number in the row is trustworthy. Visible to Agent C
    -- rather than buried in a small count nobody inspects.
    low_confidence        boolean     NOT NULL DEFAULT false,

    computed_at           timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT skill_demand_stats_trend_check
        CHECK (trend IN ('rising', 'stable', 'falling')),
    CONSTRAINT skill_demand_stats_window_order_check
        CHECK (window_end >= window_start),

    -- window_end in the PK keeps history rather than overwriting each cycle.
    -- Storage is trivial at this scale, the trend column becomes auditable
    -- against its own past rows, and a bad aggregation run cannot destroy the
    -- prior state.
    --
    -- CONSUMER CONTRACT: always filter to the latest window, e.g.
    --   WHERE window_end = (SELECT max(window_end) FROM skill_demand_stats)
    -- A sector absent from the latest window has no current demand data. That
    -- is a normal state, not an error, and it is why no sentinel row is written.
    PRIMARY KEY (sector, skill_key, window_end)
);
