-- Aggregated skill SUPPLY — the mirror of Agent B's skill_demand_stats.
--
-- Per skill: how many courses teach it, from how many providers, at what
-- levels. Joined with skill_demand_stats ON esco_code, this is what answers the
-- market question "which in-demand skills have few courses?".
--
-- Keyed by skill_key (not esco_code) for the same reason skill_demand_stats is
-- keyed by skill_key: esco_code is nullable (an unmapped skill still has real
-- supply), and a nullable column cannot sit in a primary key. esco_code is a
-- fillable column; consumers GROUP BY it to get canonical counts.

CREATE TABLE IF NOT EXISTS skill_supply_stats (
    skill                 text        NOT NULL,   -- display label, modal casing
    skill_key             text        NOT NULL,   -- lower(btrim(skill))
    -- Filled from course_esco_map. NULL = the mapper found no concept; never a
    -- guess, so treat an unmapped row as its raw skill_key.
    esco_code             text,

    window_start          date        NOT NULL,
    window_end            date        NOT NULL,

    course_count          integer     NOT NULL,
    provider_count        integer     NOT NULL DEFAULT 0,
    -- [{level, count}] and [{name, source_url}] — evidence a consumer can check
    -- the number against, the same instinct as skill_demand_stats.sample_postings.
    levels                jsonb       NOT NULL DEFAULT '[]'::jsonb,
    sample_courses        jsonb       NOT NULL DEFAULT '[]'::jsonb,

    -- Fewer than config.course_low_confidence_min_courses teach it: the number
    -- is real but thin. Visible to a consumer rather than buried.
    low_confidence        boolean     NOT NULL DEFAULT false,

    computed_at           timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT skill_supply_window_order_check
        CHECK (window_end >= window_start),

    -- window_end in the PK keeps history rather than overwriting each cycle.
    -- CONSUMER CONTRACT: filter to the latest window,
    --   WHERE window_end = (SELECT max(window_end) FROM skill_supply_stats)
    PRIMARY KEY (skill_key, window_end)
);
