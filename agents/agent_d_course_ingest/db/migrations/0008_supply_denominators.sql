-- Make the supply numbers interpretable, and give the ESCO join one honest grain.
--
-- Two problems this fixes, both measured on the live corpus 2026-07-28:
--
-- 1. `course_count = 3` cannot be read without knowing 3 out of how many. The
--    denominator was computed and discarded, exactly as Agent B's `sector_volume`
--    was before its own audit.
--
-- 2. skill_supply_stats is keyed by skill_key, and several skill_keys map to one
--    ESCO concept (measured: one concept carried SEVEN). The documented payoff
--    query joins demand to supply `USING (esco_code)`, so those variants fan out:
--    131 matched concepts produced 273 rows and inflated summed demand from 237
--    to 407 (+72%). Re-aggregating skill_key rows cannot fix it — summing double
--    counts a course that teaches two variants, taking max undercounts. The only
--    correct concept-level count is `count(DISTINCT course_id)` computed from the
--    courses themselves, which is what concept_supply_stats holds.

ALTER TABLE skill_supply_stats
    -- Eligible courses in the snapshot: the denominator `course_count` needs.
    ADD COLUMN IF NOT EXISTS total_courses integer NOT NULL DEFAULT 0,
    -- provider_count uses count(DISTINCT provider), which silently ignores NULLs
    -- — so a skill taught by 5 provider-less courses reported 5 courses from 0
    -- providers. Publishing the gap keeps the count honest rather than guessing
    -- that unknown providers are all the same one, or all different.
    ADD COLUMN IF NOT EXISTS courses_without_provider integer NOT NULL DEFAULT 0;


-- The concept grain: one row per ESCO concept per snapshot, counted from
-- courses directly so a course teaching several phrasings of one concept counts
-- ONCE. This is what joins to skill_demand_stats.esco_code.
CREATE TABLE IF NOT EXISTS concept_supply_stats (
    esco_code             text        NOT NULL,
    -- The concept's preferred label where the taxonomy is loaded; otherwise the
    -- modal raw phrasing. Display only — esco_code is the identity.
    label                 text,

    window_start          date        NOT NULL,
    window_end            date        NOT NULL,

    -- count(DISTINCT course_id) across every skill phrasing mapping to this
    -- concept. Correct by construction, unlike any rollup of skill_key rows.
    course_count          integer     NOT NULL,
    provider_count        integer     NOT NULL DEFAULT 0,
    total_courses         integer     NOT NULL DEFAULT 0,
    -- Which raw phrasings folded into this concept — the audit trail for a
    -- consumer who wants to check the merge, mirroring Agent C's
    -- `also_phrased_as` on the gap side.
    variant_keys          jsonb       NOT NULL DEFAULT '[]'::jsonb,
    levels                jsonb       NOT NULL DEFAULT '[]'::jsonb,
    sample_courses        jsonb       NOT NULL DEFAULT '[]'::jsonb,

    low_confidence        boolean     NOT NULL DEFAULT false,
    computed_at           timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT concept_supply_window_order_check CHECK (window_end >= window_start),

    -- CONSUMER CONTRACT: filter to the latest window,
    --   WHERE window_end = (SELECT max(window_end) FROM concept_supply_stats)
    PRIMARY KEY (esco_code, window_end)
);

CREATE INDEX IF NOT EXISTS concept_supply_stats_window_idx
    ON concept_supply_stats (window_end DESC, course_count DESC);
