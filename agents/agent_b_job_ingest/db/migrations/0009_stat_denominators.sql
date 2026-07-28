-- Make a demand count interpretable, and stop the trend column asserting things
-- it cannot know.
--
-- TREND VOCABULARY
-- ----------------
-- 'rising' used to cover two very different states: "this grew against a measured
-- past" and "we have no past to compare against". Measured on the live corpus
-- before this migration: every one of the 1153 rows in the latest window had
-- prior_frequency_count = 0, and 23 of them were published as 'rising'. That is a
-- finding derived from zero observations — the exact fabrication class the
-- aggregation module forbids. Two honest values are added:
--
--   'no_prior_data' — there is no earlier window at all (first run, or history
--                     pruned). Nothing can be said about direction.
--   'new'           — an earlier window exists and this skill was absent from it.
--                     Genuinely appearing, which is a real and different claim.
--
-- DENOMINATORS
-- ------------
-- frequency_count is an absolute count that reads like a rate. "58 postings ask
-- for X" means nothing without "out of how many". sector_volume was already
-- computed by the aggregation and thrown away.
--
-- distinct_posts exists because one source POST can now yield many postings: a
-- roundup advertising 19 roles becomes 19 rows sharing one source_post_url.
-- Measured: 36% of the in-window corpus is roundup children, and one sector shows
-- 51 postings drawn from only 26 distinct posts. Without this a consumer cannot
-- tell broad demand from a single employer's hiring drive.

ALTER TABLE skill_demand_stats
    DROP CONSTRAINT IF EXISTS skill_demand_stats_trend_check;

ALTER TABLE skill_demand_stats
    ADD CONSTRAINT skill_demand_stats_trend_check
        CHECK (trend IN ('rising', 'stable', 'falling', 'new', 'no_prior_data'));

ALTER TABLE skill_demand_stats
    ADD COLUMN IF NOT EXISTS sector_volume  integer,
    ADD COLUMN IF NOT EXISTS distinct_posts integer;

COMMENT ON COLUMN skill_demand_stats.sector_volume IS
    'Eligible postings in this sector and window — the denominator for '
    'frequency_count. NULL on rows written before this column existed.';

COMMENT ON COLUMN skill_demand_stats.distinct_posts IS
    'Distinct source posts behind frequency_count. Lower than frequency_count '
    'when one roundup post advertised several of the vacancies.';
