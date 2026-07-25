-- Course quality/price signals, for Agent E to tie-break between courses that
-- cover the same missing skills equally well.
--
-- All additive and nullable: a provider that does not report a signal leaves it
-- NULL (a missing rating is missing, not a 0). Stored in the provider's own
-- native scale — NEVER normalized across providers at ingestion, since a 4.5/5
-- on one platform and a 4.5/5 on another may not be comparable populations;
-- cross-provider comparison is a consumer's query-time problem.
--
-- These are volatile: price and ratings drift while a course's title/description
-- (the content_hash) does not, so they are refreshed EVERY cycle by the store's
-- lightweight path, independent of the content-gated extract/embed path.

ALTER TABLE courses
    ADD COLUMN rating            numeric,
    ADD COLUMN review_count      integer,
    ADD COLUMN enrollment_count  bigint,
    ADD COLUMN last_updated      timestamptz,   -- provider's own "last updated", if any
    ADD COLUMN price_amount      numeric,
    ADD COLUMN price_currency    text,          -- NULL for a free course (no currency applies)
    ADD COLUMN price_is_free     boolean,
    -- When the volatile snapshot above was taken. Because price fluctuates
    -- constantly (Udemy's is notoriously volatile), the honest record is
    -- "this was the price when we last looked", not a stable fact.
    ADD COLUMN price_observed_at timestamptz;

-- Non-negativity only. Deliberately NO rating upper bound: providers use
-- different scales (0-5, 0-10, percentages), and a CHECK (rating <= 5) would
-- reject a future 0-10 provider — exactly the cross-provider assumption this
-- design refuses to bake in.
ALTER TABLE courses
    ADD CONSTRAINT courses_rating_nonneg_check CHECK (rating IS NULL OR rating >= 0),
    ADD CONSTRAINT courses_review_count_nonneg_check CHECK (review_count IS NULL OR review_count >= 0),
    ADD CONSTRAINT courses_enrollment_nonneg_check CHECK (enrollment_count IS NULL OR enrollment_count >= 0),
    ADD CONSTRAINT courses_price_amount_nonneg_check CHECK (price_amount IS NULL OR price_amount >= 0);
