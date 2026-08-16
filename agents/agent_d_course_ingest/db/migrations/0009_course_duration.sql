-- How long a course takes, as the provider stated it.
--
-- The data was always there: the Coursera adapter's COURSE_FIELDS has requested
-- `workload` since it was written, and nothing ever persisted it. Meanwhile
-- `api/mapping` published `hours: null` with a comment saying "not 0, which
-- would render as '0 hours'" — and the card rendered exactly that, because the
-- front-end type declared it non-nullable.
--
-- A RANGE, not a number, and this is the whole design decision. Real values:
--
--     '1 hour 30 minutes'                  -> 1.5 .. 1.5
--     '4 weeks of study, 2-4 hours a week' -> 8   .. 16
--     '2 heures'                           -> 2   .. 2      (French, in the live data)
--     ''                                   -> null
--
-- Collapsing 8..16 to "12 hours" would publish a figure no provider ever stated.
-- That is the bug class this project has now fixed five times (gap_score null
-- not 0.0, price null not 0, birth date never inferred, salary null unless
-- stated, arrangement null unless stated), and the reason hours were left out
-- entirely the first time rather than guessed at.
--
-- `duration_text` keeps the provider's own words, so a range can always be
-- checked against what it was derived from, and an unparseable-but-present
-- workload is still worth showing a person.
ALTER TABLE courses ADD COLUMN IF NOT EXISTS hours_min      numeric;
ALTER TABLE courses ADD COLUMN IF NOT EXISTS hours_max      numeric;
ALTER TABLE courses ADD COLUMN IF NOT EXISTS duration_text  text;

-- Non-negative, and ordered. A backwards range is a parser bug, and storing one
-- would put "16 to 8 hours" on a card.
ALTER TABLE courses ADD CONSTRAINT courses_hours_sane
    CHECK (
        (hours_min IS NULL OR hours_min >= 0)
        AND (hours_max IS NULL OR hours_max >= 0)
        AND (hours_min IS NULL OR hours_max IS NULL OR hours_max >= hours_min)
    );
