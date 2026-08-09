-- Why a posting has no employer destination.
--
-- `final_url` says where a vacancy lives. Its ABSENCE has several very different
-- causes, and until now they were indistinguishable:
--
--   the article carries no external link at all — "send your CV to hr@..."
--   it links to a careers hub or a social page, not to a vacancy
--   robots.txt refuses the destination
--   the destination is unreachable
--   we simply have not looked yet
--
-- That distinction becomes load-bearing the moment postings are DELETED for
-- lacking a destination. "Why did 481 rows go?" has to be answerable from the
-- database months later, not from a terminal that has scrolled away. A column
-- is the difference between a prune that can be audited and one that has to be
-- taken on trust.
--
-- MEASURED 2026-08-09, tracing 30 el7far postings to their end: 16 had no link
-- at all, 11 pointed at a hub or LinkedIn, 2 were robots-refused, 1 was
-- unreachable, and 0 reached a vacancy page. Those five outcomes are this
-- vocabulary.
ALTER TABLE job_postings ADD COLUMN IF NOT EXISTS destination_status text;

ALTER TABLE job_postings DROP CONSTRAINT IF EXISTS job_postings_destination_status_check;
ALTER TABLE job_postings
    ADD CONSTRAINT job_postings_destination_status_check
    CHECK (destination_status IS NULL
           OR destination_status IN ('resolved',      -- final_url was recorded
                                     'no_link',       -- nothing to follow
                                     'hub',           -- a careers index, not a vacancy
                                     'social',        -- LinkedIn, WhatsApp, Facebook
                                     'robots',        -- the destination refuses us
                                     'unreachable',   -- fetch or parse failed
                                     'source_is_destination'));  -- the row's own URL is the ad

-- NULL means "never traced", which is why it is the default and not a value:
-- a row nobody has looked at must not be mistaken for one we looked at and
-- found wanting. The prune only ever removes rows whose status is a MEASURED
-- failure, so an untraced row survives by construction.
CREATE INDEX IF NOT EXISTS job_postings_destination_status_idx
    ON job_postings (destination_status)
    WHERE destination_status IS NOT NULL;
