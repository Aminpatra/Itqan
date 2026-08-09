-- Who published a posting, on what terms, and when it dies.
--
-- Added for GulfTalent, which we crawl under a NARROW EXCEPTION in its own
-- terms of use: scraping is permitted "as an internet search engine making the
-- information searchable by users, and provided you display only minimal
-- snippets of each GulfTalent page to your users, in each case mention the
-- source clearly as GulfTalent, and link each snippet back to the
-- corresponding page on GulfTalent."
--
-- Two of those three conditions are about what we DISPLAY, which is enforced in
-- `api/mapping.py` and pinned by tests. This migration is about the third
-- thing: a row must be able to state its own provenance, because it outlives
-- the config entry that created it. Agent D already learned this — `courses`
-- carries `attribution` and `license` per row so freeCodeCamp's CC-BY-SA
-- travels with the data rather than living in a source registry somebody edits
-- later.

-- How the publisher is to be named to a user. NOT the internal `source` key:
-- that is 'gulftalent', and attribution should read like a citation.
ALTER TABLE job_postings ADD COLUMN IF NOT EXISTS attribution text;

-- The terms this row was collected under. A URL rather than a copy of the text,
-- because terms change and a stale copy is worse than a pointer — this is what
-- an auditor follows to check whether we are still inside them.
ALTER TABLE job_postings ADD COLUMN IF NOT EXISTS terms_url text;

-- --------------------------------------------------------------------------
-- When the publisher says the vacancy stops being real.
--
-- Everything else in this table infers death: `missed_cycles` counts how many
-- times we looked and did not find it, and `stale_since` records when we gave
-- up. Those are OUR observations. `validThrough` in a schema.org JobPosting is
-- the EMPLOYER'S OWN STATEMENT, which is strictly better evidence.
--
-- RECORDED HERE, NOT YET ACTED ON. Wiring it into the staleness pass changes
-- what gets pruned, and this project's rule is that a data-affecting change
-- gets its own before/after measurement rather than riding along with a source
-- addition. Until then it is a column consumers may read and the pipeline does
-- not enforce.
ALTER TABLE job_postings ADD COLUMN IF NOT EXISTS expires_at timestamptz;

-- Finding an expired-but-active row is the query that justifies wiring it up
-- later, so make it cheap. Partial: the column is NULL for every source that
-- does not publish an expiry, which today is all of them but one.
CREATE INDEX IF NOT EXISTS job_postings_expires_at_idx
    ON job_postings (expires_at)
    WHERE expires_at IS NOT NULL AND status = 'active';
