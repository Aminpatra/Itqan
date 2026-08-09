-- Where the job actually is, and what the employer's own page says about it.
--
-- Aggregators post an article that LINKS to the real vacancy on the employer's
-- ATS. `root_fetch` has always followed that link and replaced the posting's
-- skills with the destination's — and then thrown the URL away. So the system
-- knew the employer page well enough to extract from it, and still sent the user
-- to the aggregator.
--
-- IDENTITY IS DELIBERATELY UNCHANGED. `posting_id = sha(source, source_url)` and
-- the (source, source_url) unique key stay exactly as they are. Making the
-- employer URL the identity would re-mint every row, age the old ones out, and
-- double-count demand across the overlap — finding E3 of the Agent B audit.
-- `source_url` remains "where we found it"; `final_url` is "where you apply".
ALTER TABLE job_postings ADD COLUMN IF NOT EXISTS final_url text;

-- Two aggregators carrying one vacancy share a destination. Exact match on a
-- canonical URL is a stronger duplicate signal than the embedding near-dup this
-- supplements — it is cheap, it is certain, and it catches cross-source repeats
-- that similarity misses because the two summaries were written differently.
CREATE INDEX IF NOT EXISTS job_postings_final_url_idx
    ON job_postings (final_url)
    WHERE final_url IS NOT NULL AND duplicate_of IS NULL;

-- --------------------------------------------------------------------------
-- What the destination page says that an aggregator summary never does.
--
-- EVERY ONE OF THESE IS NULLABLE AND STAYS NULL UNLESS THE PAGE SAYS SO. That is
-- the whole discipline: a salary of 0 reads as unpaid, a defaulted 'onsite' reads
-- as a fact about the employer, and this project has already had to fix that
-- class of bug three times (gap_score null not 0.0, course price null not 0,
-- birth date never inferred).

-- remote | hybrid | onsite.
--
-- This closes a loop. A user asked to filter by work arrangement and we could
-- only implement a retrieval-text bias, publishing
-- `arrangement_applied: "retrieval_bias"` in Agent C's calibration, because
-- NOTHING IN THE CORPUS RECORDED IT. Employer pages state it plainly; aggregator
-- summaries usually do not. With this the preference can become a real signal.
ALTER TABLE job_postings ADD COLUMN IF NOT EXISTS work_arrangement text;
ALTER TABLE job_postings DROP CONSTRAINT IF EXISTS job_postings_work_arrangement_check;
ALTER TABLE job_postings
    ADD CONSTRAINT job_postings_work_arrangement_check
    CHECK (work_arrangement IS NULL
           OR work_arrangement IN ('remote', 'hybrid', 'onsite'));

-- Pay, as stated. Finding E2 of the Agent B audit: the legitimacy prompt names
-- "unrealistic pay" as a scam tell with NOTHING TO CHECK IT AGAINST. A range,
-- because postings quote ranges; a period, because "3000" means very different
-- things monthly and annually.
ALTER TABLE job_postings ADD COLUMN IF NOT EXISTS salary_min numeric;
ALTER TABLE job_postings ADD COLUMN IF NOT EXISTS salary_max numeric;
ALTER TABLE job_postings ADD COLUMN IF NOT EXISTS salary_currency text;
ALTER TABLE job_postings ADD COLUMN IF NOT EXISTS salary_period text;

ALTER TABLE job_postings DROP CONSTRAINT IF EXISTS job_postings_salary_sane_check;
ALTER TABLE job_postings
    ADD CONSTRAINT job_postings_salary_sane_check
    CHECK (
        (salary_min IS NULL OR salary_min >= 0)
        AND (salary_max IS NULL OR salary_max >= 0)
        -- A range that runs backwards is an extraction error, not a salary.
        AND (salary_min IS NULL OR salary_max IS NULL OR salary_max >= salary_min)
        AND (salary_period IS NULL
             OR salary_period IN ('hour', 'day', 'week', 'month', 'year'))
    );

-- full_time | part_time | contract | internship | temporary.
-- An internship and a senior contract are not the same opportunity to a
-- graduate, and roundup summaries routinely omit which one it is.
ALTER TABLE job_postings ADD COLUMN IF NOT EXISTS employment_type text;
ALTER TABLE job_postings DROP CONSTRAINT IF EXISTS job_postings_employment_type_check;
ALTER TABLE job_postings
    ADD CONSTRAINT job_postings_employment_type_check
    CHECK (employment_type IS NULL
           OR employment_type IN ('full_time', 'part_time', 'contract',
                                  'internship', 'temporary'));
