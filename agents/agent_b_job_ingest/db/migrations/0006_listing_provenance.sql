-- Who posted a listing, and what they were actually asking for.
--
-- Added in phase 3, when Dubizzle turned out to be a classifieds site rather
-- than a vacancy board. Two facts that were safe to assume for a curated jobs
-- blog are not safe here, and each breaks a different thing.
--
-- listing_intent — VACANCY vs SEEKING.
--   A large share of Dubizzle's job listings are people advertising themselves:
--   "I am Looking for Job AutoCAD Draughtsman", "Freelancer AutoCAD Draughtsman
--   Available" — and they are NOT confined to the Jobs Wanted category; all four
--   of the first four listings in the IT-Telecom *vacancy* category were
--   seekers. Counting those as postings measures SUPPLY and reports it as
--   DEMAND. Agent C would tell a user that AutoCAD skills are in demand when
--   what was counted is draughtsmen looking for work. The rows look entirely
--   normal, so nothing downstream could detect it.
--
-- poster_type — COMPANY vs INDIVIDUAL.
--   Listings are written by private individuals and contain personal details
--   ("Filipina lady looking for opportunities", "I am a trained female
--   caregiver"), including nationality and gender. Embedding those into a
--   vector store that Agent C retrieves from is a different proposition from
--   ingesting corporate vacancy ads.
--
-- Both default to 'unknown' and BOTH must be resolved before a posting counts
-- toward skill_demand_stats. Defaulting either to its useful value would mean a
-- classifier failure silently publishes exactly what these columns exist to
-- keep out — the failure would look like ordinary demand data.
--
-- NOTE: Dubizzle exposes no poster identity whatsoever — no seller name, no
-- company name, no business-account badge. So poster_type cannot be determined
-- from the page today, and every Dubizzle row stays 'unknown' and therefore
-- unaggregated. That is the intended outcome, not a gap to be worked around by
-- inferring a poster type the source never stated.

ALTER TABLE job_postings
    ADD COLUMN listing_intent text NOT NULL DEFAULT 'unknown'
        CHECK (listing_intent IN ('vacancy', 'seeking', 'service', 'unknown')),
    ADD COLUMN poster_type text NOT NULL DEFAULT 'unknown'
        CHECK (poster_type IN ('company', 'individual', 'unknown'));

COMMENT ON COLUMN job_postings.listing_intent IS
    'vacancy = an employer hiring; seeking = a person advertising themselves; '
    'service = a service offering, neither of the above. Only vacancy aggregates.';

COMMENT ON COLUMN job_postings.poster_type IS
    'Only company aggregates. Never inferred from a source that does not state it.';

-- The aggregation predicate, indexed so the eligibility filter does not degrade
-- into a sequential scan as the table grows. Partial on the two values that
-- actually qualify, because everything else is excluded by definition.
CREATE INDEX IF NOT EXISTS job_postings_aggregable_idx
    ON job_postings (sector, posted_date DESC)
    WHERE status = 'active'
      AND duplicate_of IS NULL
      AND listing_intent = 'vacancy'
      AND poster_type = 'company';
