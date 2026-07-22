-- Indexes.
--
-- Several are PARTIAL. Where a partial index exists, callers must repeat its
-- predicate verbatim or the planner will not use it — noted per index below.

-- A source must never yield two ids for one URL, nor two rows for one id.
CREATE UNIQUE INDEX IF NOT EXISTS job_postings_source_url_uk
    ON job_postings (source, source_url);

-- Retrieval path. `duplicate_of IS NULL` is in the predicate because serving a
-- duplicate to Agent C would double-count one job as two opportunities.
CREATE INDEX IF NOT EXISTS job_postings_sector_date_idx
    ON job_postings (sector, posted_date DESC)
    WHERE status = 'active' AND duplicate_of IS NULL;

-- Required by the ON DELETE SET NULL foreign key. Without it, every row the
-- 60-day prune deletes triggers a sequential scan of job_postings looking for
-- referencing rows.
CREATE INDEX IF NOT EXISTS job_postings_duplicate_of_idx
    ON job_postings (duplicate_of)
    WHERE duplicate_of IS NOT NULL;

-- Candidate scan for near-duplicate detection, and the review queue.
CREATE INDEX IF NOT EXISTS job_postings_group_recent_idx
    ON job_postings (source_group, first_seen_at DESC)
    WHERE status <> 'rejected' AND duplicate_of IS NULL;

-- Audit queue for rejected postings. They are retained, never hard-deleted, so
-- a human can review borderline legitimacy calls.
CREATE INDEX IF NOT EXISTS job_postings_rejected_audit_idx
    ON job_postings (first_seen_at DESC, legitimacy_score)
    WHERE status = 'rejected';

-- Staleness and pruning.
CREATE INDEX IF NOT EXISTS job_postings_source_seen_idx
    ON job_postings (source, last_seen_at DESC);
CREATE INDEX IF NOT EXISTS job_postings_prune_idx
    ON job_postings (stale_since)
    WHERE status = 'stale';

-- Skill filtering.
CREATE INDEX IF NOT EXISTS job_postings_skills_gin
    ON job_postings USING gin (required_skills);

-- Vector search.
--
-- HNSW, not IVFFlat. IVFFlat must be TRAINED on existing rows, and this table
-- starts empty — an index built at migration time would be worthless. Worse,
-- the workload is IVFFlat's failure case: postings arrive every 12h and are
-- deleted at 60 days, so the corpus continuously drifts away from the trained
-- centroids and recall degrades SILENTLY, fixable only by a periodic REINDEX
-- with no natural trigger in a cron-driven design. HNSW maintains its graph
-- incrementally. IVFFlat only wins past roughly a million rows.
--
-- PARTIAL on two TERMINAL states. `duplicate_of IS NULL` and
-- `status <> 'rejected'` are effectively one-way, so this causes no index
-- churn — unlike predicating on `status = 'active'`, which would flip on every
-- stale and every revive.
--
-- CALLERS MUST REPEAT BOTH PREDICATES VERBATIM, and must query with `<=>`
-- (cosine distance). `<->` or `<#>` will not use this index.
CREATE INDEX IF NOT EXISTS job_postings_embedding_hnsw
    ON job_postings USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    WHERE duplicate_of IS NULL AND status <> 'rejected';

-- Agent C's lookup: latest window for a sector, highest demand first.
CREATE INDEX IF NOT EXISTS skill_demand_stats_lookup_idx
    ON skill_demand_stats (sector, window_end DESC, frequency_count DESC);
