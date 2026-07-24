-- Retrieval and maintenance indexes for courses. Mirror of Agent B's 0005.

-- HNSW on the essence embedding, PARTIAL on the two terminal states a
-- near-dup / retrieval query never wants to see. Callers must repeat both
-- predicates verbatim (duplicate_of IS NULL AND status <> 'rejected') or the
-- planner will not use the index.
CREATE INDEX IF NOT EXISTS courses_embedding_hnsw
    ON courses USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    WHERE duplicate_of IS NULL AND status <> 'rejected';

-- Required by ON DELETE SET NULL: without it the prune seq-scans courses once
-- per deleted canonical to null its duplicates' pointers.
CREATE INDEX IF NOT EXISTS courses_duplicate_of_idx
    ON courses (duplicate_of)
    WHERE duplicate_of IS NOT NULL;

-- The aggregation eligibility scan: active, canonical courses.
CREATE INDEX IF NOT EXISTS courses_active_idx
    ON courses (source)
    WHERE status = 'active' AND duplicate_of IS NULL;
