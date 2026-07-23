-- The ESCO mapping layer: canonical skill concepts, their searchable labels,
-- and the map from our free-text skill keys onto them.
--
-- Why this exists: free-text skills fragment. The first live demand table held
-- "prioritization", "priority organization" and "task organization" as three
-- rows — one concept, three counts, each wrong by division. The
-- skill_demand_stats.esco_code column was reserved for exactly this layer, and
-- Agent C needs one vocabulary to match candidate skills against demand.
--
-- NOT part of the Agent C output contract (same status as source_health): Agent
-- C reads job_postings and skill_demand_stats; these tables are how esco_code
-- gets its value and how a candidate's free-text skill finds its concept.

-- One row per ESCO concept (~13.9K in v1.2). The URI is ESCO's own identifier
-- and the natural PK; there is no shorter "code" for the skills pillar.
CREATE TABLE IF NOT EXISTS esco_skills (
    esco_uri        text PRIMARY KEY,
    preferred_label text        NOT NULL,
    -- 'skill/competence' vs 'knowledge' in ESCO's own vocabulary. Kept verbatim
    -- rather than normalised: it is their taxonomy, not ours to reshape.
    skill_type      text,
    description     text,
    esco_version    text        NOT NULL,
    loaded_at       timestamptz NOT NULL DEFAULT now()
);

-- One row PER LABEL, preferred and alternative alike (~40K rows), each with its
-- own embedding. This is the whole recall strategy: "prioritisation" is an alt
-- label of a concept whose preferred label shares no characters with it, so a
-- single per-concept vector would miss what a per-label vector catches. Lexical
-- exact-match runs against label_key; the vector search runs against embedding;
-- both resolve to the same esco_uri.
CREATE TABLE IF NOT EXISTS esco_labels (
    -- lower(btrim(label)) — the same normalisation skill_demand_stats.skill_key
    -- uses, so an exact match is a plain equality join.
    label_key    text NOT NULL,
    label        text NOT NULL,
    esco_uri     text NOT NULL REFERENCES esco_skills (esco_uri) ON DELETE CASCADE,
    is_preferred boolean NOT NULL DEFAULT false,
    -- NULL until the sync's embedding pass reaches it; the loader inserts
    -- labels first and embeds the NULL ones in batches, so an interrupted sync
    -- resumes instead of restarting.
    embedding    vector(1536),

    PRIMARY KEY (label_key, esco_uri)
);

CREATE INDEX IF NOT EXISTS esco_labels_embedding_hnsw
    ON esco_labels USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- The map: our normalised free-text key -> an ESCO concept, or an honest miss.
--
-- method='unmapped' with a NULL uri is a real answer, recorded so a cycle never
-- re-pays an embedding call for a key already known to match nothing. Such keys
-- are retried only when esco_version changes — a new taxonomy release is the
-- one event that can turn a miss into a hit.
CREATE TABLE IF NOT EXISTS skill_esco_map (
    skill_key    text PRIMARY KEY,
    esco_uri     text REFERENCES esco_skills (esco_uri) ON DELETE SET NULL,
    method       text        NOT NULL,
    -- Cosine similarity for embedding mappings; NULL for lexical ones, where
    -- "how similar" is not a meaningful question.
    similarity   numeric(5, 4),
    esco_version text        NOT NULL,
    mapped_at    timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT skill_esco_map_method_check
        CHECK (method IN ('exact', 'alt_label', 'embedding', 'unmapped')),
    -- An unmapped row must not point anywhere; a mapped row must point somewhere
    -- (until its concept is deleted, when SET NULL leaves the method as the
    -- audit trail of how it HAD been mapped).
    CONSTRAINT skill_esco_map_unmapped_null_check
        CHECK (method <> 'unmapped' OR esco_uri IS NULL)
);
