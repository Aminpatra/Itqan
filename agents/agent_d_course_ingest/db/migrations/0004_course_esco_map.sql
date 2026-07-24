-- Agent D's own ESCO mapping cache for course taught-skills.
--
-- Structurally identical to Agent B's skill_esco_map, but a SEPARATE table:
-- skill_esco_map records the POSTING corpus and is written only by Agent B.
-- Course skills are a different corpus; mixing them would let one agent's
-- mapping decisions silently change the other's. Populated by reusing the
-- READ-ONLY tier resolver shared.job_market.map_skills_to_esco (exact label ->
-- alt label -> nearest-label embedding -> unmapped) and persisting the result.
--
-- No FK to esco_skills on purpose: the ESCO taxonomy tables are created by Agent
-- B's migrations, and Agent D's migration set must not assume another agent has
-- run. esco_uri values are valid by construction (they come from querying
-- esco_labels at map time); the RUNTIME dependency on a synced taxonomy is
-- checked by the CLI, not enforced here as a migrate-time schema coupling.

CREATE TABLE IF NOT EXISTS course_esco_map (
    skill_key    text PRIMARY KEY,
    esco_uri     text,
    method       text        NOT NULL,
    similarity   numeric(5, 4),
    esco_version text        NOT NULL,
    mapped_at    timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT course_esco_map_method_check
        CHECK (method IN ('exact', 'alt_label', 'embedding', 'unmapped')),
    CONSTRAINT course_esco_map_unmapped_null_check
        CHECK (method <> 'unmapped' OR esco_uri IS NULL)
);
