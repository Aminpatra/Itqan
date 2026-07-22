-- Per-source cycle history, for detecting a sustained block.
--
-- NOT PART OF THE AGENT C CONTRACT. Agent C reads job_postings and
-- skill_demand_stats only. This is Agent B's internal operational bookkeeping —
-- the "two tables" rule governs the published interface, not every row the
-- agent stores, and schema_migrations is already a third table on that basis.
--
-- One row per configured source, forever. Bounded by construction, which is
-- precisely why this is a table and not an append-only log.
--
-- Detecting `degraded` must never change retry behaviour. That is enforced
-- structurally rather than by comment: this table is read and written ONLY in
-- the runlog node, nothing under sources/ imports it, so no code path exists
-- that could escalate retries in response. A test asserts that import never
-- appears.

CREATE TABLE IF NOT EXISTS source_health (
    source               text PRIMARY KEY,
    consecutive_failures integer     NOT NULL DEFAULT 0,
    last_success_at      timestamptz,
    last_attempt_at      timestamptz NOT NULL DEFAULT now(),
    last_error           text,
    -- Set when consecutive_failures first crosses the threshold, and NOT moved
    -- on subsequent failures — it marks when degradation began, so "degraded
    -- for more than N days" is answerable.
    degraded_since       timestamptz,
    total_cycles         bigint      NOT NULL DEFAULT 0,

    CONSTRAINT source_health_failures_nonneg_check
        CHECK (consecutive_failures >= 0)
);
