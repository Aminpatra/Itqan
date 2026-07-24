-- Per-source cycle history for Agent D's course sources. Mirror of Agent B's
-- source_health, kept separate so each agent owns its own operational
-- bookkeeping. Read and written ONLY in Agent D's runlog node — nothing under
-- sources/ imports it, so no code path can react to a recorded failure by
-- retrying harder. Not part of any consumer contract.

CREATE TABLE IF NOT EXISTS course_source_health (
    source               text PRIMARY KEY,
    consecutive_failures integer     NOT NULL DEFAULT 0,
    last_success_at      timestamptz,
    last_attempt_at      timestamptz NOT NULL DEFAULT now(),
    last_error           text,
    degraded_since       timestamptz,
    total_cycles         bigint      NOT NULL DEFAULT 0,

    CONSTRAINT course_source_health_failures_nonneg_check
        CHECK (consecutive_failures >= 0)
);
