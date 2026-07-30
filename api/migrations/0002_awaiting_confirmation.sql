-- A run now PAUSES between Agent A and Agent C.
--
-- The first version ran A -> C -> E in one pass, so the only states a run could
-- be in were "working" and "finished". That made two things impossible: the
-- confirm screen could not show the user their extracted details until the
-- course recommender had also finished (~3 minutes of skeleton for ~90 seconds
-- of relevant work), and the answers the user gave during that wait could not
-- reach Agent C, because Agent C had already run.
--
-- `awaiting_confirmation` is the pause. It is deliberately NOT a progress state
-- in the same sense as the others: every other stage advances when work
-- finishes, this one advances when a PERSON acts. That distinction matters
-- operationally — `jobs.stale_runs` looks for runs stuck mid-flight, and a user
-- taking ten minutes over a form is not a crashed process.
ALTER TABLE app_runs DROP CONSTRAINT IF EXISTS app_runs_stage_check;

ALTER TABLE app_runs
    ADD CONSTRAINT app_runs_stage_check
    CHECK (stage IN ('queued', 'reading', 'translating', 'awaiting_confirmation',
                     'matching', 'done', 'failed'));

-- What the user confirmed and answered, as it was when phase two was started.
--
-- The confirmed profile is already stored on `app_profiles`, but that row is
-- the CURRENT truth and is overwritten if the user edits their details later.
-- This column is the input to THIS run, so a gap file can be explained by the
-- preferences that actually produced it — the same reason every agent publishes
-- a `calibration` block.
ALTER TABLE app_runs ADD COLUMN IF NOT EXISTS preferences jsonb;
