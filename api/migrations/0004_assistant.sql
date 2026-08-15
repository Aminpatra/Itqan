-- Agent S: the conversation, and the quotas that bound it.
--
-- Two tables doing two different jobs, and the split is deliberate.
--
-- `app_assistant_messages` is the RECORD: what was asked, what was answered,
-- which run it was about. It is the audit trail and the conversation history.
--
-- `app_assistant_usage` is the ENFORCEMENT, and it exists because the tidier
-- design cannot be made correct. Counting rows in the messages table to decide
-- whether someone is under their daily limit is a read followed by a write, and
-- two requests interleaving between the two both see "9 of 10 used" and both
-- proceed. A single guarded UPDATE cannot be raced:
--
--     UPDATE app_assistant_usage SET used = used + 1
--      WHERE user_id = %s AND kind = %s AND period_start = %s AND used < %s
--     RETURNING used;          -- zero rows back  ==  over the limit
--
-- One statement, atomic under any isolation level, no lock to manage. So the
-- counter enforces and the messages table records; each does the thing it is
-- actually good at, and neither is asked to do the other's job.
--
-- A row per (user, kind, period) rather than a decrementing balance means there
-- is NO RESET JOB: a new day or week is simply a new row, so nothing has to run
-- at midnight and nothing can fail silently at midnight. "Resets at ..." is
-- computed from the period, never stored.
--
-- `period_start` is a DATE in Asia/Muscat, not UTC — see config.assistant_tz.
-- Telling a user in Oman their quota resets at midnight and having it reset at
-- 4am is a bug, and defaulting to UTC is exactly how you get it.

CREATE TABLE IF NOT EXISTS app_assistant_messages (
    message_id  text PRIMARY KEY,
    user_id     text        NOT NULL REFERENCES app_users(user_id) ON DELETE CASCADE,
    -- The completed run this turn was talking about. Null before any run has
    -- finished, which is a legitimate state: a user may ask a question before
    -- their first matching completes.
    run_id      text,
    role        text        NOT NULL CHECK (role IN ('user', 'assistant')),
    content     text        NOT NULL,
    -- How the answer was produced: 'model' when the model's text was published,
    -- 'template' when verification rejected it and the deterministic sentence
    -- was used instead. Null on a user turn. Same telemetry as Agent E's
    -- `rationale_source`, and for the same reason: a silent fallback that
    -- nobody can count is a fallback nobody notices has become the norm.
    answer_source text      CHECK (answer_source IN ('model', 'template')),
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS app_assistant_messages_user_idx
    ON app_assistant_messages (user_id, created_at DESC);


CREATE TABLE IF NOT EXISTS app_assistant_usage (
    user_id      text    NOT NULL REFERENCES app_users(user_id) ON DELETE CASCADE,
    kind         text    NOT NULL CHECK (kind IN ('message', 'rerun')),
    period_start date    NOT NULL,
    -- CHECK (used >= 0) is not decoration: it is the last line of defence if a
    -- caller ever writes a decrement path. Nothing may hand out credits.
    used         integer NOT NULL DEFAULT 0 CHECK (used >= 0),
    PRIMARY KEY (user_id, kind, period_start)
);
