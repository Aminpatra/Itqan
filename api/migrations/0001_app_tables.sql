-- The web app's own tables, in the SAME database as the agents.
--
-- One database, not two: a saved course or a run needs a real foreign key to
-- `courses.course_id` / `job_postings.posting_id`, and cross-database references
-- do not exist. The agents' tables are untouched here — this migration only adds
-- what the app owns, tracked in its own `schema_migrations_api` so it can never
-- collide with Agent B's or Agent D's numbering.

CREATE TABLE IF NOT EXISTS app_users (
    user_id        text PRIMARY KEY,
    email          text        NOT NULL UNIQUE,
    full_name      text        NOT NULL DEFAULT '',
    -- scrypt, via hashlib — no external dependency, and never the raw password.
    password_hash  text        NOT NULL,
    locale         text        NOT NULL DEFAULT 'ar' CHECK (locale IN ('ar', 'en')),
    -- Server-owned, per the frontend's own contract: it decides whether the user
    -- lands on onboarding or the dashboard, so it must live on the row and not in
    -- a cookie. Finishing on a phone and returning on a laptop must not restart.
    onboarded      boolean     NOT NULL DEFAULT false,
    created_at     timestamptz NOT NULL DEFAULT now()
);

-- Uploaded files. Bytes live on disk under `api_upload_dir`; this is the index.
CREATE TABLE IF NOT EXISTS app_documents (
    document_id  text PRIMARY KEY,
    user_id      text        NOT NULL REFERENCES app_users(user_id) ON DELETE CASCADE,
    file_name    text        NOT NULL,
    mime_type    text        NOT NULL,
    size_bytes   bigint      NOT NULL,
    -- The UI's DocumentKind. `cv` is the required one (Agent A requires --cv and
    -- treats --transcript as optional).
    kind         text        NOT NULL,
    stored_path  text        NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS app_documents_user_idx ON app_documents (user_id, created_at DESC);

-- One analysis run = one A -> C -> E pass.
--
-- `stage` and `progress` are what the UI's progress bar reads, and they are
-- written when a phase ACTUALLY completes rather than on a timer — a stalled
-- stage stays put, which is the honest signal. `run_id` is the pipeline's own id,
-- so the artifacts on disk and this row share one name.
CREATE TABLE IF NOT EXISTS app_runs (
    job_id        text PRIMARY KEY,
    user_id       text        NOT NULL REFERENCES app_users(user_id) ON DELETE CASCADE,
    run_id        text        NOT NULL,
    stage         text        NOT NULL DEFAULT 'queued'
                  CHECK (stage IN ('queued','reading','translating','matching','done','failed')),
    progress      real        NOT NULL DEFAULT 0,
    -- Names the agent that failed, e.g. 'agent_a_unreadable_document', so the UI
    -- can offer the right recovery instead of a generic apology.
    error_code    text,
    document_ids  jsonb       NOT NULL DEFAULT '[]'::jsonb,
    -- The three envelopes, stored as written. jsonb rather than a file path: the
    -- app must be able to serve them after a redeploy and from any instance.
    profile       jsonb,
    skill_gap     jsonb,
    recommendations jsonb,
    started_at    timestamptz NOT NULL DEFAULT now(),
    finished_at   timestamptz
);
CREATE INDEX IF NOT EXISTS app_runs_user_idx ON app_runs (user_id, started_at DESC);

-- Resumable onboarding, keyed by user rather than held in a 4 KB cookie, so the
-- flow survives a device change. One row per user; the UI PUTs the whole blob.
CREATE TABLE IF NOT EXISTS app_onboarding_progress (
    user_id     text PRIMARY KEY REFERENCES app_users(user_id) ON DELETE CASCADE,
    payload     jsonb       NOT NULL,
    updated_at  timestamptz NOT NULL DEFAULT now()
);

-- What the user CONFIRMED, which is what drives everything downstream — not the
-- raw extraction. Kept separately from app_runs for exactly that reason.
CREATE TABLE IF NOT EXISTS app_profiles (
    user_id      text PRIMARY KEY REFERENCES app_users(user_id) ON DELETE CASCADE,
    payload      jsonb       NOT NULL,
    run_id       text,
    confirmed_at timestamptz NOT NULL DEFAULT now()
);
