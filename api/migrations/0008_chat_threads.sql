-- Hud's chat: threads, and what a turn carries besides its text.
--
-- Agent S shipped with ONE flat message log per user, which is the right shape
-- for a Q&A box and the wrong one for a conversation surface with a sidebar of
-- recents. The front end asks for `GET /api/chat/threads` and
-- `/api/chat/threads/:id`; without a thread key there is nothing to answer with.
CREATE TABLE IF NOT EXISTS app_chat_threads (
    thread_id  text PRIMARY KEY,
    user_id    text        NOT NULL REFERENCES app_users(user_id) ON DELETE CASCADE,
    -- The first question, trimmed. Set once and never rewritten: a title that
    -- changed as a conversation went on would make the sidebar unrecognisable
    -- from one visit to the next.
    title      text        NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS app_chat_threads_user_idx
    ON app_chat_threads (user_id, updated_at DESC);


-- What a turn carries beyond its text.
--
-- `thread_id` is NULLABLE and existing rows keep NULL. They are the CLI-era
-- flat history and are left exactly as they are rather than back-filled into a
-- thread that never existed — inventing a conversation to make a foreign key
-- tidy would put words into a structure the user never saw.
ALTER TABLE app_assistant_messages
    ADD COLUMN IF NOT EXISTS thread_id   text REFERENCES app_chat_threads(thread_id) ON DELETE CASCADE,
    -- The jobs and courses ATTACHED to this turn, exactly as they were rendered.
    --
    -- Stored rather than re-resolved on read, and that is the deliberate half.
    -- Re-resolving would silently delete cards from history the day a posting
    -- expires — rewriting what Hud said, months later, with nobody able to tell.
    -- A snapshot is honest here BECAUSE every card already carries its own
    -- `source.retrievedAt`, so an old turn reads as "as at that date" instead of
    -- claiming to be live.
    ADD COLUMN IF NOT EXISTS cards       jsonb,
    ADD COLUMN IF NOT EXISTS suggestions jsonb,
    -- METADATA ONLY — name, type, size. Never bytes, and never a document id.
    -- A file dropped into a conversation must not become the document the
    -- pipeline runs on: that route is POST /api/documents and it exists because
    -- it has a human confirmation screen in the middle of it.
    ADD COLUMN IF NOT EXISTS attachments jsonb,
    ADD COLUMN IF NOT EXISTS rating      text;

DO $$
BEGIN
    ALTER TABLE app_assistant_messages
        ADD CONSTRAINT app_assistant_messages_rating_check
        CHECK (rating IN ('up', 'down'));
EXCEPTION
    WHEN duplicate_object THEN NULL;   -- re-applied migration; the rule is already there
END $$;


-- A third kind of answer: the daily limit, spoken by Hud.
--
-- It is neither 'model' (nothing was generated) nor 'template' (that means the
-- verifier rejected a real answer), and conflating it with either would make the
-- one number worth watching — how often the fallback fires — unreadable. The
-- constraint caught this the first time the branch ran, which is the argument
-- for having it.
ALTER TABLE app_assistant_messages
    DROP CONSTRAINT IF EXISTS app_assistant_messages_answer_source_check;
ALTER TABLE app_assistant_messages
    ADD CONSTRAINT app_assistant_messages_answer_source_check
    CHECK (answer_source IN ('model', 'template', 'limit'));

CREATE INDEX IF NOT EXISTS app_assistant_messages_thread_idx
    ON app_assistant_messages (thread_id, created_at);
