-- Recommendation feedback: the like and dislike on every job and course card.
--
-- APPEND-ONLY, and the current verdict is derived rather than stored twice.
-- Changing your mind is a new row, not an UPDATE, so the history stays readable:
-- "liked it, then disliked it after reading the location" is a different signal
-- from "disliked it", and an UPDATE-in-place throws that difference away for the
-- sake of one fewer row.
--
-- The point of all of it is the NEXT run. A verdict that lives in the browser
-- tab teaches nothing and reappears as the same card after a reload, which reads
-- as the product ignoring the person — `BACKEND.md` §1.5 is explicit about that
-- being the failure to avoid.
CREATE TABLE IF NOT EXISTS app_feedback (
    feedback_id text        PRIMARY KEY,
    user_id     text        NOT NULL REFERENCES app_users(user_id) ON DELETE CASCADE,
    subject     text        NOT NULL CHECK (subject IN ('job', 'course')),
    -- The job or course id as the service issued it. NOT a foreign key: job
    -- postings expire and are pruned, and a posting disappearing must not take
    -- the user's opinion of it with it — nor block the delete.
    item_id     text        NOT NULL,
    verdict     text        NOT NULL CHECK (verdict IN ('like', 'dislike')),
    -- From a closed list the client sends verbatim; translated in the front end,
    -- never on the wire. Nullable because a like has no reason, and because an
    -- unrecognised one is dropped rather than allowed to lose the whole verdict.
    reason      text,
    -- Free text, only when reason = 'other'. The user's own words, never parsed.
    note        text,
    -- The user asked for a replacement rather than only registering the dislike.
    -- Courses only: a posting is a real vacancy, not an interchangeable slot.
    replaced    boolean     NOT NULL DEFAULT false,
    created_at  timestamptz NOT NULL DEFAULT now()
);

-- The read this table exists for: "what is this user's latest verdict on each
-- item", answered per user, newest first.
CREATE INDEX IF NOT EXISTS app_feedback_latest_idx
    ON app_feedback (user_id, subject, item_id, created_at DESC);
