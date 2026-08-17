-- Email verification at signup: prove the address before onboarding proceeds.
--
-- Until now anyone could register with any address and go straight into the
-- pipeline. The address is the only durable identifier on an account — it is
-- where recovery sends its link and how a person gets back to a CV and a set of
-- results they have already produced — so an address nobody has proved makes
-- recovery unrecoverable.
--
-- NULL means unverified. There is no backfill clause here because every existing
-- row is being deleted (user decision, 2026-08-17), so there is nothing to
-- grandfather. Had those accounts been kept, this file would have had to set
-- them verified EXPLICITLY: a new nullable column reads as "nobody has ever
-- verified", which would have locked out every existing user on deploy.
ALTER TABLE app_users ADD COLUMN IF NOT EXISTS email_verified_at timestamptz;


-- One outstanding code per account, and the primary key says so.
--
-- Keyed on `user_id` rather than on the code, which is the load-bearing
-- decision here and the opposite of `app_password_resets`:
--
--   * a resend REPLACES its predecessor by upsert, so the code in the newest
--     email is the only one that works. Keyed on the code instead, every resend
--     would add a live credential and "which of my four codes is valid" becomes
--     a real question with an embarrassing answer;
--   * it gives `attempts` a stable home. Counting attempts per CODE would let an
--     attacker reset their own counter by asking for a new one, which makes the
--     limit decorative — and the limit is the entire defence (see below).
--
-- THE CODE IS NEVER STORED, only `sha256(code)`, consistent with the reset
-- tokens. But state the strength honestly rather than inheriting that table's
-- reasoning: a reset token is 256 bits from `secrets.token_urlsafe(32)` and its
-- hash is genuinely one-way in practice. Six digits is a million possibilities,
-- and an attacker holding this table enumerates every preimage in under a
-- second. The hash is hygiene — it keeps a working credential out of a leak, a
-- backup and a query result — not a defence.
--
-- WHAT ACTUALLY STOPS GUESSING is `attempts`: the fifth wrong answer kills the
-- code, so any attacker gets 5 tries in 1,000,000 per issued code, and asking
-- for a new one starts a new code rather than a fresh allowance against the old.
CREATE TABLE IF NOT EXISTS app_email_verifications (
    user_id     text PRIMARY KEY REFERENCES app_users(user_id) ON DELETE CASCADE,
    code_hash   text        NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    expires_at  timestamptz NOT NULL,
    attempts    integer     NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    consumed_at timestamptz
);


-- Resends reuse the delivered rate limiter rather than growing a second one.
--
-- `app_reset_throttle` already enforces "N per subject per hour" with a guarded
-- UPDATE, and its CHECK is what has to move to let a second caller in. The two
-- new kinds are separate buckets on purpose: a person's password-reset budget
-- and their verification-resend budget should not consume each other, or asking
-- twice for a code would quietly cost them a recovery attempt.
ALTER TABLE app_reset_throttle DROP CONSTRAINT IF EXISTS app_reset_throttle_kind_check;
ALTER TABLE app_reset_throttle ADD CONSTRAINT app_reset_throttle_kind_check
    CHECK (kind IN ('email', 'ip', 'verify_user', 'verify_ip'));
