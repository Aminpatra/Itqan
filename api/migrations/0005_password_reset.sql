-- Password recovery: reset tokens, and the counter that makes a reset mean
-- something.
--
-- THE TOKEN IS NEVER STORED. Only `sha256(token)` is, so a database leak yields
-- no working reset links. A fast hash is the right choice here and scrypt would
-- be wrong: this is 256 bits from `secrets.token_urlsafe(32)`, not a guessable
-- human secret, so there is nothing to slow an attacker down — they would have
-- to brute-force the keyspace, not a password list.
--
-- `used_at` makes a token single-use, and a successful reset also expires every
-- OTHER outstanding token for that user: two "forgot password" clicks must not
-- leave a spare key lying around.
CREATE TABLE IF NOT EXISTS app_password_resets (
    token_hash text        PRIMARY KEY,
    user_id    text        NOT NULL REFERENCES app_users(user_id) ON DELETE CASCADE,
    created_at timestamptz NOT NULL DEFAULT now(),
    expires_at timestamptz NOT NULL,
    used_at    timestamptz
);

CREATE INDEX IF NOT EXISTS app_password_resets_user_idx
    ON app_password_resets (user_id);


-- Rate limiting for an endpoint anyone can call, for any address.
--
-- Same shape as `app_assistant_usage` and for the same reason: a guarded
-- `UPDATE ... WHERE used < limit` is atomic, where count-then-insert is a race.
-- A new hour is simply a new row, so nothing resets anything.
--
-- `subject` is a **HASH**, never the address or the IP. This table would
-- otherwise become a list of who has tried to recover an account — which is
-- precisely the information the endpoint's identical responses exist to protect,
-- handed over by the logging instead.
CREATE TABLE IF NOT EXISTS app_reset_throttle (
    subject      text        NOT NULL,          -- sha256(email) or sha256(ip)
    kind         text        NOT NULL CHECK (kind IN ('email', 'ip')),
    period_start timestamptz NOT NULL,          -- truncated to the hour
    used         integer     NOT NULL DEFAULT 0 CHECK (used >= 0),
    PRIMARY KEY (subject, kind, period_start)
);


-- Which generation of session cookies is still valid for this account.
--
-- Session tokens were `user_id + HMAC(secret, user_id)` — deterministic and
-- permanent, so a cookie captured once worked forever and changing the password
-- did nothing to it. That made password recovery fail at precisely the case it
-- exists for: somebody else has my account.
--
-- The epoch is now part of the signed token and is compared against this column
-- on every request, so bumping it here evicts every session issued before the
-- reset. Existing rows default to 0, which matches the epoch newly-issued
-- cookies carry — but the token FORMAT changed, so everyone signed in when this
-- deploys is signed out once. That was accepted deliberately (2026-08-16).
ALTER TABLE app_users ADD COLUMN IF NOT EXISTS session_epoch integer NOT NULL DEFAULT 0;
