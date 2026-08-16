"""The app's data access, and its own migration runner.

**Same database as the agents, separate tables.** One database because the app's
rows need real foreign keys into the pipeline's — a saved course referencing
`courses.course_id` cannot be enforced across two databases — and separate tables
because the agents own theirs.

The migration runner is deliberately a small self-contained copy rather than an
import of Agent B's. That is the precedent Agent D set explicitly: the
architecture forbids reaching into another agent's internals, and `db/migrate.py`
is an internal. The cost is ~30 duplicated lines; the benefit is that `api/` can
be deleted, rewritten or deployed separately without touching an agent.

Tracked in `schema_migrations_api`, so it cannot collide with the agents' — all
three migration sets start at 0001 in the one database.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any, Iterator, Optional

import psycopg
from psycopg.rows import dict_row

from shared.config import Config

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
TRACKING_TABLE = "schema_migrations_api"
_IDENT = re.compile(r"^[a-z_][a-z0-9_]*$")
_VERSIONED = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")


class MigrationError(RuntimeError):
    pass


def apply_migrations(dsn: str, *, directory: Path | None = None) -> list[str]:
    """Apply pending migrations; return what was applied. Idempotent."""
    if not _IDENT.match(TRACKING_TABLE):                      # pragma: no cover
        raise MigrationError(f"bad tracking table {TRACKING_TABLE!r}")
    directory = directory or MIGRATIONS_DIR
    files = sorted(p for p in directory.glob("*.sql") if _VERSIONED.match(p.name))

    applied: list[str] = []
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {TRACKING_TABLE} ("
                " version text PRIMARY KEY, checksum text NOT NULL,"
                " applied_at timestamptz NOT NULL DEFAULT now())"
            )
            cur.execute(f"SELECT version, checksum FROM {TRACKING_TABLE}")
            seen = {r[0]: r[1] for r in cur.fetchall()}
        conn.commit()

        for path in files:
            version = path.name[:-4]
            body = path.read_text(encoding="utf-8")
            checksum = hashlib.sha256(body.encode("utf-8")).hexdigest()
            if version in seen:
                if seen[version] != checksum:
                    # An applied migration that has since been edited: the
                    # database and the file no longer describe the same schema,
                    # and silently continuing is how that becomes undebuggable.
                    raise MigrationError(
                        f"{version} was applied with a different checksum; migrations are "
                        "immutable once applied — add a new one instead of editing it")
                continue
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute(body)
                    cur.execute(
                        f"INSERT INTO {TRACKING_TABLE} (version, checksum) VALUES (%s, %s)",
                        (version, checksum))
            applied.append(version)
    return applied


# ---------------------------------------------------------------------------
class AppStore:
    """Every SQL string the API issues lives here — the same rule the agents
    follow. Routes call methods; they never build SQL."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._conn: Optional[psycopg.Connection] = None

    @classmethod
    def from_config(cls, config: Config) -> "AppStore":
        return cls(config.require_database_url())

    def connect(self) -> psycopg.Connection:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self.dsn, row_factory=dict_row, autocommit=True)
        return self._conn

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
        self._conn = None

    @contextmanager
    def tx(self) -> Iterator[psycopg.Connection]:
        conn = self.connect()
        with conn.transaction():
            yield conn

    def _one(self, sql: str, params: Any = None) -> Optional[dict[str, Any]]:
        with self.connect().cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
        return dict(row) if row else None

    def _all(self, sql: str, params: Any = None) -> list[dict[str, Any]]:
        with self.connect().cursor() as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    def _exec(self, sql: str, params: Any = None) -> int:
        with self.connect().cursor() as cur:
            cur.execute(sql, params)
            return cur.rowcount

    # --- users --------------------------------------------------------------
    def create_user(self, *, email: str, full_name: str, password: str,
                    locale: str = "ar") -> dict[str, Any]:
        user_id = f"u_{secrets.token_hex(8)}"
        return self._one(
            """
            INSERT INTO app_users (user_id, email, full_name, password_hash, locale)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING user_id, email, full_name, locale, onboarded
            """,
            (user_id, email.strip().lower(), full_name.strip(), hash_password(password), locale),
        )

    def user_by_email(self, email: str) -> Optional[dict[str, Any]]:
        return self._one("SELECT * FROM app_users WHERE email = %s", (email.strip().lower(),))

    def user_by_id(self, user_id: str) -> Optional[dict[str, Any]]:
        """The signed-in user, WITHOUT the password hash.

        The column list is explicit rather than `*` on purpose: this is what
        every authenticated request resolves, and the hash has no business
        travelling with it. A new column therefore has to be added here
        deliberately — which is exactly what `avatar_path` needed, and its
        absence made a photo upload return 200 while the profile kept reporting
        no photo at all.
        """
        return self._one(
            "SELECT user_id, email, full_name, locale, onboarded, avatar_path, session_epoch "
            "FROM app_users WHERE user_id = %s",
            (user_id,))

    def set_locale(self, user_id: str, locale: str) -> None:
        self._exec("UPDATE app_users SET locale = %s WHERE user_id = %s", (locale, user_id))

    def mark_onboarded(self, user_id: str) -> None:
        self._exec("UPDATE app_users SET onboarded = true WHERE user_id = %s", (user_id,))

    # --- documents ----------------------------------------------------------
    def add_document(self, *, user_id: str, file_name: str, mime_type: str,
                     size_bytes: int, kind: str, stored_path: str) -> dict[str, Any]:
        document_id = f"doc_{secrets.token_hex(6)}"
        return self._one(
            """
            INSERT INTO app_documents
                (document_id, user_id, file_name, mime_type, size_bytes, kind, stored_path)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING document_id, file_name, mime_type, size_bytes, kind
            """,
            (document_id, user_id, file_name, mime_type, size_bytes, kind, stored_path),
        )

    def documents(self, user_id: str, ids: list[str]) -> list[dict[str, Any]]:
        """Scoped to the owner on purpose: a document id in someone else's
        request must not be readable just because it exists."""
        if not ids:
            return []
        return self._all(
            "SELECT * FROM app_documents WHERE user_id = %s AND document_id = ANY(%s)",
            (user_id, ids))

    # --- runs ---------------------------------------------------------------
    def create_run(self, *, user_id: str, run_id: str, document_ids: list[str]) -> str:
        job_id = f"job_{secrets.token_hex(8)}"
        self._exec(
            "INSERT INTO app_runs (job_id, user_id, run_id, document_ids) VALUES (%s,%s,%s,%s)",
            (job_id, user_id, run_id, json.dumps(document_ids)))
        return job_id

    def set_stage(self, job_id: str, stage: str, progress: float) -> None:
        """Called when a phase ACTUALLY finishes. Progress never advances on a
        clock, so a stuck stage reads as stuck rather than as slow."""
        self._exec(
            "UPDATE app_runs SET stage = %s, progress = %s WHERE job_id = %s",
            (stage, progress, job_id))

    def set_progress(self, job_id: str, stage: str, progress: float) -> None:
        """A fine-grained checkpoint: one graph node finished.

        `GREATEST` is the point. This is called ~22 times per run from a worker
        thread, and a bar that ever moves BACKWARDS is worse than a coarse one —
        it reads as the system losing work. Two nodes completing out of order, or a
        retried step, can only ever leave the progress where the further-along one
        put it.

        The stage still moves freely: it is a label for what is happening now, not
        a measure of how much is done.
        """
        self._exec(
            "UPDATE app_runs SET stage = %s, progress = GREATEST(progress, %s::numeric) "
            "WHERE job_id = %s", (stage, progress, job_id))

    def attach_profile(self, job_id: str, profile: Any) -> None:
        """End of phase one: Agent A's envelope is stored and the run pauses.

        The profile is written BEFORE the stage moves, so a poll that arrives
        between the two never sees `awaiting_confirmation` without the result the
        confirm screen needs — the UI would render an empty form over real data.
        """
        self._exec("UPDATE app_runs SET profile = %s WHERE job_id = %s",
                   (json.dumps(profile) if profile is not None else None, job_id))

    def awaiting_run(self, user_id: str) -> Optional[dict[str, Any]]:
        """The run waiting on this user's confirmation, if any.

        Newest first: a user who re-uploaded has two paused runs, and the one they
        are looking at is the recent one. Returning None is a normal state, not an
        error — the manual-entry route has no run at all.
        """
        return self._one(
            "SELECT * FROM app_runs WHERE user_id = %s AND stage = 'awaiting_confirmation' "
            "ORDER BY started_at DESC LIMIT 1", (user_id,))

    def save_run_preferences(self, job_id: str, preferences: Any) -> None:
        """What the user answered, recorded against the run it is about to shape."""
        self._exec("UPDATE app_runs SET preferences = %s WHERE job_id = %s",
                   (json.dumps(preferences) if preferences is not None else None, job_id))

    def finish_run(self, job_id: str, *, profile: Any, skill_gap: Any,
                   recommendations: Any) -> None:
        # COALESCE on `profile`: phase two finishes a run whose profile was already
        # written by `attach_profile`, and it does not carry it again. Passing None
        # must leave the stored envelope alone — writing NULL over it would erase
        # Agent A's whole output at the moment the run succeeds.
        self._exec(
            """
            UPDATE app_runs
               SET stage = 'done', progress = 1.0, finished_at = now(),
                   profile = COALESCE(%s::jsonb, profile),
                   skill_gap = %s, recommendations = %s
             WHERE job_id = %s
            """,
            (json.dumps(profile) if profile is not None else None,
             json.dumps(skill_gap) if skill_gap is not None else None,
             json.dumps(recommendations) if recommendations is not None else None,
             job_id))

    def fail_run(self, job_id: str, error_code: str) -> None:
        self._exec(
            "UPDATE app_runs SET stage='failed', error_code=%s, finished_at=now() "
            "WHERE job_id = %s", (error_code, job_id))

    def run(self, *, job_id: str, user_id: str) -> Optional[dict[str, Any]]:
        return self._one(
            "SELECT * FROM app_runs WHERE job_id = %s AND user_id = %s", (job_id, user_id))

    def latest_completed_run(self, user_id: str) -> Optional[dict[str, Any]]:
        """The newest run that actually produced envelopes — the dashboard reads
        this, so a failed or in-flight run never replaces a good earlier one."""
        return self._one(
            "SELECT * FROM app_runs WHERE user_id = %s AND stage = 'done' "
            "ORDER BY finished_at DESC NULLS LAST LIMIT 1", (user_id,))

    # --- onboarding progress / confirmed profile ----------------------------
    def get_progress(self, user_id: str) -> Optional[Any]:
        row = self._one("SELECT payload FROM app_onboarding_progress WHERE user_id = %s",
                        (user_id,))
        return row["payload"] if row else None

    def put_progress(self, user_id: str, payload: Any) -> None:
        self._exec(
            """
            INSERT INTO app_onboarding_progress (user_id, payload, updated_at)
            VALUES (%s, %s, now())
            ON CONFLICT (user_id) DO UPDATE
               SET payload = EXCLUDED.payload, updated_at = now()
            """,
            (user_id, json.dumps(payload)))

    def clear_progress(self, user_id: str) -> None:
        self._exec("DELETE FROM app_onboarding_progress WHERE user_id = %s", (user_id,))

    def save_profile(self, user_id: str, payload: Any, run_id: Optional[str]) -> None:
        self._exec(
            """
            INSERT INTO app_profiles (user_id, payload, run_id, confirmed_at)
            VALUES (%s, %s, %s, now())
            ON CONFLICT (user_id) DO UPDATE
               SET payload = EXCLUDED.payload, run_id = EXCLUDED.run_id, confirmed_at = now()
            """,
            (user_id, json.dumps(payload), run_id))

    def profile(self, user_id: str) -> Optional[dict[str, Any]]:
        return self._one("SELECT * FROM app_profiles WHERE user_id = %s", (user_id,))

    def all_documents(self, user_id: str) -> list[dict[str, Any]]:
        """Every document this user has uploaded, newest first.

        Deliberately a separate method rather than letting `documents()` take an
        empty id list to mean "all": that method's whole point is scoping a read
        to specific ids the caller named, and loosening it so a missing filter
        returns everything is the wrong direction for a method that exists to
        stop one user reading another's files.
        """
        return self._all(
            "SELECT * FROM app_documents WHERE user_id = %s ORDER BY created_at DESC",
            (user_id,))

    def set_avatar_path(self, user_id: str, path: Optional[str]) -> None:
        """Where this user's photo lives on disk, or None once removed.

        On the ACCOUNT, like `onboarded`, because a photo set on a phone must
        still be there on a laptop. Written only by the avatar routes — never by
        a profile save, so storing a corrected birth date can never blank
        someone's picture as a side effect.
        """
        self._exec("UPDATE app_users SET avatar_path = %s WHERE user_id = %s",
                   (path, user_id))

    # --- password recovery ---------------------------------------------------
    #
    # The raw token never enters this class. Callers hash it first and pass the
    # hash, so there is no code path here that could log or store the credential
    # itself — see `api/email.py` for where the token does exist, briefly.

    def create_password_reset(self, *, user_id: str, token_hash: str,
                              minutes: int) -> None:
        self._exec(
            "INSERT INTO app_password_resets (token_hash, user_id, expires_at) "
            "VALUES (%s, %s, now() + make_interval(mins => %s))",
            (token_hash, user_id, minutes))

    def consume_password_reset(self, token_hash: str) -> Optional[dict[str, Any]]:
        """Spend a token, or return None if it cannot be spent.

        One statement, and that is the point: checking validity and marking the
        token used in separate steps leaves a window where two requests both see
        an unused token and both reset the password. `used_at IS NULL` inside the
        UPDATE closes it, exactly as `claim_quota` does for the daily limit.

        None covers every failure — unknown, expired, already used — because the
        caller must answer all three identically. A response that distinguished
        "expired" from "never existed" would confirm that a token had once been
        issued for an address.
        """
        return self._one(
            "UPDATE app_password_resets SET used_at = now() "
            " WHERE token_hash = %s AND used_at IS NULL AND expires_at > now() "
            "RETURNING user_id",
            (token_hash,))

    def invalidate_password_resets(self, user_id: str) -> int:
        """Expire every outstanding token for this user.

        Called after a successful reset. Two "forgot password" clicks issue two
        tokens, and spending one must not leave the other usable — otherwise an
        attacker who triggered a reset earlier still holds a key to an account
        whose owner believes they have just secured it.
        """
        return self._exec(
            "UPDATE app_password_resets SET used_at = now() "
            " WHERE user_id = %s AND used_at IS NULL",
            (user_id,))

    def claim_reset_slot(self, subject_hash: str, *, kind: str, limit: int) -> bool:
        """True if this request is within the hourly limit, having consumed one.

        A guarded UPDATE for the same reason `claim_quota` is one: checking a
        count and then incrementing it is a race, and here losing it means an
        inbox gets flooded.

        `subject_hash` is a hash and must stay one — see the migration. The
        period is truncated to the hour in SQL rather than in Python so that two
        processes with different clocks still agree on which bucket they are in.
        """
        self._exec(
            "INSERT INTO app_reset_throttle (subject, kind, period_start, used) "
            "VALUES (%s, %s, date_trunc('hour', now()), 0) ON CONFLICT DO NOTHING",
            (subject_hash, kind))
        row = self._one(
            "UPDATE app_reset_throttle SET used = used + 1 "
            " WHERE subject = %s AND kind = %s AND period_start = date_trunc('hour', now()) "
            "   AND used < %s "
            "RETURNING used",
            (subject_hash, kind, limit))
        return row is not None

    def set_password(self, user_id: str, password_hash: str) -> None:
        self._exec("UPDATE app_users SET password_hash = %s WHERE user_id = %s",
                   (password_hash, user_id))

    def bump_session_epoch(self, user_id: str) -> int:
        """Invalidate every session cookie issued to this account so far.

        The epoch is signed into the session token and compared on every request,
        so incrementing it here evicts anyone holding an older cookie. Without
        this, resetting a password would leave a stolen session working — the
        feature failing at the one case it exists for.
        """
        row = self._one(
            "UPDATE app_users SET session_epoch = session_epoch + 1 "
            " WHERE user_id = %s RETURNING session_epoch",
            (user_id,))
        return row["session_epoch"] if row else 0

    # --- Agent S: quotas and conversation ------------------------------------
    #
    # `claim_quota` is the only thing standing between a chat box and an
    # unbounded model bill, and it is the reason the usage table exists at all.
    # Read the docstring before changing either.

    def claim_quota(self, user_id: str, *, kind: str, limit: int,
                    period_start: date) -> Optional[int]:
        """Spend one unit of quota. Returns the new count, or None if over.

        **This must stay a single guarded UPDATE.** The obvious alternative —
        count the rows, compare, then insert — is a read and a write with a gap
        between them, and two requests interleaving in that gap both see "9 of
        10 used" and both proceed. That is not a theoretical race: FastAPI runs
        sync handlers in a threadpool, so concurrent messages from one user are
        the normal case, not the exotic one.

        `used < %s` inside the UPDATE makes the check and the increment the same
        statement, which is atomic under any isolation level and needs no lock.
        Zero rows affected means the limit was already reached.

        The INSERT is separate and deliberately `ON CONFLICT DO NOTHING`: it
        creates the period's row the first time it is needed. A new day or week
        is simply a new row, so there is no reset job to schedule and none to
        fail silently.

        Callers must claim BEFORE doing the work, and must call `refund_quota`
        if the work then fails — see `POST /api/assistant/messages`. The spec's
        rule is that usage logs only on success, and the honest way to get that
        is to make the failure path give the unit back rather than to check the
        limit, do the work, and increment afterwards (which is the same race
        wearing a different hat).
        """
        self._exec(
            "INSERT INTO app_assistant_usage (user_id, kind, period_start, used) "
            "VALUES (%s, %s, %s, 0) ON CONFLICT DO NOTHING",
            (user_id, kind, period_start))
        row = self._one(
            "UPDATE app_assistant_usage SET used = used + 1 "
            " WHERE user_id = %s AND kind = %s AND period_start = %s AND used < %s "
            "RETURNING used",
            (user_id, kind, period_start, limit))
        return row["used"] if row else None

    def refund_quota(self, user_id: str, *, kind: str, period_start: date) -> None:
        """Give back a claimed unit because the work did not happen.

        Never below zero — the CHECK constraint would reject it and the guard
        here means a double refund is a no-op rather than an error. This is the
        only method that decreases usage, and it exists so that a model timeout
        does not cost someone one of their ten daily messages.
        """
        self._exec(
            "UPDATE app_assistant_usage SET used = used - 1 "
            " WHERE user_id = %s AND kind = %s AND period_start = %s AND used > 0",
            (user_id, kind, period_start))

    def quota_used(self, user_id: str, *, kind: str, period_start: date) -> int:
        row = self._one(
            "SELECT used FROM app_assistant_usage "
            " WHERE user_id = %s AND kind = %s AND period_start = %s",
            (user_id, kind, period_start))
        return row["used"] if row else 0

    def add_assistant_message(self, *, user_id: str, role: str, content: str,
                              run_id: Optional[str] = None,
                              answer_source: Optional[str] = None) -> dict[str, Any]:
        return self._one(
            "INSERT INTO app_assistant_messages "
            "  (message_id, user_id, run_id, role, content, answer_source) "
            "VALUES (%s, %s, %s, %s, %s, %s) RETURNING *",
            (f"m_{secrets.token_hex(8)}", user_id, run_id, role, content, answer_source))

    def assistant_history(self, user_id: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """This user's turns, oldest first.

        Scoped to `user_id` with no parameter that could widen it — the same
        reason `all_documents` exists as its own method. Agent S's isolation is
        not the model declining to discuss other people; it is that no method
        here can fetch them.
        """
        rows = self._all(
            "SELECT * FROM app_assistant_messages WHERE user_id = %s "
            "ORDER BY created_at DESC, message_id DESC LIMIT %s",
            (user_id, limit))
        return list(reversed(rows))


# ---------------------------------------------------------------------------
# passwords — scrypt from the stdlib, so no new dependency and no plaintext
# ---------------------------------------------------------------------------
_SCRYPT = {"n": 2 ** 14, "r": 8, "p": 1}


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode("utf-8"), salt=salt, dklen=32, **_SCRYPT)
    return f"scrypt${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, salt_hex, want_hex = stored.split("$", 2)
        if scheme != "scrypt":
            return False
        dk = hashlib.scrypt(password.encode("utf-8"), salt=bytes.fromhex(salt_hex),
                            dklen=32, **_SCRYPT)
    except (ValueError, TypeError):
        return False
    # Constant-time: a timing difference here leaks whether a prefix matched.
    return secrets.compare_digest(dk.hex(), want_hex)
