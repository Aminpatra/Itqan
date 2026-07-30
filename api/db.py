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
        return self._one(
            "SELECT user_id, email, full_name, locale, onboarded FROM app_users WHERE user_id = %s",
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
