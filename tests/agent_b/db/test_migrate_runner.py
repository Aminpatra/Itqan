"""The migration runner against a real database.

Idempotency and the checksum guard are the two properties that matter here, and
neither can be verified without Postgres: transactional DDL is what makes
"record the version in the same transaction as its DDL" possible at all.
"""

from __future__ import annotations

import os

import psycopg
import pytest

from agents.agent_b_job_ingest.db.migrate import MigrationError, apply_migrations, pending

DSN = os.getenv("ITQAN_TEST_DATABASE_URL", "")


@pytest.fixture
def scratch(tmp_path):
    """A throwaway migration directory, cleaned out of the DB afterwards."""
    table = "mig_probe_" + os.urandom(4).hex()
    yield tmp_path, table

    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {table}")
            cur.execute("DELETE FROM schema_migrations WHERE version LIKE '9%%'")
        conn.commit()


def test_applying_twice_is_a_no_op(migrated_dsn):
    """The phase gate in one assertion: a second run must apply nothing.

    Without this, a scheduled process that migrates on startup would re-run DDL
    on every cycle.
    """
    assert pending(migrated_dsn) == []
    assert apply_migrations(migrated_dsn) == []


def test_editing_an_applied_migration_is_an_error_not_a_silent_skip(scratch, migrated_dsn):
    """The failure this prevents: someone edits a migration that already ran,
    their database keeps the old shape, a colleague's fresh database gets the
    new one, and both believe they are on the same schema.
    """
    tmp_path, table = scratch
    path = tmp_path / "9001_probe.sql"

    path.write_text(f"CREATE TABLE {table} (id int);", encoding="utf-8")
    assert apply_migrations(migrated_dsn, directory=tmp_path) == ["9001_probe"]

    # Same version, different contents.
    path.write_text(f"CREATE TABLE {table} (id int, extra text);", encoding="utf-8")

    with pytest.raises(MigrationError, match="contents have changed"):
        apply_migrations(migrated_dsn, directory=tmp_path)


def test_a_failing_migration_leaves_no_partial_state(scratch, migrated_dsn):
    """One transaction per migration: DDL and its version row commit together,
    so a failure cannot leave the schema ahead of what schema_migrations says.
    """
    tmp_path, table = scratch
    (tmp_path / "9002_broken.sql").write_text(
        f"CREATE TABLE {table} (id int); SELECT this_function_does_not_exist();",
        encoding="utf-8",
    )

    with pytest.raises(MigrationError, match="failed"):
        apply_migrations(migrated_dsn, directory=tmp_path)

    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s) AS t", (table,))
        assert cur.fetchone()[0] is None, "the table survived a failed migration"
        cur.execute("SELECT count(*) FROM schema_migrations WHERE version = '9002'")
        assert cur.fetchone()[0] == 0, "a failed migration was recorded as applied"
