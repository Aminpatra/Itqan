"""Postgres-backed tests.

Skipped entirely unless ``ITQAN_TEST_DATABASE_URL`` is set, so the default
``pytest tests/ -q`` still runs anywhere with no infrastructure. To enable:

    docker run -d --name itqan-pg -p 5432:5432 \
        -e POSTGRES_PASSWORD=itqan_dev -e POSTGRES_DB=itqan \
        -v itqan_pgdata:/var/lib/postgresql/data pgvector/pgvector:pg16
    docker exec itqan-pg psql -U postgres -c "CREATE DATABASE itqan_test;"
    ITQAN_TEST_DATABASE_URL=postgresql://postgres:itqan_dev@localhost:5432/itqan_test

Deliberately a *separate database* from ``ITQAN_DATABASE_URL``. These tests
truncate tables between cases; pointing them at the working database would
destroy real ingested postings, and the failure would look like a scraping bug.

sqlite was rejected outright as an alternative: no ``text[]``, ``unnest``,
``jsonb_agg``, ``FILTER`` or ``vector`` means the aggregation query — the most
intricate and most fabrication-prone code in Agent B — could not run at all, so
the suite would be testing a reimplementation rather than the SQL that ships.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

TEST_DSN = os.getenv("ITQAN_TEST_DATABASE_URL", "")


def pytest_collection_modifyitems(items):
    """Skip everything in this directory when no test database is configured.

    A collection hook rather than a shared ``skipif`` marker, because ``tests/``
    is deliberately not a package — the existing suite reaches its modules via
    the project root on ``sys.path``, so a test module cannot import from
    conftest. This way the test modules need no import at all.
    """
    if TEST_DSN:
        return
    skip = pytest.mark.skip(
        reason="ITQAN_TEST_DATABASE_URL is not set (see tests/agent_b/db/conftest.py)"
    )
    here = Path(__file__).parent
    for item in items:
        if here in Path(item.fspath).parents:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def migrated_dsn() -> str:
    """Apply migrations once per session against the test database."""
    if not TEST_DSN:
        pytest.skip("ITQAN_TEST_DATABASE_URL is not set")

    from agents.agent_b_job_ingest.db import apply_migrations

    apply_migrations(TEST_DSN)
    return TEST_DSN


@pytest.fixture
def store(migrated_dsn: str):
    """A JobStore on a clean schema.

    Truncates rather than dropping: it is far faster, and it keeps the
    migrations under test applied exactly once so a migration bug cannot be
    masked by a re-run between cases.
    """
    from agents.agent_b_job_ingest.db import JobStore

    with JobStore(migrated_dsn) as s:
        conn = s.connect()
        with conn.cursor() as cur:
            cur.execute(
                "TRUNCATE job_postings, skill_demand_stats, source_health, "
                "skill_esco_map, esco_labels, esco_skills RESTART IDENTITY CASCADE"
            )
        conn.commit()
        yield s
