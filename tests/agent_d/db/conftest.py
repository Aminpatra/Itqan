"""Postgres-backed Agent D tests. Skipped unless ITQAN_TEST_DATABASE_URL is set.

Applies BOTH migration sets to the test database: Agent B's (for the shared ESCO
taxonomy tables esco_skills/esco_labels that Agent D maps against) and Agent D's
(courses / skill_supply_stats / course_esco_map). Tests are allowed to reach
across agents; the "share only shared/" rule governs production code.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

TEST_DSN = os.getenv("ITQAN_TEST_DATABASE_URL", "")


def pytest_collection_modifyitems(items):
    if TEST_DSN:
        return
    skip = pytest.mark.skip(reason="ITQAN_TEST_DATABASE_URL is not set")
    here = Path(__file__).parent
    for item in items:
        if here in Path(item.fspath).parents:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def migrated_dsn() -> str:
    if not TEST_DSN:
        pytest.skip("ITQAN_TEST_DATABASE_URL is not set")
    from agents.agent_b_job_ingest.db import apply_migrations as apply_b
    from agents.agent_d_course_ingest.db import apply_migrations as apply_d

    apply_b(TEST_DSN)   # esco_skills / esco_labels (+ job tables)
    apply_d(TEST_DSN)   # courses / skill_supply_stats / course_esco_map
    return TEST_DSN


@pytest.fixture
def store(migrated_dsn: str):
    from agents.agent_d_course_ingest.db import CourseStore

    with CourseStore(migrated_dsn) as s:
        conn = s.connect()
        with conn.cursor() as cur:
            cur.execute(
                "TRUNCATE courses, skill_supply_stats, course_esco_map, course_source_health, "
                "esco_skills, esco_labels, skill_demand_stats RESTART IDENTITY CASCADE"
            )
        conn.commit()
        yield s
