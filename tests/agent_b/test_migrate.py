"""Migration discovery and ordering — offline, no database needed.

The runner's *behaviour against Postgres* is covered in tests/agent_b/db. What
is tested here is the file-level logic, which is where the mistakes are cheap to
make and silent: a mis-sorted version applies DDL in the wrong order, and a
duplicate version means one migration never runs at all.
"""

from __future__ import annotations

import pytest

from agents.agent_b_job_ingest.db.migrate import MigrationError, discover


def write(directory, name: str, sql: str = "SELECT 1;"):
    (directory / name).write_text(sql, encoding="utf-8")


def test_real_migrations_are_discoverable_and_ordered():
    migrations = discover()
    versions = [m.version for m in migrations]

    assert versions == sorted(versions, key=int)
    assert versions[0] == "0001"
    # The extension must precede any table declaring a vector column.
    assert migrations[0].name == "extensions"
    assert [m.name for m in migrations] == [
        "extensions",
        "job_postings",
        "skill_demand_stats",
        "source_health",
        "indexes",
        "listing_provenance",
        "esco",
    ]


def test_ordering_is_numeric_not_lexicographic(tmp_path):
    """`0010` must follow `0009`, not sort between `0001` and `0002`.

    Lexicographic ordering is the default from `sorted(glob())` and stays
    invisible until the tenth migration, at which point DDL starts applying in
    the wrong order against a live database.
    """
    for name in ("0001_a.sql", "0002_b.sql", "0009_c.sql", "0010_d.sql", "0011_e.sql"):
        write(tmp_path, name)

    assert [m.version for m in discover(tmp_path)] == ["0001", "0002", "0009", "0010", "0011"]


def test_duplicate_version_is_rejected(tmp_path):
    """Two files claiming one version means one of them silently never applies."""
    write(tmp_path, "0001_first.sql")
    write(tmp_path, "0001_second.sql")

    with pytest.raises(MigrationError, match="Duplicate migration version"):
        discover(tmp_path)


@pytest.mark.parametrize("bad", ["nope.sql", "abc_thing.sql", "0001.sql"])
def test_malformed_filenames_are_rejected(tmp_path, bad):
    write(tmp_path, bad)
    with pytest.raises(MigrationError, match="NNNN_name.sql"):
        discover(tmp_path)


def test_checksum_changes_when_content_changes(tmp_path):
    """The checksum is what catches a migration edited after it was applied —
    the failure mode where two environments believe they share a schema and
    don't."""
    write(tmp_path, "0001_a.sql", "CREATE TABLE a ();")
    before = discover(tmp_path)[0].checksum

    write(tmp_path, "0001_a.sql", "CREATE TABLE a (x int);")
    after = discover(tmp_path)[0].checksum

    assert before != after


def test_checksum_is_stable_for_identical_content(tmp_path):
    write(tmp_path, "0001_a.sql", "CREATE TABLE a ();")
    first = discover(tmp_path)[0].checksum
    second = discover(tmp_path)[0].checksum
    assert first == second
