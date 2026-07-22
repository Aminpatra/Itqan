"""Numbered raw-SQL migrations.

No Alembic. The schema is small, hand-written SQL is what actually runs in
production, and an autogenerate step that produces a diff nobody reads is a way
to be surprised by your own schema. The cost is that migrations are forward-only
and hand-ordered, which at this size is the right trade.

Applied migrations are recorded in ``schema_migrations`` with a checksum, so a
migration edited after it was applied is caught rather than silently ignored —
the failure mode where two environments believe they are on the same schema and
are not.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import psycopg

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"

_BOOTSTRAP = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     text PRIMARY KEY,
    checksum    text        NOT NULL,
    applied_at  timestamptz NOT NULL DEFAULT now()
);
"""


@dataclass(frozen=True)
class Migration:
    version: str          # "0002"
    name: str             # "job_postings"
    path: Path
    sql: str

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.sql.encode("utf-8")).hexdigest()[:16]


class MigrationError(RuntimeError):
    pass


def discover(directory: Path | None = None) -> list[Migration]:
    """Load migrations in version order.

    Filenames are ``NNNN_name.sql``. Sorting is on the numeric prefix, not the
    whole filename, so ``0010_x`` correctly follows ``0009_x`` instead of
    landing between ``0001`` and ``0002`` under lexicographic order.
    """
    directory = directory or MIGRATIONS_DIR
    migrations: list[Migration] = []
    seen: dict[str, Path] = {}

    for path in sorted(directory.glob("*.sql")):
        stem = path.stem
        version, _, name = stem.partition("_")
        if not version.isdigit() or not name:
            raise MigrationError(f"Migration filename must be NNNN_name.sql: {path.name}")
        if version in seen:
            raise MigrationError(
                f"Duplicate migration version {version}: {seen[version].name} and {path.name}"
            )
        seen[version] = path
        migrations.append(
            Migration(version=version, name=name, path=path, sql=path.read_text(encoding="utf-8"))
        )

    migrations.sort(key=lambda m: int(m.version))
    return migrations


def applied_versions(conn: psycopg.Connection) -> dict[str, str]:
    """version -> checksum for everything already applied."""
    with conn.cursor() as cur:
        cur.execute("SELECT version, checksum FROM schema_migrations")
        return {row[0]: row[1] for row in cur.fetchall()}


def apply_migrations(dsn: str, *, directory: Path | None = None) -> list[str]:
    """Apply every pending migration. Returns the versions applied this run.

    Re-running against an up-to-date database is a no-op and returns ``[]`` —
    that property is the whole point, and it is what the phase gate checks.
    """
    migrations = discover(directory)
    newly_applied: list[str] = []

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(_BOOTSTRAP)
        conn.commit()

        already = applied_versions(conn)

        for migration in migrations:
            recorded = already.get(migration.version)
            if recorded is not None:
                if recorded != migration.checksum:
                    raise MigrationError(
                        f"Migration {migration.version}_{migration.name} was already applied "
                        f"but its contents have changed (recorded {recorded}, "
                        f"now {migration.checksum}).\n"
                        "Editing an applied migration leaves environments silently "
                        "divergent. Write a new migration instead."
                    )
                continue

            # One transaction per migration: a failure leaves the schema at the
            # last good version rather than half-applied, and the version is
            # recorded in the same transaction as its DDL so the two cannot
            # disagree. Postgres has transactional DDL, which is what makes this
            # possible at all.
            try:
                with conn.transaction():
                    with conn.cursor() as cur:
                        cur.execute(migration.sql)
                        cur.execute(
                            "INSERT INTO schema_migrations (version, checksum) VALUES (%s, %s)",
                            (migration.version, migration.checksum),
                        )
            except psycopg.Error as exc:
                raise MigrationError(
                    f"Migration {migration.version}_{migration.name} failed: {exc}"
                ) from exc

            newly_applied.append(f"{migration.version}_{migration.name}")

    return newly_applied


def pending(dsn: str, *, directory: Path | None = None) -> list[str]:
    """Versions that would be applied, without applying them."""
    migrations = discover(directory)
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(_BOOTSTRAP)
        conn.commit()
        already = applied_versions(conn)
    return [f"{m.version}_{m.name}" for m in migrations if m.version not in already]
