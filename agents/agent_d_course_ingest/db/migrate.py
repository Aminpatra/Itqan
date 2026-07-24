"""Agent D's migration runner.

Same numbered-SQL runner as Agent B — deliberately self-contained rather than
importing Agent B's copy, because an agent must depend only on ``shared/`` and
never on another agent. The two differences from the default are baked in here:
this points at Agent D's migrations directory, and records applied versions in
``schema_migrations_agent_d`` — a SEPARATE tracking table, because both agents'
migration sets start at 0001 and share the one ``itqan`` database, so a single
tracker would collide.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import psycopg

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
TRACKING_TABLE = "schema_migrations_agent_d"

_IDENT = re.compile(r"^[a-z_][a-z0-9_]*$")


class MigrationError(RuntimeError):
    pass


def _checked(table: str) -> str:
    if not _IDENT.match(table):
        raise MigrationError(f"invalid tracking table name: {table!r}")
    return table


def _bootstrap_sql(table: str) -> str:
    return f"""
CREATE TABLE IF NOT EXISTS {_checked(table)} (
    version     text PRIMARY KEY,
    checksum    text        NOT NULL,
    applied_at  timestamptz NOT NULL DEFAULT now()
);
"""


@dataclass(frozen=True)
class Migration:
    version: str
    name: str
    path: Path
    sql: str

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.sql.encode("utf-8")).hexdigest()[:16]


def discover(directory: Path | None = None) -> list[Migration]:
    directory = directory or MIGRATIONS_DIR
    migrations: list[Migration] = []
    seen: dict[str, Path] = {}
    for path in sorted(directory.glob("*.sql")):
        version, _, name = path.stem.partition("_")
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


def apply_migrations(dsn: str, *, directory: Path | None = None) -> list[str]:
    """Apply pending migrations; return versions applied this run ([] if none).
    Re-running against an up-to-date database is a no-op."""
    migrations = discover(directory)
    applied: list[str] = []
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(_bootstrap_sql(TRACKING_TABLE))
        conn.commit()
        with conn.cursor() as cur:
            cur.execute(f"SELECT version, checksum FROM {_checked(TRACKING_TABLE)}")
            already = {r[0]: r[1] for r in cur.fetchall()}

        for m in migrations:
            recorded = already.get(m.version)
            if recorded is not None:
                if recorded != m.checksum:
                    raise MigrationError(
                        f"Migration {m.version}_{m.name} was already applied but its "
                        f"contents changed (recorded {recorded}, now {m.checksum}). "
                        "Write a new migration instead of editing an applied one."
                    )
                continue
            try:
                with conn.transaction():
                    with conn.cursor() as cur:
                        cur.execute(m.sql)
                        cur.execute(
                            f"INSERT INTO {_checked(TRACKING_TABLE)} (version, checksum) "
                            "VALUES (%s, %s)",
                            (m.version, m.checksum),
                        )
            except psycopg.Error as exc:
                raise MigrationError(f"Migration {m.version}_{m.name} failed: {exc}") from exc
            applied.append(f"{m.version}_{m.name}")
    return applied


def pending(dsn: str, *, directory: Path | None = None) -> list[str]:
    migrations = discover(directory)
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(_bootstrap_sql(TRACKING_TABLE))
        conn.commit()
        with conn.cursor() as cur:
            cur.execute(f"SELECT version FROM {_checked(TRACKING_TABLE)}")
            already = {r[0] for r in cur.fetchall()}
    return [f"{m.version}_{m.name}" for m in migrations if m.version not in already]
