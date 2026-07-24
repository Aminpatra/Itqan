"""Postgres access for Agent D. All SQL lives in ``store.py``; migrations in
``migrations/`` (tracked separately from Agent B in ``schema_migrations_agent_d``)."""

from .migrate import MigrationError, apply_migrations, pending
from .store import CourseStore, LockNotAcquired

__all__ = ["CourseStore", "LockNotAcquired", "apply_migrations", "pending", "MigrationError"]
