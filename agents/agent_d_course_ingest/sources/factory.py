"""Turn a validated CourseSourceConfig into the adapter that can fetch it."""

from __future__ import annotations

from shared.config import Config

from .base import CourseAdapter
from .config import CourseSourceConfig
from .coursera import CourseraAdapter
from .edx import EdxAdapter
from .freecodecamp import FreeCodeCampAdapter


def build_adapter(
    source: CourseSourceConfig,
    *,
    config: Config | None = None,
    is_known_unchanged=None,
) -> CourseAdapter:
    config = config or Config()
    if source.name == "coursera" or source.source_type == "api":
        # A backfill walks far past the per-cycle page cap and skips the
        # quality-signal page fetch, which is the slow part (~1MB and a polite
        # interval per course). Those signals fill in on later normal cycles
        # through the volatile-refresh path, which since P2 will not overwrite
        # them with nulls when a lookup fails.
        backfill = getattr(config, "course_backfill_pages", None)
        return CourseraAdapter(
            name=source.name, source_group=source.source_group,
            base_url=source.base_url, config=config, is_known_unchanged=is_known_unchanged,
            max_pages=backfill,
            enrich=False if backfill else None,
        )
    if source.name == "edx":
        return EdxAdapter(
            name=source.name, source_group=source.source_group,
            base_url=source.base_url, config=config, is_known_unchanged=is_known_unchanged,
        )
    if source.name == "freecodecamp":
        return FreeCodeCampAdapter(
            name=source.name, source_group=source.source_group,
            base_url=source.base_url, config=config, is_known_unchanged=is_known_unchanged,
        )
    raise NotImplementedError(f"{source.name}: no adapter for source_type {source.source_type!r}")
