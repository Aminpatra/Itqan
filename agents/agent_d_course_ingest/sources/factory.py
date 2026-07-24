"""Turn a validated CourseSourceConfig into the adapter that can fetch it."""

from __future__ import annotations

from shared.config import Config

from .base import CourseAdapter
from .config import CourseSourceConfig
from .coursera import CourseraAdapter
from .freecodecamp import FreeCodeCampAdapter


def build_adapter(
    source: CourseSourceConfig,
    *,
    config: Config | None = None,
    is_known_unchanged=None,
) -> CourseAdapter:
    config = config or Config()
    if source.name == "coursera" or source.source_type == "api":
        return CourseraAdapter(
            name=source.name, source_group=source.source_group,
            base_url=source.base_url, config=config, is_known_unchanged=is_known_unchanged,
        )
    if source.name == "freecodecamp":
        return FreeCodeCampAdapter(
            name=source.name, source_group=source.source_group,
            base_url=source.base_url, config=config, is_known_unchanged=is_known_unchanged,
        )
    raise NotImplementedError(f"{source.name}: no adapter for source_type {source.source_type!r}")
