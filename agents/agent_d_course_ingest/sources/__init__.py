"""Course source adapters: fetch and parse only.

Nothing here imports the database — the same structural rule as Agent B, which
keeps every adapter testable offline against a fixture and stops a scraping bug
reaching a write. Transport and robots come from ``shared.scraping``.
"""

from .base import AdapterResult, BaseAdapter, CourseAdapter, RawCourse
from .config import (
    DEFAULT_SOURCES,
    CourseSourceConfig,
    SourceConfigError,
    select_sources,
    validate_source_config,
)
from .factory import build_adapter

__all__ = [
    "AdapterResult",
    "BaseAdapter",
    "CourseAdapter",
    "CourseSourceConfig",
    "DEFAULT_SOURCES",
    "RawCourse",
    "SourceConfigError",
    "build_adapter",
    "select_sources",
    "validate_source_config",
]
