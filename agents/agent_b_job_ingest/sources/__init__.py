"""Source adapters: fetch and parse only.

Nothing in this package imports the database. That is structural — it is what
keeps every adapter testable offline against a saved fixture, and what makes it
impossible for a scraping bug to reach a write path.

It also holds for ``source_health``: nothing here reads or writes it, so no code
path exists that could react to a block by retrying harder. Health is recorded
in the runlog node, and the separation is asserted by a test.
"""

from .base import AdapterResult, BaseAdapter, RawPosting, SourceAdapter
from .config import (
    DEFAULT_SOURCES,
    SourceConfig,
    SourceConfigError,
    normalize_handle,
    select_sources,
    validate_source_config,
)
from .http import Blocked, PoliteClient, ResponseTooLarge, SourcePolicy
from .robots import RobotsPolicy

__all__ = [
    "AdapterResult",
    "BaseAdapter",
    "Blocked",
    "DEFAULT_SOURCES",
    "PoliteClient",
    "RawPosting",
    "ResponseTooLarge",
    "RobotsPolicy",
    "SourceAdapter",
    "SourceConfig",
    "SourceConfigError",
    "SourcePolicy",
    "normalize_handle",
    "select_sources",
    "validate_source_config",
]
