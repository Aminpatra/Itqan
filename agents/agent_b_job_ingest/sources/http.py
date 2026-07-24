"""Compatibility re-export.

The implementation moved to ``shared/scraping/http.py`` when Agent D became a
second consumer of the politeness client. Agent B's adapters and tests still
import ``from .http import ...``; this keeps those paths resolving to the one
implementation in ``shared/`` rather than duplicating it.
"""

from __future__ import annotations

from shared.scraping.http import (  # noqa: F401
    Blocked,
    PoliteClient,
    ResponseTooLarge,
    SourcePolicy,
    host_of,
)

__all__ = ["Blocked", "PoliteClient", "ResponseTooLarge", "SourcePolicy", "host_of"]
