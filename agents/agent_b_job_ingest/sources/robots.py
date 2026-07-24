"""Compatibility re-export.

The implementation moved to ``shared/scraping/robots.py`` when Agent D became a
second consumer of the fail-closed robots policy. Agent B's adapters and tests
still import ``from .robots import ...``; this keeps those paths resolving to
the one implementation in ``shared/`` rather than duplicating it.
"""

from __future__ import annotations

from shared.scraping.robots import (  # noqa: F401
    MAX_ROBOTS_BYTES,
    RobotsDecision,
    RobotsPolicy,
)

__all__ = ["MAX_ROBOTS_BYTES", "RobotsDecision", "RobotsPolicy"]
