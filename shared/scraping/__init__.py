"""Shared, agent-neutral scraping infrastructure.

Promoted out of Agent B when Agent D (course ingestion) became a second
consumer — the same "promote to shared/ when a second consumer appears" move
that ``grounding.py`` and ``embeddings.py`` went through. The politeness client
and the fail-closed robots policy are generic: neither knows anything about jobs
or courses, only about talking to the internet without being rude.

Item dataclasses and adapters stay per-agent (a ``RawPosting`` and a
``RawCourse`` have almost no fields in common), so this package deliberately
holds only the transport and robots layers.
"""

from typing import Any, Optional

from shared.config import Config

from .http import Blocked, PoliteClient, ResponseTooLarge, SourcePolicy, host_of
from .robots import RobotsDecision, RobotsPolicy


def build_client(*, source: str, policy: Optional[SourcePolicy] = None,
                 config: Optional[Config] = None) -> Any:
    """The transport for a source: httpx, or Chromium when it needs one.

    One decision point, so "which sources render" is a config value rather than
    something spread across three adapters. Both clients expose exactly
    ``get_text(url, *, params=None)`` and ``close()``, which is the entire
    contract an adapter uses — so this returns either without the caller
    knowing which.

    Chromium is imported lazily: `playwright` is a dependency of the build, but
    the ~450 MB browser binary is only installed when ``WITH_BROWSER=1``, and an
    API process that never crawls must not pay an import to find that out.
    """
    config = config or Config()
    policy = policy or SourcePolicy(max_bytes=config.max_response_bytes)
    if source in config.browser_sources and config.browser_enabled:
        from .browser import BrowserClient
        return BrowserClient(source=source, policy=policy, config=config)
    return PoliteClient(source=source, policy=policy, config=config)


def build_robots(*, source: str, policy: Optional[SourcePolicy] = None,
                 config: Optional[Config] = None) -> RobotsPolicy:
    """A robots policy, ALWAYS fetched over plain HTTP.

    robots.txt is a plain-text file. A browser hands it back wrapped in
    ``<html><body><pre>``, and the parser would then read one unparseable line —
    which parses as an empty robots file, which permits everything. So the rule
    that matters is stated once, here, rather than trusted to each adapter:
    however a source's PAGES are fetched, its permission is not.
    """
    config = config or Config()
    policy = policy or SourcePolicy(max_bytes=config.max_response_bytes)
    client = PoliteClient(source=f"{source}-robots", policy=policy, config=config)
    return RobotsPolicy(client, user_agent=config.user_agent)


__all__ = [
    "Blocked",
    "PoliteClient",
    "build_client",
    "build_robots",
    "ResponseTooLarge",
    "RobotsDecision",
    "RobotsPolicy",
    "SourcePolicy",
    "host_of",
]
