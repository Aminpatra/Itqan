"""A PoliteClient stand-in that serves fixtures instead of making requests.

Adapter tests must never touch a live site — a test suite that scrapes on every
run is itself the impolite behaviour the project is built to avoid, and it would
fail whenever a source changes or the network is down, which says nothing about
the code under test.

This fake matches only the surface the adapters use: ``get_text``,
``bytes_fetched``, ``close``. It deliberately does NOT subclass PoliteClient, so
a method an adapter starts calling shows up here as an AttributeError rather than
silently reaching the real implementation.
"""

from __future__ import annotations

from pathlib import Path

import httpx

FIXTURES = Path(__file__).parent / "fixtures"


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


class FakeClient:
    """Serves canned responses keyed by URL substring, in order.

    ``responses`` maps a URL fragment to either a string (served for every
    request) or a list of strings (served one per request, so pagination and
    stop conditions can be exercised).
    """

    def __init__(self, responses: dict[str, str | list[str]] | None = None) -> None:
        self.responses = responses or {}
        self.requests: list[tuple[str, dict | None]] = []
        self.bytes_fetched = 0
        self.closed = False

    def get_text(self, url: str, *, params: dict | None = None) -> str:
        self.requests.append((url, params))
        for fragment, body in self.responses.items():
            if fragment in url:
                if isinstance(body, list):
                    if not body:
                        return ""
                    text = body.pop(0)
                else:
                    text = body
                if isinstance(text, Exception):
                    raise text
                self.bytes_fetched += len(text.encode("utf-8"))
                return text
        raise httpx.HTTPStatusError(
            f"no fixture registered for {url}",
            request=httpx.Request("GET", url),
            response=httpx.Response(404, request=httpx.Request("GET", url)),
        )

    def close(self) -> None:
        self.closed = True


class AllowAllRobots:
    """For tests whose subject is parsing, not permission."""

    def can_fetch(self, url: str):
        from agents.agent_b_job_ingest.sources.robots import RobotsDecision

        return RobotsDecision(True, "test stub")

    def require(self, url: str) -> None:
        return None
