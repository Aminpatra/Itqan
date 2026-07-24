"""A PoliteClient stand-in serving fixtures — course adapters must never touch a
live site in tests. Mirrors Agent B's tests/agent_b/fake_source_client."""

from __future__ import annotations

from pathlib import Path

import httpx

FIXTURES = Path(__file__).parent / "fixtures"


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


class FakeClient:
    def __init__(self, responses: dict[str, str | list[str]] | None = None) -> None:
        self.responses = responses or {}
        self.requests: list[tuple[str, dict | None]] = []
        self.bytes_fetched = 0

    def get_text(self, url: str, *, params: dict | None = None) -> str:
        self.requests.append((url, params))
        for fragment, body in self.responses.items():
            if fragment in url:
                text = body.pop(0) if isinstance(body, list) else body
                if isinstance(text, Exception):
                    raise text
                self.bytes_fetched += len(text.encode("utf-8"))
                return text
        raise httpx.HTTPStatusError(
            f"no fixture for {url}",
            request=httpx.Request("GET", url),
            response=httpx.Response(404, request=httpx.Request("GET", url)),
        )

    def close(self) -> None:
        pass


class AllowAllRobots:
    def can_fetch(self, url):
        from shared.scraping.robots import RobotsDecision

        return RobotsDecision(True, "test stub")

    def require(self, url):
        return None
