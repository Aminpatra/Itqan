"""robots.txt, evaluated fail-closed.

The rule that shapes this module: **when we cannot establish that a path is
allowed, we treat it as disallowed.** The tempting default is the opposite —
an unreachable robots.txt looks like an absence of restriction — but that
inverts the risk. A false "allowed" scrapes a site that asked us not to; a false
"disallowed" fetches nothing and shows up immediately as an empty result.

The single exception is a clean **404**, which is the documented meaning of "no
restrictions" in the standard and the only case where absence is an answer
rather than a failure. A 403 on robots.txt is emphatically *not* that: it is a
site actively refusing to talk to us, and reading it as consent would be
indefensible.

Parsing is stdlib ``urllib.robotparser``. Fetching is *not* — ``RobotFileParser
.read()`` opens its own urllib connection, which would bypass the identifying
user agent, the rate limiter and the size cap. The body comes through
``PoliteClient`` and is handed to ``parse()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import httpx

from .http import Blocked, PoliteClient, ResponseTooLarge

# robots.txt is a small text file. Anything approaching a megabyte is either
# broken or hostile, and parsing it would be pointless either way.
MAX_ROBOTS_BYTES = 512_000


@dataclass(frozen=True)
class RobotsDecision:
    allowed: bool
    reason: str

    def __bool__(self) -> bool:
        return self.allowed


class RobotsPolicy:
    """Per-host robots.txt, fetched once and cached for the cycle.

    Cached because a cycle asks the same question for every page of a feed, and
    re-fetching robots.txt for each would itself be the impolite behaviour the
    file exists to prevent.
    """

    def __init__(self, client: PoliteClient, *, user_agent: str) -> None:
        self._client = client
        self._user_agent = user_agent
        self._cache: dict[str, tuple[RobotFileParser | None, str]] = {}

    # ------------------------------------------------------------------
    def can_fetch(self, url: str) -> RobotsDecision:
        parser, note = self._for_host(url)

        if parser is None:
            # note carries why. "no robots.txt (404)" is the allow case; every
            # other note is a failure to establish permission.
            if note.startswith("no robots.txt"):
                return RobotsDecision(True, note)
            return RobotsDecision(False, f"fail-closed: {note}")

        path = _path_of(url)
        allowed = parser.can_fetch(self._user_agent, path)
        verdict = "allowed by robots.txt" if allowed else "disallowed by robots.txt"
        return RobotsDecision(allowed, f"{verdict}: {path}")

    def require(self, url: str) -> None:
        """can_fetch, but raises. For call sites where continuing past a
        disallow would be a bug rather than a branch."""
        decision = self.can_fetch(url)
        if not decision.allowed:
            raise Blocked(f"robots.txt refuses {url} ({decision.reason})")

    # ------------------------------------------------------------------
    def _for_host(self, url: str) -> tuple[RobotFileParser | None, str]:
        parts = urlsplit(url)
        host = parts.netloc.lower()
        if host in self._cache:
            return self._cache[host]

        robots_url = urlunsplit((parts.scheme or "https", parts.netloc, "/robots.txt", "", ""))
        entry = self._fetch(robots_url)
        self._cache[host] = entry
        return entry

    def _fetch(self, robots_url: str) -> tuple[RobotFileParser | None, str]:
        try:
            body = self._client.get_text(robots_url)
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 404:
                return None, "no robots.txt (404)"
            return None, f"robots.txt returned {status}"
        except Blocked as exc:
            # 403 on robots.txt: the site is refusing us outright.
            return None, str(exc)
        except ResponseTooLarge:
            return None, "robots.txt exceeded the size cap"
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            return None, f"robots.txt unreachable ({type(exc).__name__})"

        if len(body.encode("utf-8", errors="ignore")) > MAX_ROBOTS_BYTES:
            return None, "robots.txt is implausibly large"

        parser = RobotFileParser()
        try:
            parser.parse(body.splitlines())
        except Exception as exc:  # noqa: BLE001 - unparseable means unestablished
            return None, f"robots.txt could not be parsed ({type(exc).__name__}: {exc})"
        return parser, "parsed"


def _path_of(url: str) -> str:
    """The path+query robotparser should match against.

    Kept as a function because the distinction it encodes is load-bearing for
    this project: on Blogger, ``/search/label/x`` is disallowed while
    ``/feeds/posts/default/-/x`` is allowed. Those are the *same content* under
    two paths, and only one of them is ours to fetch.
    """
    parts = urlsplit(url)
    path = parts.path or "/"
    return f"{path}?{parts.query}" if parts.query else path
