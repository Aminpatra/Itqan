"""The only way an ingestion agent talks to the internet.

Everything outward-facing is centralised here so the cadence, the identifying
user agent, and the response-size cap are properties of the system rather than
things each adapter is trusted to remember. An adapter that constructed its own
``httpx.Client`` would bypass all three at once, so adapters are given a client
and never build one.

The size cap is not a performance tuning knob. A feed parsed with stdlib
ElementTree, or an API page loaded whole before parsing, turns a malformed or
hostile document into an out-of-memory kill of an unattended scheduled job.
Streaming and aborting past ``max_bytes`` means the parser never sees a document
larger than we agreed to accept.

Agent-neutral: this lives in ``shared/`` because both Agent B (jobs) and Agent D
(courses) fetch the web, and politeness is not a per-agent concern.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx

from shared.config import Config


class ResponseTooLarge(Exception):
    """The body exceeded max_bytes and was abandoned mid-stream."""


class Blocked(Exception):
    """The source refused us: 403, 429, or robots.txt.

    Distinct from a transport error on purpose. A timeout is worth one retry; a
    block is a decision by the operator, and retrying it harder is precisely the
    behaviour the project constraints forbid.
    """

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class SourcePolicy:
    """How politely to treat one source.

    Defaults are conservative. ``min_interval_s`` is a floor between requests to
    the SAME source, not a global one: two sources on different hosts have no
    reason to queue behind each other, and a shared limiter would make adding a
    source slow down every existing one.
    """

    min_interval_s: float = 2.0
    timeout_s: float = 20.0
    max_bytes: int = 5_000_000
    max_retries: int = 2
    # Only these get a retry. 403 and 404 are answers, not failures.
    retry_statuses: frozenset[int] = field(default_factory=lambda: frozenset({500, 502, 503, 504}))
    # 429 is separated out: it is a block, and the response is to stop, not to
    # come back sooner. Retrying a rate limit is how a polite scraper becomes an
    # impolite one without anybody deciding to.
    backoff_s: float = 5.0

    @classmethod
    def for_telegram(cls) -> "SourcePolicy":
        """Slower on purpose. t.me serves no robots.txt, so there is no stated
        crawl policy to honour — which is a reason for more restraint, not less.
        """
        return cls(min_interval_s=5.0)


class _RateLimiter:
    """One per source. Blocks until min_interval_s has elapsed since the last
    request. Thread-safe because the cycle fans sources out concurrently."""

    def __init__(self, min_interval_s: float) -> None:
        self.min_interval_s = min_interval_s
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._last
            remaining = self.min_interval_s - elapsed
            if remaining > 0:
                time.sleep(remaining)
            self._last = time.monotonic()


class PoliteClient:
    """An HTTP client scoped to one source.

    One client per source, one limiter inside it. This is why duplicate source
    handles must be fatal at config load: two entries for one channel produce two
    clients with two independent limiters pointed at one host, which halves the
    effective interval while every log line still reports the polite one.
    """

    def __init__(
        self,
        *,
        source: str,
        policy: SourcePolicy | None = None,
        config: Config | None = None,
        client: httpx.Client | None = None,
        user_agent: str | None = None,
    ) -> None:
        config = config or Config()
        self.source = source
        self.policy = policy or SourcePolicy(max_bytes=config.max_response_bytes)
        self._limiter = _RateLimiter(self.policy.min_interval_s)
        self.bytes_fetched = 0
        self.requests_made = 0

        # require_identified_user_agent() rather than the raw field: this is the
        # last point before bytes leave the machine, so it is the honest place to
        # refuse. Injected clients (tests, dry runs) skip the check by passing
        # user_agent explicitly.
        self._owns_client = client is None
        ua = user_agent if user_agent is not None else config.require_identified_user_agent()
        self._client = client or httpx.Client(
            timeout=self.policy.timeout_s,
            follow_redirects=True,
            headers={
                "User-Agent": ua,
                "Accept-Encoding": "gzip, deflate",
            },
        )

    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "PoliteClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    def get_text(self, url: str, *, params: dict | None = None) -> str:
        """Fetch a URL as text, rate-limited, size-capped and retried.

        Raises ``Blocked`` on 403/429, ``ResponseTooLarge`` past the cap, and
        ``httpx`` errors after retries are exhausted. Callers convert these into
        an errored ``AdapterResult``; nothing swallows them here, because a
        silent empty string would look identical to a source with no items.
        """
        last_exc: Exception | None = None

        for attempt in range(self.policy.max_retries + 1):
            self._limiter.wait()
            try:
                return self._get_once(url, params=params)
            except Blocked:
                raise
            except ResponseTooLarge:
                raise
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status not in self.policy.retry_statuses:
                    raise
                last_exc = exc
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc

            if attempt < self.policy.max_retries:
                time.sleep(self.policy.backoff_s * (attempt + 1))

        assert last_exc is not None
        raise last_exc

    def _get_once(self, url: str, *, params: dict | None = None) -> str:
        self.requests_made += 1
        with self._client.stream("GET", url, params=params) as response:
            if response.status_code in (403, 429):
                raise Blocked(
                    f"{self.source}: {url} returned {response.status_code}",
                    status=response.status_code,
                )
            response.raise_for_status()

            # Check the declared length first — cheap, and lets us refuse before
            # transferring anything. Not trusted on its own: it is absent on
            # chunked responses and a server may simply lie, so the streaming
            # accumulator below is the real enforcement.
            declared = response.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > self.policy.max_bytes:
                raise ResponseTooLarge(
                    f"{self.source}: {url} declared {declared} bytes, cap is {self.policy.max_bytes}"
                )

            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_bytes():
                total += len(chunk)
                if total > self.policy.max_bytes:
                    raise ResponseTooLarge(
                        f"{self.source}: {url} exceeded {self.policy.max_bytes} bytes mid-stream"
                    )
                chunks.append(chunk)

            self.bytes_fetched += total
            body = b"".join(chunks)

        encoding = response.encoding or "utf-8"
        # errors="replace": one bad byte in a large feed must not lose the rest.
        # Mojibake is visible downstream; an exception here is a silently empty
        # cycle.
        return body.decode(encoding, errors="replace")


def host_of(url: str) -> str:
    return urlsplit(url).netloc.lower()
