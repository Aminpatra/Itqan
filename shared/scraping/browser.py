"""A real browser as a transport, for sources httpx cannot read.

`PoliteClient` fetches HTML and stops, so a site that renders its listings in
JavaScript is invisible to it. This project has already paid for that: Agent D
reads a GitHub JSON file for freeCodeCamp because `/learn/` is a Gatsby SPA with
zero usable links in 499 KB of served HTML.

**This is a transport swap, not a new subsystem.** It exposes exactly the method
adapters already call — ``get_text(url) -> str`` — so el7far, telegram, dubizzle
and `root_fetch` change one construction line and nothing else. Politeness is not
reimplemented either: `SourcePolicy`, `_RateLimiter` and `Blocked` are imported
from `http.py`, so the cadence and the refusal semantics are literally the same
objects the httpx path uses.

What a browser adds beyond fetching:

* JavaScript runs, so SPAs, client-side redirects and lazily-rendered listings
  resolve.
* Images, media, fonts and stylesheets are **aborted**. We want text, and not
  pulling megabytes of assets we discard is politer than pulling them.
* `domcontentloaded`, not `networkidle` — analytics beacons and websockets keep
  modern pages "busy" indefinitely, so waiting for idle is waiting for a timeout.
* A fresh context per request, so no cookie or storage state leaks between sites.

**What this deliberately does NOT do.** It does not strip `navigator.webdriver`,
forge `navigator.plugins`, fake the chrome runtime, or sit on a bot-challenge page
waiting for it to pass. Those defeat bot detection, which is a different activity
from being polite, and this crawler's whole posture is the opposite: it says who
it is and leaves when told to. A challenge page raises ``Blocked`` — the same
signal a robots refusal raises — so the refused-host memory and the cycle
counters already handle it with no new code.

The user agent is a real Chrome string with the project's contact appended. Sites
gating on "is this a real browser" are satisfied because it *is* one, and an
operator who dislikes our cadence can still reach a human instead of only being
able to block an IP.
"""

from __future__ import annotations

import random
import threading
import time
from typing import Any, Optional

from shared.config import Config
from shared.scraping.http import Blocked, SourcePolicy, _RateLimiter

# Resource types we abort. Text is the entire product of a crawl; everything here
# is bytes the target pays to serve and we would immediately discard.
_BLOCKED_RESOURCES = frozenset({"image", "media", "font", "stylesheet"})

# Markers that a bot check stands between us and the content.
#
# Split into two kinds, because matching on vocabulary alone is how a detector
# starts dropping real sources. `turnstile` is the clearest case: it is a
# Cloudflare widget AND a piece of physical access-control hardware, so a
# facilities-management vacancy legitimately contains the word. Same trap with
# "just a moment", which is ordinary English prose.
#
# STRUCTURAL markers are unambiguous — they only appear in a challenge's own
# markup, never in prose.
_CHALLENGE_STRUCTURAL = (
    "cf-challenge",
    "cf_chl_",
    "cf-turnstile",
    "challenges.cloudflare.com",
    "/cdn-cgi/challenge-platform",
    "__cf_chl",
)
# PHRASE markers are suggestive but not proof, so they only count on a page with
# almost no content. A challenge interstitial is a few hundred characters; a job
# posting is thousands. Requiring both is what stops a real vacancy that happens
# to say "just a moment" from taking its whole source offline.
_CHALLENGE_PHRASES = (
    "checking your browser",
    "just a moment",
    "enable javascript and cookies to continue",
    "verifying you are human",
    "please wait while we verify",
)
# Below this many characters of body text, a page has no content worth having —
# so a phrase marker is very likely to be the interstitial rather than prose.
_CHALLENGE_MAX_TEXT = 1200

# One real Chrome UA, with our identity appended rather than hidden. NOT rotated:
# rotation exists to frustrate correlation, which is the opposite of what
# `require_identified_user_agent` is for.
_CHROME = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
           "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36")


def browser_user_agent(config: Optional[Config] = None) -> str:
    """A genuine Chrome UA carrying our contact string.

    Both halves matter. The Chrome half means a site that gates on browser-shaped
    clients serves us the same page it serves a person — which is honest, because
    a browser is exactly what is asking. The appended half means the operator can
    identify and contact us, which is the whole reason
    `require_identified_user_agent` refuses to start without one.
    """
    return f"{_CHROME} {(config or Config()).require_identified_user_agent()}"


class BrowserClient:
    """A polite, robots-respecting browser scoped to one source.

    Mirrors `PoliteClient`: same `get_text` signature, same `Blocked` semantics,
    same per-source limiter. Chromium starts on the first request and is reused
    for the life of the client; each request still gets its own context.
    """

    # Chromium is heavy — a browser process plus 300-500 MB per open context — and
    # it shares a 4 GB box with Postgres and an OCR pass that peaks at 1.2 GB. Two
    # is the ceiling the plan committed to, enforced here rather than trusted to
    # each caller.
    _slots = threading.Semaphore(2)
    _playwright: Any = None
    _browser: Any = None
    _launch_lock = threading.Lock()

    def __init__(
        self,
        *,
        source: str,
        policy: SourcePolicy | None = None,
        config: Config | None = None,
        user_agent: str | None = None,
    ) -> None:
        config = config or Config()
        self.config = config
        self.source = source
        self.policy = policy or SourcePolicy(max_bytes=config.max_response_bytes)
        # Floor of 1.5s even if a policy asks for less: this is a browser, so each
        # request costs the target a full page render rather than one document.
        interval = max(self.policy.min_interval_s, config.browser_min_interval_s)
        self._limiter = _RateLimiter(interval)
        self.bytes_fetched = 0
        self.requests_made = 0
        self.challenged = 0
        # Same reasoning as PoliteClient: this is the last point before a request
        # leaves the machine, so it is the honest place to refuse an unidentified
        # one. An explicit user_agent is the test/dry-run seam.
        self._user_agent = user_agent if user_agent is not None else browser_user_agent(config)

    # ------------------------------------------------------------------
    @classmethod
    def _ensure_browser(cls, config: Config) -> Any:
        """Launch Chromium once per process, on first use.

        Lazy because most runs never crawl — the API imports this module chain and
        should not pay a browser launch to do it.
        """
        with cls._launch_lock:
            if cls._browser is None:
                from playwright.sync_api import sync_playwright

                cls._playwright = sync_playwright().start()
                cls._browser = cls._playwright.chromium.launch(
                    headless=True,
                    args=["--disable-dev-shm-usage"],   # /dev/shm is small in containers
                )
            return cls._browser

    @classmethod
    def shutdown(cls) -> None:
        """Close the shared browser. Called at the end of a cycle."""
        with cls._launch_lock:
            if cls._browser is not None:
                cls._browser.close()
                cls._browser = None
            if cls._playwright is not None:
                cls._playwright.stop()
                cls._playwright = None

    def close(self) -> None:
        """Per-client no-op: the browser is shared, so one source finishing does
        not entitle it to close the process others are using. `shutdown()` ends it."""

    def __enter__(self) -> "BrowserClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    def get_text(self, url: str, *, params: dict | None = None) -> str:
        """Render a URL and return its text. Same contract as `PoliteClient`.

        Raises `Blocked` on 403/429 and on a bot-challenge page; propagates the
        last error once retries are exhausted. Nothing is swallowed — an empty
        string would be indistinguishable from a source with no items.
        """
        if params:
            from urllib.parse import urlencode
            url = f"{url}{'&' if '?' in url else '?'}{urlencode(params)}"

        last_exc: Exception | None = None
        for attempt in range(self.policy.max_retries + 1):
            self._limiter.wait()
            # Jitter so a cycle's requests are not mechanically spaced — the
            # pattern is what makes a polite crawler look like a script.
            time.sleep(random.uniform(0, self.config.browser_jitter_s))
            try:
                return self._render_once(url)
            except Blocked:
                raise                      # a decision, not a failure. Never retried.
            except Exception as exc:       # noqa: BLE001 - playwright's errors are its own
                last_exc = exc

            if attempt < self.policy.max_retries:
                time.sleep(self.policy.backoff_s * (attempt + 1))

        assert last_exc is not None
        raise last_exc

    def _render_once(self, url: str) -> str:
        browser = self._ensure_browser(self.config)
        self.requests_made += 1

        with self._slots:
            context = browser.new_context(
                user_agent=self._user_agent,
                viewport={"width": 1366, "height": 768},
                locale="en-US",
                timezone_id="Asia/Muscat",
                extra_http_headers={"Accept-Language": "en-US,en;q=0.9,ar;q=0.8"},
            )
            try:
                page = context.new_page()
                page.route("**/*", _abort_heavy_resources)
                response = page.goto(
                    url,
                    wait_until="domcontentloaded",
                    # Hard ceiling so one slow host cannot stall an unattended
                    # cycle. Playwright wants milliseconds.
                    timeout=self.config.browser_timeout_s * 1000,
                )
                if response is not None and response.status in (403, 429):
                    raise Blocked(f"{self.source}: {url} returned {response.status}",
                                  status=response.status)

                html = page.content()
                text = page.inner_text("body") if page.query_selector("body") else ""

                if _looks_challenged(html, text):
                    self.challenged += 1
                    raise Blocked(
                        f"{self.source}: {url} served a bot challenge — treating as a "
                        f"refusal, not waiting it out")

                self.bytes_fetched += len(html.encode("utf-8", "ignore"))
                if self.bytes_fetched > self.policy.max_bytes * 50:
                    # A per-cycle ceiling rather than per-response: a rendered page
                    # legitimately exceeds a single document's cap, but a crawl that
                    # has pulled fifty caps' worth is running away.
                    raise Blocked(f"{self.source}: byte budget exhausted for this cycle")
                return html
            finally:
                context.close()


def _abort_heavy_resources(route: Any, request: Any) -> None:
    if request.resource_type in _BLOCKED_RESOURCES:
        route.abort()
    else:
        route.continue_()


def _looks_challenged(html: str, text: str) -> bool:
    """Is a bot check standing between us and the content?

    Used to STOP, not to trigger a wait-and-retry: a site that puts a challenge in
    front of its content has said no as clearly as a robots disallow does, and the
    honest response is the same one.

    Because it stops, a false positive is expensive — it silently drops a real
    source and records a refusal that never happened. So structural markers decide
    on their own, while an English phrase only counts on a page too small to hold
    anything else. A facilities vacancy mentioning a turnstile, or copy that says
    "just a moment", stays a page.
    """
    blob = f"{html}\n{text}".lower()
    if any(marker in blob for marker in _CHALLENGE_STRUCTURAL):
        return True
    if len(text.strip()) > _CHALLENGE_MAX_TEXT:
        return False                    # it has real content; it is not an interstitial
    return any(phrase in blob for phrase in _CHALLENGE_PHRASES)
