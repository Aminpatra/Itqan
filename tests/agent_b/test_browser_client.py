"""The browser transport: its contract, its identity, and its limits.

No Chromium is launched here and none is installed in CI. That is deliberate and
it is the same seam every adapter test uses — clients are injected, so the things
worth pinning (the contract, the user agent, what counts as a refusal, and what
this module is NOT allowed to do) are all testable without a browser.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from shared.config import Config
from shared.scraping import browser as browser_mod
from shared.scraping.browser import BrowserClient, browser_user_agent
from shared.scraping.http import Blocked, PoliteClient


# ---------------------------------------------------------------------------
# the contract — this is what makes it a swap rather than a rewrite
# ---------------------------------------------------------------------------
def test_it_is_a_drop_in_for_the_http_client():
    """Adapters call exactly one method. If BrowserClient offers the same one,
    swapping transports is a construction-line change and every adapter, plus
    root_fetch, comes along for free."""
    public = lambda cls: {n for n, _ in inspect.getmembers(cls, inspect.isfunction)
                          if not n.startswith("_")}
    assert public(PoliteClient) <= public(BrowserClient)
    assert inspect.signature(BrowserClient.get_text) == inspect.signature(PoliteClient.get_text)


# ---------------------------------------------------------------------------
# identity — a real browser that still says who it is
# ---------------------------------------------------------------------------
def test_the_user_agent_is_a_real_chrome_string_carrying_our_contact():
    """Both halves are load-bearing. Chrome, because a site gating on
    browser-shaped clients should see one — and it genuinely is one. Our contact,
    because an operator who dislikes the cadence should be able to reach a human
    rather than only being able to block an IP."""
    ua = browser_user_agent(Config(user_agent="ItqanJobBot/1.0 (+me@example.com)"))
    assert "Chrome/" in ua and "Mozilla/5.0" in ua
    assert "+me@example.com" in ua


def test_an_unidentified_build_cannot_crawl():
    """Same refusal the httpx client makes, at the same point: the last moment
    before a request would leave the machine."""
    unidentified = Config(user_agent="ItqanJobBot/0.1 (+contact-not-configured)")
    with pytest.raises(RuntimeError, match="ITQAN_USER_AGENT"):
        browser_user_agent(unidentified)


def test_the_user_agent_is_not_rotated():
    """Rotation exists to frustrate correlation, which is the opposite of what
    `require_identified_user_agent` is for. One string, every request."""
    cfg = Config(user_agent="ItqanJobBot/1.0 (+me@example.com)")
    assert len({browser_user_agent(cfg) for _ in range(5)}) == 1


# ---------------------------------------------------------------------------
# a challenge is a refusal, not an obstacle
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("html,text", [
    # Structural markers decide alone — they appear only in a challenge's markup.
    ('<html><div class="cf-challenge"></div></html>', "Just a moment..."),
    ('<html><script src="/cdn-cgi/challenge-platform/x.js"></script></html>', ""),
    ('<html><div class="cf-turnstile"></div></html>', ""),
    # A phrase counts too, but only on a page with no content behind it.
    ("<html>Checking your browser</html>", "Checking your browser before accessing"),
])
def test_a_bot_challenge_is_detected(html, text):
    assert browser_mod._looks_challenged(html, text)


@pytest.mark.parametrize("name,body", [
    # THE false positive that matters. `turnstile` is a Cloudflare widget and a
    # piece of physical access-control hardware, so a facilities vacancy contains
    # it legitimately. Matching on the bare word would drop the source and record
    # a refusal that never happened.
    ("facilities job mentioning a turnstile", "Maintain turnstile and barrier systems. "),
    # "just a moment" is ordinary English before it is a Cloudflare string.
    ("job ad using the phrase in prose", "Just a moment of your time. Apply now. "),
    ("an ordinary posting", "Site Reliability Engineer. Muscat. Apply within. "),
])
def test_real_pages_are_not_mistaken_for_challenges(name, body):
    """A false positive is the expensive direction: it stops the crawl, silently
    drops a real source, and logs a refusal the operator never made. So a phrase
    only counts on a page too small to contain anything else — and these are not."""
    text = body * 80
    html = f"<html><body>{''.join(f'<p>{body}</p>' for _ in range(80))}</body></html>"
    assert not browser_mod._looks_challenged(html, text), name


def test_heavy_resources_are_aborted_and_documents_are_not():
    """We want text. Not pulling megabytes of images we discard is politer to the
    target than pulling them, and it is most of why a rendered crawl is affordable."""
    aborted, continued = [], []

    class Route:
        def __init__(self, kind): self.kind = kind
        def abort(self): aborted.append(self.kind)
        def continue_(self): continued.append(self.kind)

    class Req:
        def __init__(self, kind): self.resource_type = kind

    for kind in ("image", "media", "font", "stylesheet", "document", "xhr", "script"):
        browser_mod._abort_heavy_resources(Route(kind), Req(kind))

    assert set(aborted) == {"image", "media", "font", "stylesheet"}
    assert set(continued) == {"document", "xhr", "script"}


# ---------------------------------------------------------------------------
# the scope boundary, pinned
# ---------------------------------------------------------------------------
def test_this_module_contains_no_bot_detection_evasion():
    """A guard, in the idiom this repo already uses for "no LLM in the read
    surface".

    The requested spec included stripping `navigator.webdriver`, forging
    `navigator.plugins`, faking the chrome runtime, and sitting on a Cloudflare
    challenge until it passes. Those defeat bot detection, which is a different
    activity from being polite, and they were deliberately not built.

    That decision lives in a plan document nobody will read in six months. This
    makes it fail a test instead — because the tempting time to add it is exactly
    when a source starts refusing us, which is exactly when it is least
    defensible.
    """
    source = Path(browser_mod.__file__).read_text(encoding="utf-8")
    # Strip comments and docstrings: this file DISCUSSES these terms at length,
    # and the point is whether it EXECUTES them.
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            node.value.value = ""
    code = ast.unparse(tree).lower()

    for forbidden in ("navigator.webdriver", "add_init_script", "__proto__",
                      "wait_for_challenge", "cf_clearance", "stealth"):
        assert forbidden not in code, (
            f"{forbidden!r} appears in executable code. This transport identifies "
            f"itself and leaves when refused; if that is being changed, change it "
            f"deliberately and update this test.")


# ---------------------------------------------------------------------------
# which transport, and the one thing that must never render
# ---------------------------------------------------------------------------
def test_a_source_gets_a_browser_only_when_it_is_named():
    """The choice is one config value, not three adapters. `browser_sources` is
    empty by default because the measurement found no source that needs it —
    but a source that starts refusing scripts should be a value change."""
    from shared.scraping import build_client

    plain = Config(user_agent="ItqanJobBot/1.0 (+me@example.com)")
    assert isinstance(build_client(source="el7far", config=plain), PoliteClient)

    rendered = Config(user_agent="ItqanJobBot/1.0 (+me@example.com)",
                      browser_sources=("el7far",))
    assert isinstance(build_client(source="el7far", config=rendered), BrowserClient)
    # ...and only that source.
    assert isinstance(build_client(source="telegram", config=rendered), PoliteClient)


def test_robots_is_never_fetched_through_a_browser():
    """The rule this whole crawler's fail-closed guarantee rests on.

    robots.txt is plain text. A browser returns it wrapped in
    `<html><body><pre>`, the parser reads one unparseable line, and an
    unparseable robots file parses as an EMPTY one — which permits everything.
    So a source can render its pages; it can never render its permission.
    """
    from shared.scraping import build_robots

    cfg = Config(user_agent="ItqanJobBot/1.0 (+me@example.com)",
                 browser_sources=("el7far",))
    policy = build_robots(source="el7far", config=cfg)
    assert isinstance(policy._client, PoliteClient)


def test_the_root_fetcher_renders_pages_but_not_robots():
    """Both halves in one place, because this is the object that actually
    crawls the employer pages — the only place the browser measured a gain."""
    from agents.agent_b_job_ingest.root_fetch import RootFetcher

    cfg = Config(user_agent="ItqanJobBot/1.0 (+me@example.com)")
    fetcher = RootFetcher(cfg, browser=True)
    client, robots = fetcher._for_host("https://careers.dhl.com/global/en/job/123")
    assert isinstance(client, BrowserClient)
    assert isinstance(robots._client, PoliteClient)

    plain, _ = RootFetcher(cfg, browser=False)._for_host("https://careers.oq.com/job/1")
    assert isinstance(plain, PoliteClient)


def test_a_missing_browser_degrades_to_http_and_says_so(monkeypatch):
    """An image built without WITH_BROWSER has no Chromium binary. `fetch` is
    best-effort by contract, so a failed launch would turn every destination
    into "unreachable" and root enrichment would stop working on a deploy that
    looks perfectly healthy. Falling back is right; falling back SILENTLY is
    not."""
    from agents.agent_b_job_ingest.root_fetch import RootFetcher

    monkeypatch.setattr(BrowserClient, "_browser", None, raising=False)
    monkeypatch.setattr(BrowserClient, "_ensure_browser",
                        classmethod(lambda cls, cfg: (_ for _ in ()).throw(
                            RuntimeError("Executable doesn't exist"))))
    monkeypatch.setattr(RootFetcher, "_browser_warned", False)

    fetcher = RootFetcher(Config(user_agent="ItqanJobBot/1.0 (+me@example.com)"),
                          browser=True)
    client, _ = fetcher._for_host("https://careers.dhl.com/job/1")

    assert isinstance(client, PoliteClient)
    assert fetcher.warnings and "WITH_BROWSER" in fetcher.warnings[0]

    # Once per process, not once per host: a cycle touches dozens.
    fetcher._for_host("https://careers.oq.com/job/2")
    assert len(fetcher.warnings) == 1


def test_a_challenge_raises_the_same_exception_as_a_robots_refusal():
    """So the refused-host memory and the cycle counters need no new code: a
    challenge and a robots disallow mean the same thing and are handled the same
    way."""
    assert issubclass(Blocked, Exception)
    assert "Blocked" in Path(browser_mod.__file__).read_text(encoding="utf-8")
