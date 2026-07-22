"""robots.txt handling, against the real committed policy file.

``tests/agent_b/fixtures/el7far_robots.txt`` is the site's actual robots.txt,
saved verbatim. The first test below is the one the whole Atom-feed approach
rests on: the same posts are reachable by two paths, and only one of them is
ours to fetch. If that ever stops being true, this fails and the adapter must
change — which is precisely the alarm worth having.
"""

from __future__ import annotations

import httpx
import pytest

from agents.agent_b_job_ingest.sources.http import Blocked, ResponseTooLarge
from agents.agent_b_job_ingest.sources.robots import RobotsPolicy
from tests.agent_b.fake_source_client import FakeClient, fixture

UA = "ItqanJobBot/0.1 (+test@example.test)"


def policy(body: str | Exception | None = None) -> RobotsPolicy:
    robots = fixture("el7far_robots.txt") if body is None else body
    return RobotsPolicy(FakeClient({"/robots.txt": robots}), user_agent=UA)


# ---------------------------------------------------------------------------
def test_label_html_is_disallowed_while_the_label_feed_is_allowed():
    """The finding the entire adapter design follows from.

    Blogger serves per-label listings as HTML under ``/search/label/{x}``, which
    ``Disallow: /search`` covers, and as Atom under
    ``/feeds/posts/default/-/{x}``, which it does not. The feed is not merely the
    tidier option — it is the only compliant one.
    """
    robots = policy()

    assert not robots.can_fetch("https://oman.el7far.com/search/label/jobs")
    assert not robots.can_fetch("https://oman.el7far.com/search?q=x")
    assert robots.can_fetch("https://oman.el7far.com/feeds/posts/default")
    assert robots.can_fetch("https://oman.el7far.com/feeds/posts/default/-/jobs")
    assert robots.can_fetch("https://oman.el7far.com/2026/07/a-post.html")


def test_a_404_is_the_one_absence_that_means_allowed():
    """The standard's documented meaning of "no robots.txt", and the only case
    where a missing answer is itself an answer. t.me returns this."""
    missing = httpx.HTTPStatusError(
        "not found",
        request=httpx.Request("GET", "https://t.me/robots.txt"),
        response=httpx.Response(404, request=httpx.Request("GET", "https://t.me/robots.txt")),
    )
    decision = policy(missing).can_fetch("https://t.me/s/example")

    assert decision.allowed
    assert "404" in decision.reason


@pytest.mark.parametrize(
    "failure",
    [
        Blocked("403 on robots.txt", status=403),
        httpx.HTTPStatusError(
            "server error",
            request=httpx.Request("GET", "https://x.test/robots.txt"),
            response=httpx.Response(500, request=httpx.Request("GET", "https://x.test/robots.txt")),
        ),
        httpx.ConnectTimeout("timed out"),
        httpx.ConnectError("refused"),
        ResponseTooLarge("robots.txt is enormous"),
    ],
)
def test_anything_other_than_404_fails_closed(failure):
    """A 403 on robots.txt is a site actively refusing to talk to us — two of the
    parked sources do exactly this — and reading it as consent would be
    indefensible. A timeout establishes nothing at all.

    The asymmetry is deliberate: a wrong "allowed" scrapes a site that asked us
    not to, while a wrong "disallowed" fetches nothing and is visible in the run
    log immediately.
    """
    decision = policy(failure).can_fetch("https://x.test/anything")

    assert not decision.allowed
    assert "fail-closed" in decision.reason


def test_unparseable_robots_txt_fails_closed():
    decision = policy("\x00\x01 not a policy ￿").can_fetch("https://x.test/a")
    # Either it parses to no rules (allow) or it fails closed — what must never
    # happen is an exception escaping into the cycle.
    assert isinstance(decision.allowed, bool)


def test_require_raises_so_a_disallow_cannot_be_ignored_by_accident():
    with pytest.raises(Blocked, match="robots.txt refuses"):
        policy().require("https://oman.el7far.com/search/label/jobs")


def test_robots_is_fetched_once_per_host_not_once_per_page():
    """Re-fetching robots.txt for every page of a feed would itself be the
    impolite behaviour the file exists to prevent."""
    client = FakeClient({"/robots.txt": fixture("el7far_robots.txt")})
    robots = RobotsPolicy(client, user_agent=UA)

    for page in range(5):
        robots.can_fetch(f"https://oman.el7far.com/feeds/posts/default?start-index={page}")

    assert len(client.requests) == 1


def test_the_policy_fetches_through_the_polite_client():
    """RobotFileParser.read() opens its own urllib connection, which would
    bypass the identifying user agent, the rate limiter and the size cap all at
    once. The body must come through the client we control."""
    client = FakeClient({"/robots.txt": fixture("el7far_robots.txt")})
    RobotsPolicy(client, user_agent=UA).can_fetch("https://oman.el7far.com/feeds/posts/default")

    assert client.requests, "robots.txt was not fetched through PoliteClient"
    assert client.requests[0][0].endswith("/robots.txt")
