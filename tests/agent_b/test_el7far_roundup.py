"""Roundup/index posts expand into one posting per linked job page.

A single Blogger post that lists several vacancies (each linking to its own
same-site detail page) must become one row per vacancy — with each job's OWN
title and body — not one merged posting. Uses a THREE-job roundup deliberately:
the count is whatever the links yield, never assumed to be any particular number.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from agents.agent_b_job_ingest.sources.el7far import El7farAdapter
from agents.agent_b_job_ingest.sources.http import Blocked
from shared.config import Config
from tests.agent_b.fake_source_client import AllowAllRobots, FakeClient

HOST = "https://example.test"
RECENT = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()

# Three distinct vacancies, each linking to its own same-site page.
JOB_SLUGS = ["marketing-manager", "hr-officer", "finance-analyst"]


def _roundup_feed() -> str:
    links = "".join(
        f'&lt;p&gt;{slug}: &lt;a href="/2026/07/{slug}.html"&gt;Apply&lt;/a&gt;&lt;/p&gt;'
        for slug in JOB_SLUGS
    )
    content = f"&lt;div&gt;&lt;p&gt;3 Jobs in Marketing, HR and Finance.&lt;/p&gt;{links}&lt;/div&gt;"
    return f"""<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title type="text">Example Jobs Blog</title>
  <entry>
    <id>tag:blogger.com,1999:blog-1.post-9</id>
    <published>{RECENT}</published>
    <category scheme="http://www.blogger.com/atom/ns#" term="Roundup"/>
    <title type="text">Digital Mall Oman Careers 2026: 3 Jobs in Marketing, HR and Finance</title>
    <content type="html">{content}</content>
    <link rel="alternate" type="text/html" href="{HOST}/2026/07/careers-roundup.html"/>
    <author><name>Mahmoud</name></author>
  </entry>
</feed>"""


def _job_page(title: str, body: str) -> str:
    return (
        f"<html><head><title>{title} — Example Jobs</title></head><body>"
        f"<nav>site menu, unrelated links</nav>"
        f'<div class="post-body"><h1 class="post-title">{title}</h1><p>{body}</p>'
        f'<p>Apply: <a href="https://careers.example.test/{title}">form</a></p></div>'
        f"<footer>site footer</footer></body></html>"
    )


PAGES = {
    "/2026/07/marketing-manager.html": _job_page(
        "Marketing Manager", "Requirements: SEO, Google Analytics and content strategy."
    ),
    "/2026/07/hr-officer.html": _job_page(
        "HR Officer", "Requirements: recruitment, payroll administration and onboarding."
    ),
    "/2026/07/finance-analyst.html": _job_page(
        "Finance Analyst", "Requirements: financial modelling, forecasting and Excel."
    ),
}


def _adapter(*, config=None, robots=None, feed=None, pages=None):
    responses = {"/feeds/posts/default": feed or _roundup_feed()}
    responses.update(pages or PAGES)
    client = FakeClient(responses)
    return El7farAdapter(
        client=client, config=config or Config(), robots=robots or AllowAllRobots()
    ), client


# ---------------------------------------------------------------------------
def test_roundup_expands_into_one_posting_per_linked_job():
    adapter, _ = _adapter()
    result = adapter.fetch()
    by_title = {p.title: p for p in result.postings}

    assert set(by_title) == {"Marketing Manager", "HR Officer", "Finance Analyst"}, (
        "expected one posting per linked job, not a single merged roundup row"
    )
    # each posting carries ITS OWN page body, not the roundup index text
    assert "SEO" in by_title["Marketing Manager"].raw_description
    assert "payroll" in by_title["HR Officer"].raw_description
    assert "financial modelling" in by_title["Finance Analyst"].raw_description
    # the index post itself is not emitted as a job
    assert all("careers-roundup" not in p.source_url for p in result.postings)
    # site chrome did not leak into the body
    assert "site footer" not in by_title["Marketing Manager"].raw_description


def test_each_child_url_is_its_own_job_page_for_idempotent_identity():
    adapter, _ = _adapter()
    urls = {p.source_url for p in adapter.fetch().postings}
    assert urls == {
        f"{HOST}/2026/07/marketing-manager.html",
        f"{HOST}/2026/07/hr-officer.html",
        f"{HOST}/2026/07/finance-analyst.html",
    }, "a child must use its own job-page URL so it converges with a direct feed entry"


def test_single_job_post_is_not_expanded():
    """A post with fewer than min_links same-site job links stays one posting —
    the ordinary case must not regress."""
    single = _roundup_feed().replace(
        f'&lt;p&gt;hr-officer: &lt;a href="/2026/07/hr-officer.html"&gt;Apply&lt;/a&gt;&lt;/p&gt;', ""
    ).replace(
        f'&lt;p&gt;finance-analyst: &lt;a href="/2026/07/finance-analyst.html"&gt;Apply&lt;/a&gt;&lt;/p&gt;', ""
    )
    # now the entry has a single same-site job link -> below the threshold
    adapter, client = _adapter(feed=single)
    result = adapter.fetch()
    assert len(result.postings) == 1
    assert "careers-roundup" in result.postings[0].source_url
    assert not any("/2026/07/marketing-manager.html" in u for u, _ in client.requests)


def test_expansion_disabled_keeps_the_roundup_as_one_posting():
    adapter, _ = _adapter(config=Config(blogger_expand_roundups=False))
    result = adapter.fetch()
    assert len(result.postings) == 1
    assert "careers-roundup" in result.postings[0].source_url


def test_link_cap_bounds_the_fetches():
    adapter, client = _adapter(config=Config(blogger_max_links_per_post=2))
    result = adapter.fetch()
    assert len(result.postings) == 2
    page_fetches = [u for u, _ in client.requests if "/2026/07/" in u and "roundup" not in u]
    assert len(page_fetches) == 2


def test_a_robots_blocked_job_page_is_skipped_others_survive():
    class DenyFinance:
        def can_fetch(self, url):
            from agents.agent_b_job_ingest.sources.robots import RobotsDecision

            return RobotsDecision("finance-analyst" not in url, "test")

        def require(self, url):
            if "finance-analyst" in url:
                raise Blocked("robots denies this page")
            return None

    adapter, _ = _adapter(robots=DenyFinance())
    titles = {p.title for p in adapter.fetch().postings}
    assert titles == {"Marketing Manager", "HR Officer"}, "blocked page skipped, rest emitted"
