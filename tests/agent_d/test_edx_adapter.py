"""The edX adapter: sitemap enumeration, JSON-LD parsing, and what it refuses.

`edx_course.html` is trimmed from a REAL page fetched 2026-08-16 — the JSON-LD
is the site's own, not something invented to match the parser. The full page is
552 KB; keeping only the `@graph` makes the fixture about parsing rather than
about page weight.

Two tests carry the file. `test_a_week_long_course_reports_weeks_and_no_hours`
is the honesty one: edX states `P4W`, and converting four calendar weeks into
672 hours of study would be wrong by two orders of magnitude while sounding
precise. And `test_the_spanish_catalogue_is_not_ingested_twice` guards the
duplicate that would otherwise double every row in the corpus.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.agent_d_course_ingest.sources.edx import EdxAdapter
from shared.config import Config
from shared.scraping.robots import RobotsDecision
from tests.agent_d.fake_client import FakeClient

FIXTURES = Path(__file__).parent / "fixtures"
PAGE = (FIXTURES / "edx_course.html").read_text(encoding="utf-8")
COURSE_URL = ("https://www.edx.org/learn/writing/"
              "university-of-cambridge-stand-up-comedy-writing-and-performance-poetry")

SITEMAP = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://www.edx.org/learn/writing/university-of-cambridge-stand-up-comedy-writing-and-performance-poetry</loc></url>
  <url><loc>https://www.edx.org/es/learn/writing/university-of-cambridge-stand-up-comedy-writing-and-performance-poetry</loc></url>
  <url><loc>https://www.edx.org/learn/computer-science</loc></url>
  <url><loc>https://www.edx.org/school/harvardx</loc></url>
  <url><loc>https://www.edx.org/resources/what-is-python</loc></url>
</urlset>
"""


class AllowAll:
    """Returns the REAL decision type. A bare `True` passes the page check (which
    uses truthiness) and fails the sitemap check (which reads `.allowed`) — a fake
    that diverges from the thing it stands in for tests the wrong code."""

    def can_fetch(self, url): return RobotsDecision(True, "allowed by test")


def adapter(pages=None, **kw):
    responses = {"/sitemap.xml": SITEMAP}
    responses.update(pages or {COURSE_URL: PAGE})
    return EdxAdapter(client=FakeClient(responses), robots=AllowAll(),
                      config=Config(), **kw)


def fetch(**kw):
    return adapter(**kw).fetch()


# ---------------------------------------------------------------------------
# enumeration
# ---------------------------------------------------------------------------
def test_only_course_shaped_urls_are_taken():
    """`/learn/<topic>` is a category page and `/school/...` is an institution.
    Neither is a course, and ingesting them would inflate the supply side of the
    demand-vs-supply join with things nobody can enrol in."""
    result = fetch()

    assert len(result.courses) == 1
    assert result.courses[0].source_url == COURSE_URL


def test_the_spanish_catalogue_is_not_ingested_twice():
    """`/es/learn/...` is the SAME course. Taking both would double every row
    and double its weight in `skill_supply_stats`."""
    result = fetch()

    assert not any("/es/" in c.source_url for c in result.courses)


def test_an_empty_sitemap_is_a_shape_change_not_an_empty_catalogue():
    """5,304 courses do not vanish. Reading it as empty would age the whole
    inventory toward deletion — the failure `census=False` and this guard both
    exist to prevent."""
    empty = ('<?xml version="1.0"?><urlset '
             'xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"></urlset>')
    result = EdxAdapter(client=FakeClient({"/sitemap.xml": empty}),
                        robots=AllowAll(), config=Config()).fetch()

    assert not result.ok
    assert result.courses == []


def test_pages_that_parse_to_nothing_fail_loudly():
    """Fetched pages and zero courses means the markup moved."""
    result = fetch(pages={COURSE_URL: "<html><body>no json-ld here</body></html>"})

    assert not result.ok


# ---------------------------------------------------------------------------
# what the JSON-LD yields
# ---------------------------------------------------------------------------
def test_the_real_page_yields_the_fields_that_matter():
    course = fetch().courses[0]

    assert "Comedy Writing" in course.name
    # The teaching institution, not "edX" — the same distinction Coursera draws.
    assert course.provider == "University of Cambridge"
    assert course.primary_language == "en"
    assert course.level == "intermediate"
    assert course.raw_description


def test_the_rating_is_read_because_that_is_why_this_source_is_here():
    """250 of 2,000 Coursera courses have a rating, because most genuinely have
    none. Agent E's tiebreak ranks on rating, so a source that publishes one is
    worth more than its row count suggests."""
    course = fetch().courses[0]

    assert course.rating == 5.0
    assert course.review_count == 8
    assert course.volatile_observed is True, (
        "the page WAS read, so the store must not preserve older values over it")


def test_a_week_long_course_reports_weeks_and_no_hours():
    """THE honesty test for this adapter.

    edX states `timeRequired: P4W`. Four calendar weeks is not 672 hours of
    study, and converting it would be wrong by two orders of magnitude while
    sounding authoritative. So the words are repeated and the hours stay unset.
    """
    course = fetch().courses[0]

    assert course.duration_text == "4 weeks"

    from agents.agent_d_course_ingest.duration import parse_workload
    assert parse_workload(course.duration_text) == (None, None)


def test_the_paid_offer_is_the_price_not_the_free_audit_track():
    """edX lists a free audit track alongside a paid certificate. What a learner
    pays to COMPLETE the course is the honest figure."""
    course = fetch().courses[0]

    assert course.price == {"amount": 299.0, "currency": "USD", "is_free": False}


# ---------------------------------------------------------------------------
# politeness and safety
# ---------------------------------------------------------------------------
def test_a_course_already_held_is_not_fetched_again():
    """5,304 pages at one request each is not something to repeat every cycle."""
    client = FakeClient({"/sitemap.xml": SITEMAP, COURSE_URL: PAGE})
    result = EdxAdapter(client=client, robots=AllowAll(), config=Config(),
                        is_known_unchanged=lambda _c: True).fetch()

    assert result.courses == []
    assert len(client.requests) == 1, "only the sitemap should have been fetched"


def test_a_capped_fetch_is_marked_truncated():
    """A cap is OUR limit, not the source ending. Without this, staleness ages
    every course the run simply did not reach."""
    result = adapter().fetch(limit=0)

    assert result.truncated is True


def test_robots_refusing_the_sitemap_ends_the_fetch():
    class RefuseAll:
        def can_fetch(self, url): return RobotsDecision(False, "disallowed by test")

    result = EdxAdapter(client=FakeClient({"/sitemap.xml": SITEMAP}),
                        robots=RefuseAll(), config=Config()).fetch()

    assert not result.ok
    assert "robots" in (result.error or "").lower()


@pytest.mark.parametrize("bad", ["not xml at all", "<urlset><broken>"])
def test_an_unparseable_sitemap_fails_rather_than_reads_as_empty(bad):
    result = EdxAdapter(client=FakeClient({"/sitemap.xml": bad}),
                        robots=AllowAll(), config=Config()).fetch()

    assert not result.ok
