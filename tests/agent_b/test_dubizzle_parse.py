"""The Dubizzle html_scrape adapter, offline against fixtures.

This source is the reason migration 0006 exists. It is a classifieds site: a
large share of its job listings are people advertising themselves, and the page
names no poster at all. The tests that matter here are the ones asserting the
adapter does NOT paper over either fact.
"""

from __future__ import annotations

import sys

import pytest

from agents.agent_b_job_ingest.sources.dubizzle import DubizzleAdapter
from shared.config import Config
from tests.agent_b.fake_source_client import AllowAllRobots, FakeClient, fixture


def build(**kwargs) -> DubizzleAdapter:
    client = FakeClient(
        {
            "/en/jobs-services/": fixture("dubizzle_listing.html"),
            "warehouse-supervisor": fixture("dubizzle_detail_vacancy.html"),
            "looking-for-job-as-draughtsman": fixture("dubizzle_detail_seeking.html"),
        }
    )
    return DubizzleAdapter(
        base_url="https://example.test",
        client=client,
        robots=AllowAllRobots(),
        config=Config(),
        **kwargs,
    )


def postings():
    return build().fetch().postings


# ---------------------------------------------------------------------------
# the two facts this source forced into the schema
# ---------------------------------------------------------------------------
def test_a_seeker_ad_is_marked_seeking_from_the_publishers_own_category():
    """Counting a job-seeker as a posting measures SUPPLY and publishes it as
    demand — Agent C would report draughtsman skills in demand when what was
    counted is draughtsmen looking for work. The row looks entirely normal, so
    nothing downstream could detect it."""
    seeker = [p for p in postings() if "draughtsman" in p.source_url][0]

    assert seeker.listing_intent == "seeking"


def test_intent_is_never_guessed_from_a_vacancy_category():
    """Only "Jobs Wanted" is a STATED claim. A vacancy category is not evidence
    of a vacancy: the live sample had seekers throughout the IT-Telecom vacancy
    category, so trusting the category would be wrong in exactly the cases that
    matter."""
    vacancy = [p for p in postings() if "warehouse" in p.source_url][0]

    assert vacancy.listing_intent == "unknown", "a vacancy category was treated as proof"


def test_poster_type_is_always_unknown_because_the_page_names_nobody():
    """No seller name, no company name, no business badge — an anonymous user
    photo is the only poster element. Under migration 0006 'unknown' is
    ineligible for aggregation, so this is what keeps the source out of
    Agent C's data until posters can actually be classified."""
    assert all(p.poster_type == "unknown" for p in postings())


def test_no_employer_is_invented():
    assert all(p.company is None for p in postings())


# ---------------------------------------------------------------------------
# relative dates
# ---------------------------------------------------------------------------
def test_relative_dates_are_kept_as_text_and_never_parsed_here():
    """"2 days ago" means a different day on every fetch. Resolving it per cycle
    would move a posting's date every time it is seen; resolution happens once,
    downstream, against first-seen time."""
    vacancy = [p for p in postings() if "warehouse" in p.source_url][0]

    assert vacancy.posted_date is None
    assert vacancy.posted_date_text == "2 days ago"


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------
def test_the_description_comes_from_the_detail_page():
    vacancy = [p for p in postings() if "warehouse" in p.source_url][0]

    assert "stock control" in vacancy.raw_description
    assert vacancy.title == "Warehouse Supervisor"


def test_the_description_heading_is_not_kept_as_content():
    vacancy = [p for p in postings() if "warehouse" in p.source_url][0]
    assert not vacancy.raw_description.startswith("Description")


def test_location_is_taken_from_the_card():
    vacancy = [p for p in postings() if "warehouse" in p.source_url][0]
    assert vacancy.location_text == "Sohar, Al Batinah"


def test_a_card_with_no_ad_link_is_skipped():
    """Inline promos render as <article> too."""
    result = build().fetch()

    assert result.skipped == 1
    assert len(result.postings) == 2


def test_nothing_selects_on_css_classes():
    """The live markup's classes are build hashes that change on every frontend
    deploy. A class-based selector would break silently and the adapter would
    report zero postings as though the site had gone quiet — so the failure
    would look like a source going dark rather than like a bug.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(sys.modules[DubizzleAdapter.__module__]))
    docstrings = {
        ast.get_docstring(n, clean=False)
        for n in ast.walk(tree)
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef))
    }
    # Only what the CODE uses. Prose naming the hashes to explain why they are
    # avoided must not fail the test that enforces avoiding them.
    literals = [
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, str)
        and n.value not in docstrings
    ]

    for hashed in ("c0b6be79", "_948d9e0a", "_02b46ab0", "f65fa7cc", "be462f54"):
        for literal in literals:
            assert hashed not in literal, f"adapter selects on the build-hash class {hashed}"


def test_limit_is_honoured_before_detail_pages_are_fetched():
    """Each posting costs TWO requests on this source, so a limit that only
    trimmed the result afterwards would still have paid for every detail page."""
    adapter = build()
    result = adapter.fetch(limit=1)

    assert len(result.postings) == 1
    fetched = [url for url, _ in adapter._client.requests]
    assert not any("draughtsman" in url for url in fetched), (
        "a detail page was fetched beyond the limit"
    )


def test_known_unchanged_listings_skip_their_detail_request():
    """The incremental path, and the reason the card is parsed before the detail
    page: on a two-request-per-posting source this is the difference between an
    incremental cycle and a full re-crawl every twelve hours."""
    adapter = build(is_known_unchanged=lambda p: "warehouse" in p.source_url)
    result = adapter.fetch()

    fetched = [url for url, _ in adapter._client.requests]
    assert not any("warehouse-supervisor" in url for url in fetched)
    assert len(result.postings) == 1


@pytest.mark.parametrize("blocked_url", ["/en/jobs-services/"])
def test_a_robots_disallow_stops_the_source_rather_than_being_ignored(blocked_url):
    class DenyAll:
        def can_fetch(self, url):
            from agents.agent_b_job_ingest.sources.robots import RobotsDecision

            return RobotsDecision(False, "test stub denies")

        def require(self, url):
            from agents.agent_b_job_ingest.sources.http import Blocked

            raise Blocked("denied")

    adapter = build()
    adapter._robots = DenyAll()
    result = adapter.fetch()

    assert result.postings == []
    assert result.error is not None
    assert not result.ok
