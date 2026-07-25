"""Course adapters, offline against fixtures — parsing, pagination, filters."""

from __future__ import annotations

from agents.agent_d_course_ingest.sources.coursera import CourseraAdapter
from agents.agent_d_course_ingest.sources.freecodecamp import FreeCodeCampAdapter
from shared.config import Config
from tests.agent_d.fake_client import AllowAllRobots, FakeClient, fixture


# ---------------------------------------------------------------------------
# Coursera (api)
# ---------------------------------------------------------------------------
def coursera(page1=None, page2=None):
    client = FakeClient({"/api/courses.v1": [
        page1 or fixture("coursera_courses.json"),
        page2 or fixture("coursera_page2.json"),
    ]})
    return CourseraAdapter(client=client, config=Config()), client


def test_coursera_resolves_partner_names_from_linked():
    adapter, _ = coursera()
    courses = {c.name: c for c in adapter.fetch().courses}
    assert courses["Python Basics"].provider == "Google Cloud"
    assert courses["SPSS: Logistic Regression"].provider == "IBM"


def test_coursera_course_url_is_built_from_the_slug():
    course = coursera()[0].fetch().courses[0]
    assert course.source_url == "https://www.coursera.org/learn/python-basics"


def test_coursera_skips_non_english_and_slugless_elements():
    courses = coursera()[0].fetch().courses
    names = {c.name for c in courses}
    assert "Cours de Francais" not in names, "a non-English course was ingested"
    assert "No Slug Course" not in names, "a slugless element was ingested"
    assert all(c.primary_language == "en" for c in courses)


def test_coursera_follows_pagination_until_next_is_absent():
    courses = coursera()[0].fetch().courses
    # page 1 yields 2 eligible (python, spss); page 2 yields 1 (data-analysis-r)
    assert {c.name for c in courses} >= {"Python Basics", "Data Analysis with R"}


def test_coursera_license_is_none_but_attribution_names_the_provider():
    course = coursera()[0].fetch().courses[0]
    assert course.license is None
    assert "Coursera" in course.attribution


def test_coursera_limit_is_honoured():
    assert len(coursera()[0].fetch(limit=1).courses) == 1


def test_coursera_malformed_json_returns_errored_result():
    adapter, _ = coursera(page1="{not json")
    result = adapter.fetch()
    assert result.courses == [] and result.error and not result.ok


# ---------------------------------------------------------------------------
# freeCodeCamp (html_scrape)
# ---------------------------------------------------------------------------
def fcc():
    client = FakeClient({"/news/freecodecamp-certifications/": fixture("freecodecamp_article.html")})
    return FreeCodeCampAdapter(client=client, robots=AllowAllRobots(), config=Config())


def test_fcc_extracts_distinct_certifications_deduped_and_year_stripped():
    courses = fcc().fetch().courses
    urls = [c.source_url for c in courses]

    assert "https://www.freecodecamp.org/learn/responsive-web-design" in urls
    assert "https://www.freecodecamp.org/learn/relational-database" in urls
    # the /learn/2022/responsive-web-design/ and the duplicate collapse:
    assert len(urls) == len(set(urls)), "duplicate certifications leaked"
    assert not any("2022" in u for u in urls), "a year segment survived in the slug"
    # the /news/ link is not a course
    assert not any("/news/" in u for u in urls)


def test_fcc_name_is_titleized_from_the_slug():
    by_url = {c.source_url.rsplit("/", 1)[-1]: c for c in fcc().fetch().courses}
    assert by_url["responsive-web-design"].name == "Responsive Web Design"
    assert by_url["back-end-development-and-apis"].name == "Back End Development and APIs"


def test_fcc_carries_the_ccbysa_license_and_attribution():
    course = fcc().fetch().courses[0]
    assert course.license == "CC-BY-SA-4.0"
    assert "freeCodeCamp" in course.attribution
    assert course.provider == "freeCodeCamp"


def test_fcc_courses_are_free_with_amount_zero_not_null():
    """A free course is amount 0.0 / is_free True — never null amount. currency
    is None (no currency applies to $0). No rating/enrollment (unrated SPA)."""
    course = fcc().fetch().courses[0]
    assert course.price == {"amount": 0.0, "currency": None, "is_free": True}
    assert course.rating is None and course.enrollment_count is None


# ---------------------------------------------------------------------------
# Coursera page-scrape enrichment (rating / reviews / enrollment)
# ---------------------------------------------------------------------------
def coursera_enriched(page):
    api = FakeClient({"/api/courses.v1": [fixture("coursera_courses.json"),
                                          fixture("coursera_page2.json")]})
    page_client = FakeClient({"/learn/": page})
    return CourseraAdapter(client=api, page_client=page_client, robots=AllowAllRobots(),
                           config=Config()), page_client


def test_coursera_enrichment_parses_rating_reviews_enrollment():
    adapter, _ = coursera_enriched(fixture("coursera_course_page.html"))
    course = adapter.fetch(limit=1).courses[0]
    assert course.rating == 4.9                 # 4.896... rounded to native precision
    assert course.review_count == 32430
    assert course.enrollment_count == 1219611
    # Coursera exposes no price/last_updated on the page -> stay None, not guessed.
    assert course.price is None and course.last_updated is None


def test_coursera_enrichment_missing_fields_stay_none_and_do_not_crash():
    """A bare course page (a lab, no ratings) must leave rating/reviews/
    enrollment None and still ingest the course."""
    adapter, _ = coursera_enriched(fixture("coursera_course_page_bare.html"))
    course = adapter.fetch(limit=1).courses[0]
    assert course.rating is None
    assert course.review_count is None
    assert course.enrollment_count is None
    assert course.name == "Python Basics"       # course still ingested


def test_coursera_enrichment_disabled_skips_page_fetch():
    api = FakeClient({"/api/courses.v1": [fixture("coursera_courses.json")]})
    page = FakeClient({"/learn/": fixture("coursera_course_page.html")})
    adapter = CourseraAdapter(client=api, page_client=page, robots=AllowAllRobots(),
                              config=Config(coursera_enrich=False))
    course = adapter.fetch(limit=1).courses[0]
    assert course.rating is None
    assert page.requests == [], "page was fetched despite coursera_enrich=False"


def test_coursera_enrichment_robots_block_leaves_fields_none_but_keeps_course():
    class Deny:
        def can_fetch(self, url):
            from shared.scraping.robots import RobotsDecision
            return RobotsDecision(False, "deny")

    api = FakeClient({"/api/courses.v1": [fixture("coursera_courses.json")]})
    page = FakeClient({"/learn/": fixture("coursera_course_page.html")})
    adapter = CourseraAdapter(client=api, page_client=page, robots=Deny(), config=Config())
    course = adapter.fetch(limit=1).courses[0]
    assert course.rating is None and course.name == "Python Basics"


def test_fcc_honours_robots_via_require():
    """Unlike the Coursera API, freeCodeCamp is a web scrape and must consult
    robots. A denying policy stops it."""
    class Deny:
        def require(self, url):
            from shared.scraping.http import Blocked
            raise Blocked("robots denies")
        def can_fetch(self, url):
            from shared.scraping.robots import RobotsDecision
            return RobotsDecision(False, "deny")

    client = FakeClient({"/news/": fixture("freecodecamp_article.html")})
    adapter = FreeCodeCampAdapter(client=client, robots=Deny(), config=Config())
    result = adapter.fetch()
    assert result.courses == [] and not result.ok
