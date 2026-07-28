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
    client = FakeClient({"intro.json": fixture("freecodecamp_intro.json")})
    return FreeCodeCampAdapter(client=client, robots=AllowAllRobots(), config=Config())


def test_fcc_reads_every_superblock_as_a_course():
    """The site cannot supply this: /learn/ is a Gatsby SPA with zero course
    links in 499KB of HTML. The public CC-BY-SA curriculum file can, and it
    carries 100 courses where the certifications article carried 11."""
    courses = fcc().fetch().courses
    urls = [c.source_url for c in courses]

    assert "https://www.freecodecamp.org/learn/responsive-web-design" in urls
    assert "https://www.freecodecamp.org/learn/css-flexbox" in urls
    assert len(urls) == len(set(urls)), "duplicate courses leaked"
    assert not any("/news/" in u for u in urls)


def test_a_versioned_superblock_is_its_own_course_not_a_duplicate():
    """`responsive-web-design` and `2022/responsive-web-design` are two distinct
    curricula that both exist at their own /learn/ URLs. The old slug logic
    stripped the year segment, which would now collapse them into one."""
    urls = {c.source_url for c in fcc().fetch().courses}
    assert "https://www.freecodecamp.org/learn/2022/responsive-web-design" in urls
    assert "https://www.freecodecamp.org/learn/responsive-web-design" in urls


def test_fcc_identity_is_unchanged_for_courses_that_already_existed():
    """A superblock key IS the slug the previous adapter used, so switching
    sources must not re-mint course_ids and orphan the stored rows."""
    from agents.agent_d_course_ingest.hashing import id_for

    course = next(c for c in fcc().fetch().courses
                  if c.source_url.endswith("/machine-learning-with-python"))
    assert id_for(course) == id_for(course)          # stable
    assert course.source_url == "https://www.freecodecamp.org/learn/machine-learning-with-python"


def test_the_course_name_and_syllabus_come_from_the_curriculum():
    """The old adapter had no description beyond a synthesized one-liner, which
    is why courses like `information-security` extracted no skills at all and
    were rejected as empty. The blocks ARE the syllabus."""
    by_slug = {c.source_url.rsplit("/", 1)[-1]: c for c in fcc().fetch().courses}
    ml = by_slug["machine-learning-with-python"]
    assert ml.name == "Machine Learning with Python"
    assert "neural networks" in ml.raw_description
    assert "TensorFlow" in ml.raw_description, "the block titles are not in the description"


def test_an_entry_that_is_not_a_course_is_skipped_not_emitted():
    """The file also carries shared UI strings. Counted as skipped, so a wholly
    unparsable file stays distinguishable from an empty one."""
    result = fcc().fetch()
    assert result.skipped >= 1
    assert not any(c.source_url.endswith("/misc-text") for c in result.courses)


def test_fcc_carries_the_ccbysa_license_and_attribution():
    course = fcc().fetch().courses[0]
    assert course.license == "CC-BY-SA-4.0"
    assert "freeCodeCamp" in course.attribution
    assert course.provider == "freeCodeCamp"


def test_fcc_courses_are_free_with_amount_zero_not_null():
    """A free course is amount 0.0 / is_free True — never null amount. currency
    is None (no currency applies to $0). No rating/enrollment (unrated)."""
    course = fcc().fetch().courses[0]
    assert course.price == {"amount": 0.0, "currency": None, "is_free": True}
    assert course.rating is None and course.enrollment_count is None
    assert course.volatile_observed is True, "a stated price is an observation"


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

    client = FakeClient({"intro.json": fixture("freecodecamp_intro.json")})
    adapter = FreeCodeCampAdapter(client=client, robots=Deny(), config=Config())
    result = adapter.fetch()
    assert result.courses == [] and not result.ok
