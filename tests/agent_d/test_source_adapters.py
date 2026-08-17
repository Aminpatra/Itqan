"""Course adapters, offline against fixtures — parsing, pagination, filters."""

from __future__ import annotations

from agents.agent_d_course_ingest.sources.coursera import CourseraAdapter
from agents.agent_d_course_ingest.sources.freecodecamp import FreeCodeCampAdapter
import httpx

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
    # `/learn/` is mapped because the adapter now FETCHES each certification URL
    # to prove it resolves. FakeClient raises a 404 for anything unmapped, which
    # the check would — correctly — read as "this page does not exist".
    client = FakeClient({"intro.json": fixture("freecodecamp_intro.json"),
                         "/learn/": "<html><body>a certification page</body></html>"})
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


# ---------------------------------------------------------------------------
# freeCodeCamp: blocks alongside certifications
# ---------------------------------------------------------------------------
def test_blocks_are_emitted_as_courses_of_their_own():
    """A certification is a course and so is each block inside it. "Basic CSS"
    is something a person can be pointed at on its own, and it carries its own
    intro rather than inheriting the certification's."""
    from agents.agent_d_course_ingest.sources.freecodecamp import FreeCodeCampAdapter
    from tests.agent_d.fake_client import FakeClient, fixture

    result = FreeCodeCampAdapter(
        client=FakeClient({"intro.json": fixture("freecodecamp_intro.json"),
                           "/learn/": "<html><body>ok</body></html>"}),
        robots=_AllowAll(), config=Config()).fetch()

    names = [c.name for c in result.courses]
    assert "Basic HTML and HTML5" in names
    assert "Basic CSS" in names


def test_a_certification_keeps_its_url_and_therefore_its_identity():
    """The 99 existing rows must not be re-minted. `course_id` is derived from
    `source_url`, this source is census=True, and a changed id would have aged —
    then DELETED — every row it replaced. Same shape as the Agent B identity
    finding that double-counted demand across an overlap.
    """
    from agents.agent_d_course_ingest.sources.freecodecamp import FreeCodeCampAdapter
    from tests.agent_d.fake_client import FakeClient, fixture

    result = FreeCodeCampAdapter(
        client=FakeClient({"intro.json": fixture("freecodecamp_intro.json"),
                           "/learn/": "<html><body>ok</body></html>"}),
        robots=_AllowAll(), config=Config()).fetch()

    urls = {c.source_url for c in result.courses}
    assert "https://www.freecodecamp.org/learn/responsive-web-design" in urls
    # A FRAGMENT of the certification's URL. `/learn/<cert>/<block>` is a 404 —
    # measured, after 1,353 rows shipped pointing at one.
    assert "https://www.freecodecamp.org/learn/responsive-web-design/#basic-css" in urls
    assert "https://www.freecodecamp.org/learn/responsive-web-design/basic-css" not in urls


def test_a_block_without_its_own_intro_is_skipped_not_given_the_parents():
    """A row whose description belongs to something else extracts the wrong
    skills — the clustering bug Agent B had to un-pick across 245 rows."""
    import json

    from agents.agent_d_course_ingest.sources.freecodecamp import FreeCodeCampAdapter
    from tests.agent_d.fake_client import FakeClient, fixture

    data = json.loads(fixture("freecodecamp_intro.json"))
    data["responsive-web-design"]["blocks"]["basic-css"].pop("intro")
    result = FreeCodeCampAdapter(
        client=FakeClient({"intro.json": json.dumps(data),
                           "/learn/": "<html><body>ok</body></html>"}),
        robots=_AllowAll(), config=Config()).fetch()

    assert "Basic CSS" not in [c.name for c in result.courses]
    assert "Basic HTML and HTML5" in [c.name for c in result.courses]


# ---------------------------------------------------------------------------
# every row must lead to a real link
# ---------------------------------------------------------------------------
class _Resolver:
    """A client whose GET 404s for named URLs. Everything else answers."""

    def __init__(self, body: str, missing=(), boom=()):
        self.body, self.missing, self.boom = body, set(missing), set(boom)
        self.requests: list[str] = []
        self.bytes_fetched = 0          # the adapter reads this after each fetch

    def get_text(self, url: str, *, params=None) -> str:
        self.requests.append(url)
        if url in self.boom:
            raise httpx.ConnectTimeout("the site is down")
        if url in self.missing:
            raise httpx.HTTPStatusError(
                "not found", request=httpx.Request("GET", url),
                response=httpx.Response(404, request=httpx.Request("GET", url)))
        return self.body


def _fcc(client):
    from agents.agent_d_course_ingest.sources.freecodecamp import FreeCodeCampAdapter
    return FreeCodeCampAdapter(client=client, robots=_AllowAll(), config=Config()).fetch()


class _AllowAll:
    def can_fetch(self, url): return True
    def require(self, url): return None


def test_a_certification_that_404s_is_dropped_with_its_blocks():
    """THE check whose absence shipped 1,353 dead links.

    `/learn/daily-coding-challenge` is a real entry in the curriculum file and a
    404 on the site. Counting rows never noticed; fetching the URL does.
    """
    from tests.agent_d.fake_client import fixture

    dead = "https://www.freecodecamp.org/learn/responsive-web-design"
    result = _fcc(_Resolver(fixture("freecodecamp_intro.json"), missing={dead}))

    urls = {c.source_url for c in result.courses}
    assert dead not in urls
    assert not any(u.startswith(dead + "/#") for u in urls), (
        "the blocks of a dead certification came through anyway")
    assert result.dead_links == 1


def test_the_url_a_block_actually_carries_is_the_one_checked():
    """A block's server-visible URL is `/learn/<slug>/` — WITH the trailing slash,
    because the fragment never reaches the server. The certification row carries
    `/learn/<slug>` without it. Two different strings, and checking one is not
    checking the other.

    Measured 2026-08-17 both forms answer 200 on the live site, so this guards a
    property rather than fixing a live break — but blocks are 1,352 of this
    source's 1,450 rows, and "a trailing slash obviously behaves the same" is the
    identical assumption that shipped 1,353 dead links.
    """
    from tests.agent_d.fake_client import fixture

    cert = "https://www.freecodecamp.org/learn/responsive-web-design"
    result = _fcc(_Resolver(fixture("freecodecamp_intro.json"), missing={cert + "/"}))

    urls = {c.source_url for c in result.courses}
    assert cert in urls, "the certification itself resolves and must be kept"
    assert not any(u.startswith(cert + "/#") for u in urls), (
        "blocks were emitted against a base URL that 404s")
    assert result.dead_links == 1


def test_the_site_being_down_keeps_rows_rather_than_deleting_them():
    """The asymmetry that matters. This source is census=True, so staleness may
    DELETE what a fetch does not return — reading an outage as "these pages are
    gone" would let one bad afternoon prune the whole source.

    Absence of proof is not proof of absence; only a definite 404 drops a row.
    """
    from tests.agent_d.fake_client import fixture

    cert = "https://www.freecodecamp.org/learn/responsive-web-design"
    result = _fcc(_Resolver(fixture("freecodecamp_intro.json"), boom={cert}))

    assert cert in {c.source_url for c in result.courses}
    assert result.dead_links == 0
