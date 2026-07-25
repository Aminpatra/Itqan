"""freeCodeCamp adapter (source_type='html_scrape').

robots.txt is fully open (``Allow: /``), and unlike the Coursera API this is a
web scrape, so it DOES honor robots via ``shared.scraping.robots``.

The 11 certifications are the course-analog. Their ``/learn/`` pages are
client-rendered SPAs with no server content, but the public certifications
ARTICLE is server-rendered and links to every one. So the adapter reads the
article, extracts the distinct certification ``/learn/{slug}`` links (the list is
fetched live, not hardcoded), and derives each course's name by formatting its
slug — a deterministic transform of fetched data. Skills are extracted downstream
from the descriptive names ("Machine Learning with Python" → Python, ML).

License: the curriculum is CC-BY-SA-4.0, which requires attribution; every row
carries ``license`` and ``attribution``. We catalog skill facts + a link, never
redistribute curriculum content.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit

from selectolax.parser import HTMLParser

from shared.config import Config
from shared.scraping.http import Blocked, PoliteClient, ResponseTooLarge, SourcePolicy
from shared.scraping.robots import RobotsPolicy

from .base import AdapterResult, BaseAdapter, RawCourse

ARTICLE_PATH = "/news/freecodecamp-certifications/"
LEARN_URL = "https://www.freecodecamp.org/learn/{slug}"
LICENSE = "CC-BY-SA-4.0"
ATTRIBUTION = "freeCodeCamp (curriculum licensed CC-BY-SA-4.0)"

# Slugs are learn paths; a leading year segment (/learn/2022/...) is versioning,
# not part of the certification identity.
_YEAR = re.compile(r"^\d{4}$")
_SMALL = {"with", "and", "the", "of", "for", "to", "in", "a", "an"}


class FreeCodeCampAdapter(BaseAdapter):
    source_type = "html_scrape"

    def __init__(
        self,
        *,
        name: str = "freecodecamp",
        source_group: str = "freecodecamp",
        base_url: str = "https://www.freecodecamp.org",
        client: PoliteClient | None = None,
        config: Config | None = None,
        robots: RobotsPolicy | None = None,
        is_known_unchanged=None,
    ) -> None:
        super().__init__(name=name, source_group=source_group)
        self.config = config or Config()
        self.base_url = base_url.rstrip("/")
        self._client = client
        self._robots = robots
        self._is_known_unchanged = is_known_unchanged

    @property
    def article_url(self) -> str:
        return f"{self.base_url}{ARTICLE_PATH}"

    def _ensure_client(self) -> PoliteClient:
        if self._client is None:
            self._client = PoliteClient(
                source=self.name,
                policy=SourcePolicy(min_interval_s=2.0, max_bytes=self.config.max_response_bytes),
                config=self.config,
            )
        if self._robots is None:
            self._robots = RobotsPolicy(self._client, user_agent=self.config.user_agent)
        return self._client

    def _fetch(self, result: AdapterResult, *, limit: int | None) -> None:
        client = self._ensure_client()
        assert self._robots is not None
        self._robots.require(self.article_url)

        try:
            html = client.get_text(self.article_url)
        except (Blocked, ResponseTooLarge) as exc:
            result.fail(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            result.fail(f"{type(exc).__name__}: {exc}")
            return

        result.pages_fetched += 1
        result.bytes_fetched = client.bytes_fetched

        tree = HTMLParser(html)
        seen: set[str] = set()
        for node in tree.css("a[href*='/learn/']"):
            slug = _slug_of(node.attributes.get("href") or "")
            if not slug or slug in seen:
                continue
            seen.add(slug)
            course = self._course(slug)
            if self._is_known_unchanged and self._is_known_unchanged(course):
                continue
            result.courses.append(course)
            if limit is not None and len(result.courses) >= limit:
                return

    def _course(self, slug: str) -> RawCourse:
        name = _titleize(slug)
        return RawCourse(
            source=self.name,
            source_group=self.source_group,
            source_type=self.source_type,
            source_url=LEARN_URL.format(slug=slug),
            name=name,
            # Factual, not fabricated: each IS a free freeCodeCamp certification
            # with this name. Skills are extracted from the descriptive name.
            raw_description=f"{name}. A free certification from freeCodeCamp.",
            provider="freeCodeCamp",
            level=None,
            primary_language="en",
            attribution=ATTRIBUTION,
            license=LICENSE,
            # freeCodeCamp certifications are free. amount 0.0 (NOT null), no
            # currency (none applies to a $0 course). It exposes no rating /
            # enrollment / last-updated, so those stay None — a missing signal is
            # missing, never a fabricated 0.
            price={"amount": 0.0, "currency": None, "is_free": True},
        )


def _slug_of(href: str) -> str:
    """Last meaningful path segment of a /learn/ link, dropping a year segment
    and any trailing slash. '/learn/2022/responsive-web-design/' -> that slug."""
    path = urlsplit(href).path
    segments = [s for s in path.split("/") if s]
    if "learn" not in segments:
        return ""
    tail = segments[segments.index("learn") + 1:]
    tail = [s for s in tail if not _YEAR.match(s)]
    return tail[-1] if tail else ""


def _titleize(slug: str) -> str:
    words = slug.replace("-", " ").split()
    out = []
    for i, w in enumerate(words):
        out.append(w if (w in _SMALL and i != 0) else w.capitalize())
    # Common acronyms read better upper-cased.
    return " ".join(out).replace("Apis", "APIs")
