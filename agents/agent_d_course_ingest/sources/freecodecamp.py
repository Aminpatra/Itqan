"""freeCodeCamp adapter (source_type='html_scrape').

Reads the **public curriculum** from freeCodeCamp's own repository rather than
scraping the site, because the site cannot supply it: ``/learn/`` is a Gatsby SPA
that ships 499 KB of HTML containing zero course links and ~5 KB of rendered
text, and its ``page-data.json`` is 266 bytes — the curriculum arrives in
build-hashed chunks that change every deploy. The repo file is the same content,
published deliberately, under a licence that permits this.

``client/i18n/locales/english/intro.json`` carries every superblock (a course) with
its title, its introduction, and the titles of the blocks inside it. That is
**100 courses** where the previous source — a news article listing the
certifications — yielded 11, and each one now arrives with a real syllabus
instead of the synthesized one-liner "X. A free certification from freeCodeCamp."
that left several courses with no extractable skill at all.

Consent: robots is checked and allows it (this is a scrape, so it honors robots,
unlike the terms-gated Coursera API). The curriculum is **CC-BY-SA-4.0**, which
requires attribution, so every row carries ``license`` and ``attribution``. We
catalog skill facts and a link back; we never redistribute curriculum content.

Identity is unchanged for courses that already existed: a superblock key IS the
slug in ``/learn/{slug}``, which is the URL the previous adapter used, so
``course_id = sha(freecodecamp, url)`` still resolves to the same row.
"""

from __future__ import annotations

import json
import re
from urllib.parse import urlsplit

from shared.config import Config
from shared.scraping.http import Blocked, PoliteClient, ResponseTooLarge, SourcePolicy
from shared.scraping.robots import RobotsPolicy

from .base import AdapterResult, BaseAdapter, RawCourse

CURRICULUM_URL = (
    "https://raw.githubusercontent.com/freeCodeCamp/freeCodeCamp/main"
    "/client/i18n/locales/english/intro.json"
)
LEARN_URL = "https://www.freecodecamp.org/learn/{slug}"
LICENSE = "CC-BY-SA-4.0"
ATTRIBUTION = "freeCodeCamp (curriculum licensed CC-BY-SA-4.0)"

# Blocks listed in a course's description. Enough to name what it covers without
# reproducing the curriculum itself.
MAX_BLOCKS_LISTED = 40

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
        curriculum_url: str = CURRICULUM_URL,
    ) -> None:
        super().__init__(name=name, source_group=source_group)
        self.config = config or Config()
        self.base_url = base_url.rstrip("/")
        self.curriculum_url = curriculum_url
        self._client = client
        self._robots = robots
        self._is_known_unchanged = is_known_unchanged

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
        self._robots.require(self.curriculum_url)

        try:
            body = client.get_text(self.curriculum_url)
        except (Blocked, ResponseTooLarge) as exc:
            result.fail(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            result.fail(f"{type(exc).__name__}: {exc}")
            return

        result.pages_fetched += 1
        result.bytes_fetched = client.bytes_fetched

        try:
            curriculum = json.loads(body)
        except json.JSONDecodeError as exc:
            result.fail(f"curriculum was not JSON: {exc}")
            return
        if not isinstance(curriculum, dict):
            result.fail("curriculum was not an object of superblocks")
            return

        for slug, payload in curriculum.items():
            course = self._course(slug, payload)
            if course is None:
                # A structural entry with no title or no description — real in
                # this file (e.g. shared UI strings), not a course. Counted so a
                # sudden rise is visible, and so a wholly unparsable file is
                # still distinguishable from an empty one.
                result.skipped += 1
                continue
            if self._is_known_unchanged and self._is_known_unchanged(course):
                continue
            result.courses.append(course)
            if limit is not None and len(result.courses) >= limit:
                return

        # The file loaded but nothing in it parsed as a course: freeCodeCamp has
        # not withdrawn its curriculum, the file changed shape. This source is
        # census=True, so a silent empty result would age — and eventually
        # delete — every course it owns.
        result.anchor_missed("any superblock in the curriculum file")

    # ------------------------------------------------------------------
    def _course(self, slug: str, payload: object) -> RawCourse | None:
        if not slug or not isinstance(payload, dict):
            return None
        title = (payload.get("title") or "").strip() or _titleize(slug)
        description = _describe(payload)
        if not title or not description:
            return None

        return RawCourse(
            source=self.name,
            source_group=self.source_group,
            source_type=self.source_type,
            source_url=LEARN_URL.format(slug=slug),
            name=title,
            raw_description=description,
            provider="freeCodeCamp",
            level=None,                  # the curriculum does not label difficulty
            primary_language="en",
            attribution=ATTRIBUTION,
            license=LICENSE,
            # freeCodeCamp's curriculum is free. amount 0.0 (NOT null), no
            # currency (none applies to a $0 course). It exposes no rating /
            # enrollment / last-updated, so those stay None — a missing signal is
            # missing, never a fabricated 0. This IS an observation: the price is
            # a stated fact about the source, not a lookup that failed.
            price={"amount": 0.0, "currency": None, "is_free": True},
            volatile_observed=True,
        )


def _describe(payload: dict) -> str:
    """The course's own introduction, plus the titles of the blocks it contains.

    The block titles are the syllabus, and they are what makes extraction work:
    "Learn CSS Flexbox" alone names one skill, while its blocks name the specific
    properties and techniques taught. The previous adapter had no description at
    all beyond the course name, which is why several courses extracted nothing
    and were rejected as empty.
    """
    parts: list[str] = []
    intro = payload.get("intro")
    if isinstance(intro, list):
        parts.extend(str(p).strip() for p in intro if str(p).strip())
    elif isinstance(intro, str) and intro.strip():
        parts.append(intro.strip())

    blocks = payload.get("blocks")
    if isinstance(blocks, dict):
        titles: list[str] = []
        for block in blocks.values():
            if not isinstance(block, dict):
                continue
            name = (block.get("title") or "").strip()
            if name:
                titles.append(name)
        if titles:
            parts.append("This course covers: " + ", ".join(titles[:MAX_BLOCKS_LISTED]) + ".")

    return "\n".join(parts).strip()


def _slug_of(href: str) -> str:
    """Last meaningful path segment of a /learn/ link, dropping a year segment
    and any trailing slash. Retained for callers that still resolve site links."""
    path = urlsplit(href).path
    segments = [s for s in path.split("/") if s]
    if "learn" not in segments:
        return ""
    tail = segments[segments.index("learn") + 1:]
    tail = [s for s in tail if not _YEAR.match(s)]
    return tail[-1] if tail else ""


def _titleize(slug: str) -> str:
    words = slug.replace("/", " ").replace("-", " ").split()
    out = []
    for i, w in enumerate(words):
        out.append(w if (w in _SMALL and i != 0) else w.capitalize())
    return " ".join(out).replace("Apis", "APIs")
