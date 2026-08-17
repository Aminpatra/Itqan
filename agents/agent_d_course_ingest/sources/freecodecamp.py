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

import httpx

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

            # Does this certification actually EXIST as a page?
            #
            # The check whose absence let 1,353 dead links ship: this adapter
            # counted rows and never once asked whether a link worked. Measured
            # 2026-08-16, `/learn/daily-coding-challenge` is a 404 — it is a
            # curriculum grouping with no `intro`, and it became a row only
            # because the description fell back to listing its one block title.
            #
            # 98 requests cover all ~1,451 rows, because a block's URL is now a
            # FRAGMENT of its certification's: the certification resolving is
            # exactly the condition for the block resolving. So one check per
            # superblock validates the superblock and everything inside it.
            if not self._resolves(client, course.source_url, result):
                result.skipped += 1
                continue

            if not (self._is_known_unchanged and self._is_known_unchanged(course)):
                result.courses.append(course)
                if limit is not None and len(result.courses) >= limit:
                    return

            # Then each BLOCK inside it, as its own row.
            #
            # A superblock is a whole certification ("Responsive Web Design");
            # a block is a self-contained unit ("Basic CSS", "CSS Flexbox") that
            # a person can be pointed at on its own, and it carries its own
            # intro. Emitting both takes freeCodeCamp from 99 rows to several
            # hundred without touching a single existing `course_id` — the
            # superblock rows keep their identity, so nothing is orphaned and
            # nothing ages out. That mattered: this source is census=True, and a
            # re-minted id would have deleted the 99 rows it replaced.
            #
            # HONEST COST, stated rather than discovered: a certification and its
            # own blocks both teach CSS, so `skill_supply_stats` counts more
            # courses per skill than before. The count becomes "how many distinct
            # things teach this", which is defensible — but the demand-vs-supply
            # join reads differently, so it wants a before/after rather than a
            # shrug.
            # A block's server-visible URL is NOT the one checked above.
            #
            # The certification row carries `/learn/<slug>`; a block carries
            # `/learn/<slug>/#<key>`, whose fragment never reaches the server —
            # so what the server actually sees is `/learn/<slug>/`, WITH the
            # trailing slash. Those are two different strings to a server, and
            # verifying one is not verifying the other.
            #
            # Measured 2026-08-17: all 98 of both forms answer 200, so no stored
            # row is wrong today. It is checked anyway because blocks are 1,352 of
            # this source's 1,450 rows: trusting that a trailing slash "obviously"
            # behaves the same is the identical assumption that shipped 1,353 dead
            # links, and it would fail exactly as silently.
            #
            # Costs one extra request per certification per three-day cycle.
            if not self._resolves(client, f"{LEARN_URL.format(slug=slug)}/", result):
                result.skipped += 1
                continue

            for block in self._blocks(slug, payload):
                if self._is_known_unchanged and self._is_known_unchanged(block):
                    continue
                result.courses.append(block)
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


    def _resolves(self, client: PoliteClient, url: str, result: AdapterResult) -> bool:
        """Does this URL answer? A definite 404 is the only reason to say no.

        **A transport failure means KEEP the row**, and that asymmetry is
        load-bearing. This source is `census=True`, which licenses staleness to
        delete courses missing from a fetch — so reading a freeCodeCamp outage as
        "these pages are gone" would let one bad afternoon prune the whole
        source. Absence of proof is not proof of absence: the same rule
        `RobotsPolicy` applies in the other direction when it fails closed.

        A GET rather than a HEAD because `PoliteClient` only speaks `get_text`,
        and a transport that cannot be reached through the polite client is not
        worth reaching around — the interval, the identifying user agent and the
        size cap all live in it. One request per certification, once per
        three-day cycle.
        """
        try:
            client.get_text(url)
        except Blocked:
            # 403/429 is the site declining to answer us, not the page being
            # gone. Keep the row.
            return True
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                result.dead_links += 1
                return False
            return True          # 500s and the rest: unproven
        except Exception:        # noqa: BLE001 - timeout, DNS, reset
            return True
        return True

    def _blocks(self, superblock: str, payload: object) -> list[RawCourse]:
        """One course per block inside a certification.

        A block with no title or no intro is skipped rather than given the
        superblock's text: a row whose description belongs to something else
        extracts the wrong skills, which is the clustering bug Agent B had to
        un-pick across 245 rows.
        """
        if not isinstance(payload, dict):
            return []
        blocks = payload.get("blocks")
        if not isinstance(blocks, dict):
            return []

        out: list[RawCourse] = []
        for key, block in blocks.items():
            if not key or not isinstance(block, dict):
                continue
            title = (block.get("title") or "").strip()
            intro = _intro_text(block.get("intro"))
            if not title or not intro:
                continue
            out.append(RawCourse(
                source=self.name,
                source_group=self.source_group,
                source_type=self.source_type,
                # A FRAGMENT of the certification's URL, not a page of its own.
                # Measured 2026-08-16: `/learn/<cert>/<block>` is a 404 for every
                # block — freeCodeCamp renders blocks inside the certification
                # page, and the block slug is not even in the rendered markup.
                # 1,353 rows shipped with a dead link before this was checked.
                #
                # The fragment never reaches the server, so this resolves 200 to
                # the certification that genuinely contains the block. It is also
                # what keeps `course_id` distinct: `sha(source, source_url)` over
                # the bare certification URL would collapse every block of a
                # certification into one row. Do not 'tidy' it away.
                source_url=f"{LEARN_URL.format(slug=superblock)}/#{key}",
                name=title,
                raw_description=intro,
                provider="freeCodeCamp",
                level=None,
                primary_language="en",
                attribution=ATTRIBUTION,
                license=LICENSE,
                price={"amount": 0.0, "currency": None, "is_free": True},
                volatile_observed=True,
            ))
        return out


def _intro_text(intro: object) -> str:
    """A block's intro is a list of paragraphs, occasionally a bare string."""
    if isinstance(intro, list):
        return "\n".join(str(p).strip() for p in intro if str(p).strip()).strip()
    if isinstance(intro, str):
        return intro.strip()
    return ""


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
