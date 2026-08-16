"""edX adapter (source_type='html_scrape').

A third course source, and the one that fills the gap Coursera leaves. Measured
2026-08-16:

* **5,304 English course pages** in the sitemap (`/learn/<topic>/<slug>`);
* robots.txt is reachable and its 31 `Disallow` rules cover UTM parameters and
  asset directories — **not** course pages;
* every page carries schema.org JSON-LD in an `@graph` with `aggregateRating`,
  `educationalLevel`, `offers` and `provider`;
* **552 KB of server-rendered HTML to plain httpx.** No browser is needed, which
  is why `browser_sources` stays empty — see `shared/config.py`.

That rating matters: only 250 of 2,000 Coursera courses have one, because most
Coursera courses genuinely have none. Ratings are what Agent E's tiebreak ranks
on, so a source that publishes them is worth more than its row count suggests.

**Why the sitemap rather than the catalogue API.** edX's catalogue API needs
OAuth client credentials, which is a commercial arrangement rather than a
technical obstacle. The sitemap is public, advertised in robots.txt, and gives
the whole inventory in one request.

**JSON-LD, never a regex over markup.** `json.loads` plus explicit key reads, so
a layout change raises rather than yielding half a course.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterator
from urllib.parse import urlsplit
from xml.etree import ElementTree as ET

from selectolax.parser import HTMLParser

from shared.config import Config
from shared.scraping import build_client, build_robots
from shared.scraping.http import Blocked, PoliteClient, ResponseTooLarge, SourcePolicy
from shared.scraping.robots import RobotsPolicy

from ..duration import parse_iso8601
from .base import AdapterResult, BaseAdapter, RawCourse

SITEMAP_PATH = "/sitemap.xml"
_SITEMAP_NS = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}

# MEASURED 2026-08-16: the sitemap is ~10 MB across 17,450 URLs, well over the
# project's 5 MB default. 32 MB is headroom to the PROTOCOL's ceiling
# (sitemaps.org caps one sitemap at 50,000 URLs), not to today's file plus a
# margin — so the catalogue merely growing cannot outgrow it.
SITEMAP_MAX_BYTES = 32 * 1024 * 1024

MAX_DESCRIPTION_CHARS = 20_000

# A course page: /learn/<topic>/<slug>. The `/es/` tree is the same catalogue in
# Spanish and would double every row, so it is excluded by requiring `learn` to
# be the FIRST segment rather than by searching for "/es/" anywhere.
_COURSE_PATH = re.compile(r"^/learn/[^/]+/[^/]+$")

# edX's own vocabulary -> ours. Anything unlisted stays None: an unmapped value
# is a question, not a default.
_LEVELS = {
    "introductory": "beginner", "beginner": "beginner",
    "intermediate": "intermediate",
    "advanced": "advanced",
}


class EdxAdapter(BaseAdapter):
    source_type = "html_scrape"

    def __init__(
        self,
        *,
        name: str = "edx",
        source_group: str = "edx",
        base_url: str = "https://www.edx.org",
        client: PoliteClient | None = None,
        config: Config | None = None,
        robots: RobotsPolicy | None = None,
        is_known_unchanged=None,
        attribution: str = "edX",
        terms_url: str = "https://www.edx.org/edx-terms-service",
    ) -> None:
        super().__init__(name=name, source_group=source_group)
        self.config = config or Config()
        self.base_url = base_url.rstrip("/")
        self._client = client
        self._robots = robots
        self._is_known_unchanged = is_known_unchanged
        self.attribution = attribution
        self.terms_url = terms_url

    @property
    def sitemap_url(self) -> str:
        return f"{self.base_url}{SITEMAP_PATH}"

    def _ensure_client(self) -> PoliteClient:
        if self._client is None:
            # Through the shared factory, so Chromium becomes a config value
            # rather than a code change. Measured unnecessary here — the pages
            # are server-rendered — and therefore off.
            self._client = build_client(
                source=self.name,
                # One request per course, so the same restraint Dubizzle and
                # GulfTalent get for the same reason.
                policy=SourcePolicy(min_interval_s=3.0, max_bytes=SITEMAP_MAX_BYTES),
                config=self.config,
            )
        if self._robots is None:
            self._robots = build_robots(source=self.name, config=self.config)
        return self._client

    # ------------------------------------------------------------------
    def _fetch(self, result: AdapterResult, *, limit: int | None) -> None:
        client = self._ensure_client()
        assert self._robots is not None

        decision = self._robots.can_fetch(self.sitemap_url)
        if not decision.allowed:
            result.fail(f"robots.txt refuses {self.sitemap_url} ({decision.reason})")
            return

        try:
            urls = self._enumerate(client, result)
        except (Blocked, ResponseTooLarge) as exc:
            result.fail(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            result.fail(f"{type(exc).__name__}: {exc}")
            return
        if result.error:
            return

        if not urls:
            # The sitemap parsed and yielded no course URLs. That is either a
            # genuinely empty catalogue (it is not — 5,304 were measured) or the
            # URL shape moved. Refusing to call it empty is the conservative
            # side, because "empty" ages the whole inventory toward deletion.
            result.anchor_missed("the sitemap (no /learn/<topic>/<slug> URLs)")
            return

        # Pages that carried NO JSON-LD Course node at all, as distinct from
        # pages parsed and then filtered out (wrong language, missing title).
        # Only the first means the markup moved; conflating them would let a
        # capped run that happened to hit non-English courses report a layout
        # change that had not happened.
        unparseable = 0

        for url in urls:
            if limit is not None and len(result.courses) >= limit:
                # A capped fetch is not a census. Without this, staleness ages
                # every course we simply did not page to.
                result.truncated = True
                return

            if not self._robots.can_fetch(url):
                result.skipped += 1
                continue

            # A course we already hold costs nothing. 5,304 pages at one request
            # each is not something to repeat every cycle.
            if self._is_known_unchanged and self._is_known_unchanged(self._stub(url)):
                continue

            try:
                html = client.get_text(url)
            except (Blocked, ResponseTooLarge) as exc:
                result.fail(str(exc))
                return
            except Exception:  # noqa: BLE001 - one bad page must not end the source
                result.skipped += 1
                continue

            result.pages_fetched += 1
            result.bytes_fetched = client.bytes_fetched

            course = self._from_page(url, html)
            if course is None:
                result.skipped += 1
                if not _course_node(html):
                    unparseable += 1
                continue
            result.courses.append(course)

        # Every page we read lacked a Course node: the markup moved. `fail`
        # rather than `anchor_missed`, which by contract only fires when NOTHING
        # was skipped — and here everything was. Same shape as the GulfTalent
        # adapter's guard, and the reason it exists is the same: "the catalogue
        # is empty" and "our selectors stopped matching" are indistinguishable by
        # count, and reading the second as the first ages the inventory toward
        # deletion the day a site is redesigned.
        if unparseable and unparseable == result.skipped and not result.courses:
            result.fail(
                f"fetched {unparseable} course page(s) and parsed no JSON-LD Course; "
                f"the page structure may have changed")

    # ------------------------------------------------------------------
    def _enumerate(self, client: PoliteClient, result: AdapterResult) -> list[str]:
        """One request for the whole inventory, from the file robots advertises."""
        body = client.get_text(self.sitemap_url)
        result.pages_fetched += 1
        try:
            root = ET.fromstring(body)
        except ET.ParseError as exc:
            result.fail(f"sitemap did not parse as XML: {exc}")
            return []

        urls: list[str] = []
        seen: set[str] = set()
        for loc in root.findall(".//s:loc", _SITEMAP_NS):
            raw = (loc.text or "").strip()
            if not raw or not _COURSE_PATH.match(urlsplit(raw).path):
                continue
            if raw not in seen:
                seen.add(raw)
                urls.append(raw)
        return urls

    def _stub(self, url: str) -> RawCourse:
        """Identity only, built without spending a request — which is the whole
        saving of asking "do we already have this?" before fetching."""
        return RawCourse(
            source=self.name, source_group=self.source_group,
            source_type=self.source_type, source_url=url,
            name=url.rsplit("/", 1)[-1].replace("-", " ")[:300],
            raw_description="",
        )

    def _from_page(self, url: str, html: str) -> RawCourse | None:
        node = _course_node(html)
        if not node:
            return None

        name = _text(node.get("name")) or _h1(html)
        description = _plain(_text(node.get("description")))
        if not name or not description:
            return None

        language = _language(node)
        if language and language != "en":
            # English catalogue only, the same scope choice the Coursera adapter
            # makes: a non-English description extracts poorly against an English
            # prompt, and the /es/ tree would duplicate every row anyway.
            return None

        rating, reviews = _rating(node)
        price = _price(node)
        return RawCourse(
            source=self.name,
            source_group=self.source_group,
            source_type=self.source_type,
            source_url=url,
            name=name[:300],
            raw_description=description[:MAX_DESCRIPTION_CHARS],
            provider=_provider(node),
            level=_level(node),
            primary_language=language,
            duration_text=_duration_text(node),
            rating=rating,
            review_count=reviews,
            price=price,
            # These came from the page we just read, so they were genuinely
            # OBSERVED — which is what stops the store preserving stale values
            # over them, and equally what stops a failed read nulling good ones.
            volatile_observed=True,
            attribution=self.attribution,
            license=None,             # proprietary; we catalogue facts and a link
        )


# ---------------------------------------------------------------------------
# JSON-LD readers. Explicit key reads: a missing key is answered with None, never
# with a plausible-looking default.
# ---------------------------------------------------------------------------
def _course_node(html: str) -> dict[str, Any] | None:
    for block in HTMLParser(html).css('script[type="application/ld+json"]'):
        raw = block.text(strip=True)
        if not raw or "Course" not in raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for node in _flatten(data):
            if node.get("@type") == "Course":
                return node
    return None


def _flatten(data: Any) -> Iterator[dict[str, Any]]:
    """Walk a payload: a bare object, a list, or edX's `@graph` wrapper."""
    if isinstance(data, dict):
        yield data
        graph = data.get("@graph")
        if isinstance(graph, list):
            for item in graph:
                yield from _flatten(item)
    elif isinstance(data, list):
        for item in data:
            yield from _flatten(item)


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _plain(text: str) -> str:
    """edX descriptions carry HTML."""
    if not text:
        return ""
    out = HTMLParser(text).text(separator="\n") if "<" in text else text
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", out)).strip()


def _h1(html: str) -> str:
    node = HTMLParser(html).css_first("h1")
    return node.text(strip=True) if node else ""


def _provider(node: dict[str, Any]) -> str | None:
    """The teaching institution, not "edX" — the same distinction the Coursera
    adapter draws between a partner university and the platform."""
    provider = node.get("provider")
    if isinstance(provider, list):
        provider = provider[0] if provider else None
    if isinstance(provider, dict):
        return _text(provider.get("name")) or None
    return _text(provider) or None


def _language(node: dict[str, Any]) -> str | None:
    value = node.get("inLanguage") or node.get("availableLanguage")
    if isinstance(value, list):
        value = value[0] if value else None
    text = _text(value if isinstance(value, str) else (value or {}).get("name")
                 if isinstance(value, dict) else "")
    return text.split("-")[0].lower()[:2] or None if text else None


def _level(node: dict[str, Any]) -> str | None:
    return _LEVELS.get(_text(node.get("educationalLevel")).lower())


def _rating(node: dict[str, Any]) -> tuple[float | None, int | None]:
    agg = node.get("aggregateRating")
    if not isinstance(agg, dict):
        return None, None
    return _number(agg.get("ratingValue")), _int(agg.get("ratingCount"))


def _duration_text(node: dict[str, Any]) -> str | None:
    """The provider's own statement of length, in words a person reads.

    `timeRequired` is ISO-8601 and edX usually gives WEEKS (`P4W`). Weeks are
    elapsed calendar time, not study effort, so `parse_iso8601` refuses to turn
    them into hours — claiming 4 weeks is 672 hours of work would be wrong by two
    orders of magnitude. What is honest is repeating what they said, so "P4W"
    becomes "4 weeks" and the hours columns stay null.
    """
    raw = _text(node.get("timeRequired"))
    if not raw:
        return None
    weeks = re.fullmatch(r"P(\d+)W", raw.upper())
    if weeks:
        n = int(weeks.group(1))
        return f"{n} week" if n == 1 else f"{n} weeks"
    low, _high = parse_iso8601(raw)
    if low is not None:
        return f"{low:g} hours" if low != 1 else "1 hour"
    return None


def _price(node: dict[str, Any]) -> dict | None:
    """edX lists several offers — typically a free audit track and a paid
    certificate. The PAID one is what a learner pays to complete it, so that is
    the figure; a course with only a free offer is free.
    """
    offers = node.get("offers")
    if isinstance(offers, dict):
        offers = [offers]
    if not isinstance(offers, list) or not offers:
        return None

    priced = [o for o in offers if isinstance(o, dict) and _number(o.get("price")) is not None]
    if priced:
        best = max(priced, key=lambda o: _number(o.get("price")) or 0.0)
        amount = _number(best.get("price"))
        return {"amount": amount,
                "currency": _text(best.get("priceCurrency")) or None,
                "is_free": amount == 0.0}

    categories = {_text(o.get("category")).lower() for o in offers if isinstance(o, dict)}
    if "free" in categories:
        return {"amount": 0.0, "currency": None, "is_free": True}
    # "Partially Free" and friends are not a price. Saying nothing beats guessing.
    return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(",", "").strip())
        except ValueError:
            return None
    return None


def _int(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None
