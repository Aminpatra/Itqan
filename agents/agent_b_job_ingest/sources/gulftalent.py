"""GulfTalent — a second independent publisher, crawled under its own exception.

**Read the terms clause before changing anything here.** GulfTalent's Terms of
Use prohibit crawling, with one exception, quoted in full in `config.py`: we may
crawl "as an internet search engine making the information searchable by users,
and provided you display only minimal snippets… mention the source clearly as
GulfTalent, and link each snippet back to the corresponding page on GulfTalent."

That is the entire legal basis for this file existing, and its three conditions
are enforced in code rather than remembered:

* **minimal snippets** — `api/mapping` never publishes `raw_description`;
* **name the source** — `display_name="GulfTalent"` reaches every job card;
* **link back** — `link_back_required=True` stops `final_url` from redirecting
  the apply link to the employer's own site.

`tests/agent_b/test_attribution_compliance.py` pins all three. If one is ever
removed, this source must be disabled in the same change.

---

**Why the sitemap rather than pagination.** `/oman/jobs` shows 25 of 431 and its
pagination is neither `?page=` nor `?p=` — both were tested and return page 1's
items, which is the failure mode that silently ingests the same 25 rows forever
while reporting success. `sitemap_jx000.xml` lists every job URL, is advertised
in robots.txt, and costs two requests for the whole inventory.

**Why JSON-LD rather than scraping the rendered page.** Every ad carries a
schema.org `JobPosting` block: `hiringOrganization`, `employmentType`,
`baseSalary`, `datePosted`, `validThrough`. These are the publisher's own
structured statements — the same class of fact as a form field, not prose a
model interpreted — so they are authoritative and are deliberately NOT re-checked
by `stated_facts`, which exists to catch a model's assertions. This is also the
first source in the corpus that publishes **salary**.

**Politeness.** robots.txt's `Crawl-delay: 30` entries are scoped to named SEO
crawlers (SemrushBot, AhrefsBot…); the group that applies to us is
`User-agent: *` → `Allow: /` with no delay. We take 3s anyway, matching
Dubizzle's reasoning: a source costing one request per posting earns a slower
cadence than one that answers in a single feed.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Iterator
from urllib.parse import urlsplit
from xml.etree import ElementTree as ET

from selectolax.parser import HTMLParser

from shared.config import Config

from ..hashing import posting_id
from .base import AdapterResult, BaseAdapter, RawPosting
from .el7far import canonical_url
from .http import (Blocked, PoliteClient, ResponseTooLarge, SourcePolicy,
                   build_client, build_robots)
from .robots import RobotsPolicy

MAX_DESCRIPTION_CHARS = 20_000

# A sitemap is a bulk index and is legitimately far bigger than a page.
# MEASURED 2026-08-08: `sitemap_jx000.xml` is **5,106,450 bytes** across 28,978
# URLs — just over the project's 5 MB default, which is what a first live run
# failed on.
#
# 16 MB is not that number plus a margin; it is headroom to the PROTOCOL's own
# ceiling. sitemaps.org caps one sitemap at 50,000 URLs, and at ~176 bytes per
# entry here that is ~8.8 MB at the theoretical maximum — after which the
# publisher must roll to `sitemap_jx001`, which the `sitemap_jx` prefix match
# already picks up. So this cap cannot be outgrown by the catalogue merely
# getting bigger, which a 6 MB cap would have been within a year.
#
# It applies to the ad pages too, since they share the client (one client, one
# rate limiter — two on one host would halve the real interval while every log
# line showed the polite number). That is acceptable: ad pages measure ~200 KB,
# they are fetched one at a time behind a 3-second floor, and the byte cap is a
# runaway guard rather than a statement about expected page size.
SITEMAP_MAX_BYTES = 16 * 1024 * 1024

SITEMAP_PATH = "/sitemap.xml"
_SITEMAP_NS = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}

# The sub-sitemap holding individual job ads. Measured 2026-08-08: `jx000`
# carried 28,947 ad URLs while `jl000` (30,000 URLs) held none — they are
# listing/category pages. The prefix is matched rather than hardcoded in full so
# a roll to `jx001` as the catalogue grows is picked up automatically.
_AD_SITEMAP_HINT = "sitemap_jx"

# A job ad's path: /<country>/jobs/<slug>-<numeric id>. The trailing id is what
# separates an ad from a category page like /oman/jobs/sales.
_AD_PATH = re.compile(r"^/([a-z\-]+)/jobs?/[a-z0-9\-]+-\d{4,}$")

# schema.org employmentType -> our vocabulary (migration 0011's CHECK).
# Anything unlisted stays None: an unmapped value is a question, not a default.
_EMPLOYMENT = {
    "FULL_TIME": "full_time",
    "PART_TIME": "part_time",
    "CONTRACTOR": "contract",
    "TEMPORARY": "temporary",
    "INTERN": "internship",
    "OTHER": None,
    "PER_DIEM": None,
    "VOLUNTEER": None,
}

# schema.org unitText on a QuantitativeValue -> our salary_period vocabulary.
_PERIOD = {"HOUR": "hour", "DAY": "day", "WEEK": "week",
           "MONTH": "month", "YEAR": "year"}


class GulfTalentAdapter(BaseAdapter):
    source_type = "html_scrape"

    def __init__(
        self,
        *,
        name: str = "gulftalent",
        source_group: str = "gulftalent",
        base_url: str = "https://www.gulftalent.com",
        client: PoliteClient | None = None,
        config: Config | None = None,
        robots: RobotsPolicy | None = None,
        is_known_unchanged=None,
        attribution: str = "GulfTalent",
        terms_url: str = "https://www.gulftalent.com/terms",
    ) -> None:
        super().__init__(name=name, source_group=source_group)
        self.config = config or Config()
        self.base_url = base_url.rstrip("/")
        self._client = client
        self._robots = robots
        self._is_known_unchanged = is_known_unchanged
        # Carried onto every row this adapter produces. Per row, not per config
        # entry, so a posting can still say who published it and under what
        # terms long after anyone edits the registry.
        self.attribution = attribution
        self.terms_url = terms_url
        # Ads whose country did not survive verification. Counted rather than
        # silently dropped — see `_fetch`.
        self.out_of_scope = 0

    # ------------------------------------------------------------------
    @property
    def sitemap_url(self) -> str:
        return f"{self.base_url}{SITEMAP_PATH}"

    def _ensure_client(self) -> PoliteClient:
        if self._client is None:
            self._client = build_client(
                source=self.name,
                # One request per posting, like Dubizzle, so the same cadence.
                # robots imposes no delay on us; this is our own restraint.
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
            ad_urls = self._enumerate(client, result)
        except (Blocked, ResponseTooLarge) as exc:
            result.fail(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            result.fail(f"{type(exc).__name__}: {exc}")
            return

        if result.error:
            return

        if not ad_urls:
            # The sitemap parsed but yielded no in-scope ads. That is either a
            # real empty day for Oman (possible) or the URL shape moved (far
            # more likely, and the failure that ages the whole inventory toward
            # deletion). Refusing to call it empty is the conservative side.
            result.fail(
                "no Oman job URLs in the sitemap — the URL shape may have changed; "
                "not treating this as an empty listing"
            )
            return

        for url in ad_urls:
            if limit is not None and len(result.postings) >= limit:
                # A capped fetch is NOT a census: `truncated` stops the staleness
                # pass from ageing everything this run did not reach.
                result.truncated = True
                return

            if not self._robots.can_fetch(url):
                result.skipped += 1
                continue

            # A posting we already hold costs nothing. Without this the source
            # re-fetches 430 ad pages every cycle — 860 requests a day at a
            # 12-hour cadence, against a site that throttles SEO crawlers to one
            # request every 30 seconds.
            #
            # Reporting the id is not optional: a posting the pipeline never
            # sees is never touched, `age_missed` counts it as missing, and the
            # source deletes its own inventory for the crime of being unchanged.
            stub = self._stub(url)
            if self._is_known_unchanged and self._is_known_unchanged(stub):
                result.seen_unchanged_ids.append(posting_id(self.name, url))
                continue

            try:
                html = client.get_text(url)
            except (Blocked, ResponseTooLarge) as exc:
                result.fail(str(exc))
                return
            except Exception:  # noqa: BLE001 - one bad ad must not end the source
                result.skipped += 1
                continue

            result.pages_fetched += 1
            result.bytes_fetched = client.bytes_fetched

            posting = self._from_ad(url, html, result)
            if posting is None:
                result.skipped += 1
                continue
            result.postings.append(posting)

        # Every ad page loaded and not one carried a JobPosting block: the
        # layout moved. Same anchor-miss reasoning every adapter here uses.
        if result.pages_fetched > 0 and not result.postings and result.skipped:
            result.fail(
                f"fetched {result.pages_fetched} ad page(s) and parsed no JobPosting "
                f"JSON-LD — the page structure may have changed"
            )

    # ------------------------------------------------------------------
    def _enumerate(self, client: PoliteClient, result: AdapterResult) -> list[str]:
        """Sitemap index -> ad sitemap -> the in-scope ad URLs.

        Two requests for the whole inventory, from a file robots.txt advertises.
        """
        index = client.get_text(self.sitemap_url)
        result.pages_fetched += 1
        try:
            root = ET.fromstring(index)
        except ET.ParseError as exc:
            result.fail(f"sitemap index did not parse as XML: {exc}")
            return []

        sub = [loc.text or "" for loc in root.findall(".//s:loc", _SITEMAP_NS)]
        ad_maps = [u for u in sub if _AD_SITEMAP_HINT in u]
        if not ad_maps:
            result.fail(
                f"no {_AD_SITEMAP_HINT}* sub-sitemap among {len(sub)} entries — the "
                f"sitemap layout may have changed"
            )
            return []

        urls: list[str] = []
        seen: set[str] = set()
        for sitemap_url in ad_maps:
            if not self._robots.can_fetch(sitemap_url):
                continue
            body = client.get_text(sitemap_url)
            result.pages_fetched += 1
            try:
                tree = ET.fromstring(body)
            except ET.ParseError as exc:
                result.fail(f"{sitemap_url} did not parse as XML: {exc}")
                return []

            for loc in tree.findall(".//s:url/s:loc", _SITEMAP_NS):
                raw = (loc.text or "").strip()
                if not raw or not self._is_in_scope_url(raw):
                    continue
                url = canonical_url(raw)
                if url not in seen:
                    seen.add(url)
                    urls.append(url)
        return urls

    # ---- OMAN ONLY ---------------------------------------------------
    def _is_in_scope_url(self, url: str) -> bool:
        """CHECK 1 of 3: the country lives in the URL's own path segment.

        GulfTalent is Gulf-wide — 28,947 ads, 430 of them Oman. A filter bug here
        does not degrade the corpus, it REPLACES it: 28,517 UAE/Saudi/Qatar
        postings would swamp `skill_demand_stats` and every gap score Agent C
        publishes would describe a labour market our users do not live in.

        Matching the path segment, NOT `'oman' in url` — the latter also matches
        a Dubai ad slugged `omani-driver-wanted-12345`, which is exactly the kind
        of near-miss that passes review and fails in production.
        """
        match = _AD_PATH.match(urlsplit(url).path)
        return bool(match) and match.group(1) in self._countries()

    def _countries(self) -> set[str]:
        """URL-path spellings of the countries we ingest.

        Driven by `in_scope_countries` rather than hardcoded, so the ISO code
        that governs aggregation and the path segment that governs the crawl can
        never drift apart.
        """
        names = {"OM": "oman", "AE": "uae", "SA": "saudi-arabia",
                 "QA": "qatar", "BH": "bahrain", "KW": "kuwait"}
        return {names[c] for c in self.config.in_scope_countries if c in names}

    # ------------------------------------------------------------------
    def _stub(self, url: str) -> RawPosting:
        """Identity only, built without spending a request.

        `is_known_unchanged` needs a posting-shaped object to derive an id from,
        and that id comes from (source, source_url) — so a stub with a
        placeholder title answers "do we already have this?" perfectly well, and
        answering it before the fetch is the entire saving.
        """
        return RawPosting(
            source=self.name,
            source_group=self.source_group,
            source_type=self.source_type,
            source_url=url,
            title=url.rsplit("/", 1)[-1].replace("-", " ")[:300],
            raw_description="",
            attribution=self.attribution,
            terms_url=self.terms_url,
        )

    def _from_ad(self, url: str, html: str, result: AdapterResult) -> RawPosting | None:
        data = _job_posting_ld(html)
        if not data:
            return None

        title = _text(data.get("title")) or _h1(html)
        description = _plain(_text(data.get("description")))
        if not title or not description:
            return None

        country, location = _location_of(data)

        # CHECK 2 of 3: the page's own statement of where the job is. A URL
        # saying Oman and a payload saying UAE means our filter is wrong or we
        # followed a redirect — either way the row is not wanted, and counting it
        # makes the disagreement visible instead of silent.
        if country and country not in self.config.in_scope_countries:
            self.out_of_scope += 1
            return None

        # Fall back to the publisher's own filing when the payload omits it. The
        # URL segment is how GulfTalent itself classifies the ad, and it is the
        # evidence we admitted the row on — declining to record it would leave
        # `country` NULL, and `export_for_agent_c` filters on country, so the
        # row would be stored and never retrieved. Measured: 21 of 28 rows in
        # the first live cycle had a NULL country for exactly this reason.
        country = country or _country_from_url(url)

        salary_min, salary_max, currency, period = _salary_of(data)
        employer = _organisation(data)

        return RawPosting(
            source=self.name,
            source_group=self.source_group,
            source_type=self.source_type,
            source_url=url,
            title=title[:300],
            raw_description=description[:MAX_DESCRIPTION_CHARS],
            company=employer,
            location_text=location,
            country=country,
            posted_date=_iso_datetime(data.get("datePosted")),
            expires_at=_iso_datetime(data.get("validThrough")),
            # The publisher's own structured statements. Authoritative, and
            # deliberately not re-verified against the prose — see the module
            # docstring.
            employment_type=_employment_of(data),
            salary_min=salary_min,
            salary_max=salary_max,
            salary_currency=currency,
            salary_period=period,
            # A board for employers advertising vacancies: the listing intent is
            # a property of the site, not a guess about this ad.
            listing_intent="vacancy",
            # 'company' ONLY when the publisher named an employer in the
            # machine-readable `hiringOrganization` property. That is a stated
            # fact, and a stronger one than this system's usual test — the
            # pipeline's grounded-company check exists because a model may
            # invent an employer from prose, and nothing was invented here.
            #
            # This is load-bearing, not a nicety. `poster_type = 'unknown'` is
            # ineligible for aggregation under migration 0006, so leaving it
            # unknown would mean every row from this source is STORED AND NEVER
            # COUNTED — precisely the trap dubizzle sits in. An ad with no
            # hiringOrganization stays unknown and is decided downstream.
            poster_type="company" if employer else "unknown",
            attribution=self.attribution,
            terms_url=self.terms_url,
            outbound_links=(),
        )


# ---------------------------------------------------------------------------
# JSON-LD readers. Explicit key reads throughout: a KeyError or a None is a
# question we answer with None, never with a plausible-looking default.
# ---------------------------------------------------------------------------
def _job_posting_ld(html: str) -> dict[str, Any] | None:
    """The page's schema.org JobPosting block, if it has one."""
    for node in HTMLParser(html).css('script[type="application/ld+json"]'):
        raw = node.text(strip=True)
        if not raw or "JobPosting" not in raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for candidate in _flatten(data):
            if candidate.get("@type") == "JobPosting":
                return candidate
    return None


def _flatten(data: Any) -> Iterator[dict[str, Any]]:
    """Walk a JSON-LD payload: a bare object, a list, or an @graph wrapper."""
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


def _plain(html_or_text: str) -> str:
    """JobPosting.description is HTML by specification."""
    if not html_or_text:
        return ""
    text = HTMLParser(html_or_text).text(separator="\n") if "<" in html_or_text else html_or_text
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text)).strip()


def _h1(html: str) -> str:
    node = HTMLParser(html).css_first("h1")
    return node.text(strip=True) if node else ""


def _organisation(data: dict[str, Any]) -> str | None:
    org = data.get("hiringOrganization")
    if isinstance(org, dict):
        return _text(org.get("name")) or None
    return _text(org) or None


def _location_of(data: dict[str, Any]) -> tuple[str | None, str | None]:
    """(ISO country, human location) — either may be None."""
    place = data.get("jobLocation")
    if isinstance(place, list):
        place = place[0] if place else None
    if not isinstance(place, dict):
        return None, None
    address = place.get("address")
    if not isinstance(address, dict):
        return None, _text(place.get("name")) or None

    country = _text(address.get("addressCountry"))
    # schema.org permits a nested Country object as well as a bare string.
    if not country and isinstance(address.get("addressCountry"), dict):
        country = _text(address["addressCountry"].get("name"))

    locality = _text(address.get("addressLocality"))
    region = _text(address.get("addressRegion"))
    human = ", ".join(dict.fromkeys([p for p in (locality, region) if p])) or None
    return (_iso_country(country), human)


# Only the spellings this source actually uses. An unrecognised country returns
# None, which the caller treats as "unstated" rather than "in scope" — the
# conservative direction for a filter whose failure mode is importing 28,517
# postings from the wrong countries.
_COUNTRIES = {
    "om": "OM", "oman": "OM", "sultanate of oman": "OM",
    "ae": "AE", "uae": "AE", "united arab emirates": "AE",
    "sa": "SA", "saudi arabia": "SA", "ksa": "SA",
    "qa": "QA", "qatar": "QA",
    "bh": "BH", "bahrain": "BH",
    "kw": "KW", "kuwait": "KW",
}


def _country_from_url(url: str) -> str | None:
    """The country GulfTalent itself filed the ad under, from the URL path.

    Not a guess: `/oman/jobs/...` is the publisher's own classification, and it
    is the evidence the ad was admitted on in the first place.
    """
    match = _AD_PATH.match(urlsplit(url).path)
    return _iso_country(match.group(1).replace("-", " ")) if match else None


def _iso_country(value: str) -> str | None:
    return _COUNTRIES.get(value.strip().casefold()) if value else None


def _employment_of(data: dict[str, Any]) -> str | None:
    value = data.get("employmentType")
    if isinstance(value, list):
        value = value[0] if value else None
    if not isinstance(value, str):
        return None
    return _EMPLOYMENT.get(value.strip().upper().replace("-", "_"))


def _salary_of(data: dict[str, Any]) -> tuple[float | None, float | None, str | None, str | None]:
    """schema.org MonetaryAmount -> (min, max, currency, period).

    A single `value` (not a range) is recorded as BOTH ends: the publisher stated
    one figure, and reporting it as an open-ended range would widen a claim they
    did not make.
    """
    salary = data.get("baseSalary")
    if not isinstance(salary, dict):
        return None, None, None, None

    amount = salary.get("value")
    currency = _text(salary.get("currency")) or _text(data.get("salaryCurrency")) or None

    if isinstance(amount, (int, float)):
        return float(amount), float(amount), currency, None
    if not isinstance(amount, dict):
        return None, None, None, None

    low = _number(amount.get("minValue"))
    high = _number(amount.get("maxValue"))
    if low is None and high is None:
        single = _number(amount.get("value"))
        low = high = single
    period = _PERIOD.get(_text(amount.get("unitText")).upper())
    # A backwards range is an error at the source, not a salary. Migration
    # 0011's CHECK would reject the row and lose the whole posting with it.
    if low is not None and high is not None and high < low:
        return None, None, None, None
    return low, high, currency, period


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _iso_datetime(value: Any) -> datetime | None:
    text = _text(value)
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
