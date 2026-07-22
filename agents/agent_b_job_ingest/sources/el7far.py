"""Blogger Atom adapter for oman.el7far.com.

Blogger publishes the same posts under two paths, and only one is ours to use:
per-label HTML lives at ``/search/label/{x}``, which the site's robots.txt
**disallows** via ``Disallow: /search``; per-label Atom lives at
``/feeds/posts/default/-/{x}``, which it allows. The feed is therefore not
merely the tidier option — it is the only compliant one. Asserted in a test
against the committed robots.txt rather than left as a comment here.

Parsing is stdlib ``xml.etree.ElementTree``, not ``feedparser``. feedparser
earns its place by tolerating malformed RSS across the open web; here there is
one known generator emitting verified well-formed Atom, and tolerance is the
wrong instinct — a feed that changed shape should fail loudly, not half-parse
into postings with silently missing fields. ElementTree raises, and ``fetch``
converts that into an errored ``AdapterResult``.

Entity-expansion risk is handled upstream by ``PoliteClient``'s streaming
``max_bytes`` cap, which refuses an oversized document before the parser is ever
handed it — so no third-party XML hardening dependency is needed.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit, urlunsplit
from xml.etree import ElementTree as ET

from selectolax.parser import HTMLParser

from shared.config import Config

from .base import AdapterResult, BaseAdapter, RawPosting
from .http import Blocked, PoliteClient, ResponseTooLarge, SourcePolicy
from .robots import RobotsPolicy

ATOM = "http://www.w3.org/2005/Atom"
NS = {"a": ATOM}

FEED_PATH = "/feeds/posts/default"
PAGE_SIZE = 25

# Entries carry the full rendered post: ~40KB of HTML each, much of it markup,
# navigation and SEO boilerplate. Truncation is deterministic, so it does not
# destabilise content_hash across cycles.
MAX_DESCRIPTION_CHARS = 20_000

# Blogger resurfaces edited old posts at the top of the feed, so ONE known id is
# not evidence that the rest of the page is known. Ten consecutive is.
KNOWN_RUN_TO_STOP = 10


class El7farAdapter(BaseAdapter):
    source_type = "blogger_feed"

    def __init__(
        self,
        *,
        name: str = "el7far",
        source_group: str = "el7far_network",
        base_url: str = "https://oman.el7far.com",
        client: PoliteClient | None = None,
        config: Config | None = None,
        robots: RobotsPolicy | None = None,
        is_known_unchanged=None,
    ) -> None:
        super().__init__(name=name, source_group=source_group)
        self.config = config or Config()
        self.base_url = base_url.rstrip("/")
        self._client = client
        self._owns_client = client is None
        self._robots = robots
        # Injected, not imported: an adapter that could ask the database
        # questions is one step from an adapter that writes to it. Absent, every
        # entry is treated as new, which is correct for a dry run.
        self._is_known_unchanged = is_known_unchanged

    # ------------------------------------------------------------------
    @property
    def feed_url(self) -> str:
        return f"{self.base_url}{FEED_PATH}"

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
        self._robots.require(self.feed_url)

        cutoff = datetime.now(timezone.utc) - timedelta(days=self.config.blogger_lookback_days)
        known_run = 0
        seen_ids: set[str] = set()

        for page in range(self.config.blogger_max_pages):
            start = page * PAGE_SIZE + 1
            params = {
                "alt": "atom",
                "max-results": str(PAGE_SIZE),
                "start-index": str(start),
                # Verified server-side: a window with no posts returns zero
                # entries rather than falling back to the newest ones.
                "published-min": cutoff.isoformat(),
            }

            try:
                body = client.get_text(self.feed_url, params=params)
            except (Blocked, ResponseTooLarge) as exc:
                result.fail(str(exc))
                return
            except Exception as exc:  # noqa: BLE001 - network shapes vary; all are partial
                result.fail(f"{type(exc).__name__}: {exc}")
                return

            result.pages_fetched += 1
            result.bytes_fetched = client.bytes_fetched

            try:
                root = ET.fromstring(body)
            except ET.ParseError as exc:
                # A shape change must not half-parse into postings with quietly
                # missing fields.
                result.fail(f"feed did not parse as XML: {exc}")
                return

            entries = root.findall("a:entry", NS)
            if not entries:
                return

            for entry in entries:
                posting = self._parse_entry(entry)
                if posting is None:
                    result.skipped += 1
                    continue
                if posting.source_url in seen_ids:
                    continue
                seen_ids.add(posting.source_url)

                if posting.posted_date and posting.posted_date < cutoff:
                    return

                if self._is_known_unchanged and self._is_known_unchanged(posting):
                    known_run += 1
                    if known_run >= KNOWN_RUN_TO_STOP:
                        return
                    continue
                known_run = 0

                result.postings.append(posting)
                if limit is not None and len(result.postings) >= limit:
                    return

            if len(entries) < PAGE_SIZE:
                return

    # ------------------------------------------------------------------
    def _parse_entry(self, entry: ET.Element) -> RawPosting | None:
        url = _alternate_link(entry)
        if not url:
            return None

        title = (entry.findtext("a:title", "", NS) or "").strip()
        content_el = entry.find("a:content", NS)
        html = (content_el.text or "") if content_el is not None else ""
        canonical = canonical_url(url)
        text, links = _html_to_text_and_links(html, base_url=url, self_url=canonical)

        if not title and not text:
            return None

        # DELIBERATELY NOT SET: company.
        #
        # atom:author/name on this feed is the blogger — a live fetch returns a
        # personal name, not an employer. Mapping it to `company` would invent an
        # employer that does not exist AND suppress the `no_employer_named`
        # legitimacy signal, converting a fabrication into a scam-filter bypass.
        # The employer is extracted from the posting body downstream, or stays
        # None. This is the single most consequential line in the adapter.

        return RawPosting(
            source=self.name,
            source_group=self.source_group,
            source_type=self.source_type,
            source_url=canonical,
            title=title,
            raw_description=text[:MAX_DESCRIPTION_CHARS],
            company=None,
            location_text=None,
            # A real timestamp from the source, so there is no relative-date
            # string to re-resolve on each fetch.
            posted_date=_parse_published(entry.findtext("a:published", "", NS)),
            posted_date_text=None,
            labels=tuple(
                term
                for cat in entry.findall("a:category", NS)
                if (term := (cat.get("term") or "").strip())
            ),
            outbound_links=links,
        )


# ---------------------------------------------------------------------------
def _alternate_link(entry: ET.Element) -> str:
    """The post's own page.

    Blogger emits several ``<link>`` elements per entry — ``replies``, ``edit``,
    ``self`` and ``alternate`` — and two of them point at text/html. Taking the
    first html link would yield the *comments* feed page for the post. Only
    rel="alternate" is the posting.
    """
    for link in entry.findall("a:link", NS):
        if link.get("rel") == "alternate":
            return (link.get("href") or "").strip()
    return ""


def _parse_published(value: str | None) -> datetime | None:
    """RFC3339 with milliseconds and an offset, e.g. 2026-07-21T10:18:24.984+03:00."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    # Normalised to UTC so comparisons against the cutoff and against other
    # sources' timestamps cannot silently mix offsets.
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _html_to_text_and_links(
    html: str, *, base_url: str, self_url: str = ""
) -> tuple[str, tuple[str, ...]]:
    """Rendered post HTML to readable text plus the URLs it points at.

    The links are not decoration: a posting that links elsewhere is how the
    deterministic dedup path resolves a Telegram repost to its blog original,
    without relying on embedding similarity.

    ``self_url`` is excluded. Live posts link to themselves (a canonical tag
    rendered into the body, a "read more" anchor), and phase 4's rule — the
    linker is the duplicate, the target is canonical — would make such a posting
    its own duplicate. That is not merely wrong: the schema's
    ``CHECK (duplicate_of IS DISTINCT FROM posting_id)`` would abort the
    transaction part-way through an unattended cycle.
    """
    if not html.strip():
        return "", ()

    tree = HTMLParser(html)
    # decompose(), NOT unwrap_tags(): unwrapping removes the tag and KEEPS its
    # text, which is the exact opposite of what is wanted here — script bodies
    # would survive into raw_description and then be embedded and extracted as
    # though they were part of the job posting.
    for node in tree.css("script, style"):
        node.decompose()

    text = tree.text(separator="\n", strip=True) or ""
    text = "\n".join(line for line in (ln.strip() for ln in text.splitlines()) if line)

    links: list[str] = []
    seen: set[str] = set()
    for node in tree.css("a[href]"):
        href = (node.attributes.get("href") or "").strip()
        if not href or href.startswith(("#", "mailto:", "javascript:", "tel:")):
            continue
        absolute = canonical_url(href if "//" in href else _join(base_url, href))
        if is_asset_link(absolute) or node.css_first("img") is not None:
            continue
        if self_url and absolute == self_url:
            continue
        if absolute not in seen:
            seen.add(absolute)
            links.append(absolute)
    return text, tuple(links)


# Blogger wraps every inline image in an anchor to the full-size file, so a live
# post's first anchor is usually its own photo rather than anything it refers
# to. Left in, those would sit at the head of `outbound_links` and the phase-4
# dedup rule — "prefer the first link" — would be reading a CDN path instead of
# a posting URL.
_ASSET_SUFFIXES = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".bmp", ".ico", ".pdf")


def is_asset_link(url: str) -> bool:
    """An anchor pointing at a file, not at another page.

    Only half the test. Blogger's CDN serves extensionless paths
    (``/img/a/AVvXsEi…``), so the live gate run showed image links surviving
    this check — which is why the caller ALSO rejects any anchor that wraps an
    ``<img>``. That structural test is the reliable one; this handles bare
    file links with no image element to look at.

    Matched on the URL's shape rather than on a list of CDN hostnames, which
    would need updating whenever a source changed image host and would fail
    silently when it did.
    """
    return urlsplit(url).path.casefold().endswith(_ASSET_SUFFIXES)


def _join(base: str, path: str) -> str:
    parts = urlsplit(base)
    if not path.startswith("/"):
        path = "/" + path
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def canonical_url(url: str) -> str:
    """Strip the parts that vary without changing the content.

    Query strings and fragments on a Blogger permalink are tracking and anchors;
    keeping them would make one post look like several and inflate every
    frequency count derived from it.
    """
    parts = urlsplit(url.strip())
    scheme = "https" if parts.scheme in ("", "http") else parts.scheme
    netloc = parts.netloc.lower()
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((scheme, netloc, path, "", ""))
