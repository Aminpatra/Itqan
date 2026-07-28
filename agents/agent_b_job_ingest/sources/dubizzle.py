"""Dubizzle Oman — the ``html_scrape`` reference adapter.

Two structural facts about this source shape everything below, and both were
found by inspection rather than assumed.

**1. It is a classifieds site, not a vacancy board.** A large share of its job
listings are people advertising *themselves*, and they are not confined to the
"Jobs Wanted" category — all four of the first four listings in the IT-Telecom
*vacancy* category were seekers. Counting those as postings measures supply and
publishes it as demand. So this adapter records ``listing_intent`` only where
the publisher's own category states it, and leaves everything else ``unknown``;
migration 0006 makes ``unknown`` ineligible for aggregation.

**2. It exposes no poster identity at all.** No seller name, no company name, no
business-account badge — an anonymous user photo is the only poster element on
the page. ``poster_type`` therefore stays ``unknown`` for every row, which under
the same migration means nothing from this source aggregates. That is the
intended outcome. Inferring "company" from the wording of an ad would put a
guess in the column that exists to keep guesses out.

**Selectors use ``aria-label`` and semantic elements, never CSS classes.** The
live markup's classes are build hashes (``c0b6be79``, ``_948d9e0a``) that change
on every frontend deploy; a class-based selector would break silently and the
adapter would report zero postings as though the site had gone quiet.
"""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

from selectolax.parser import HTMLParser, Node

from shared.config import Config

from .base import AdapterResult, BaseAdapter, RawPosting
from .el7far import canonical_url
from .http import Blocked, PoliteClient, ResponseTooLarge, SourcePolicy
from .robots import RobotsPolicy

MAX_DESCRIPTION_CHARS = 20_000

# Category paths, from the site's own navigation. `jobs-wanted` is listed but
# deliberately NOT fetched: it is wholly job-seekers, so every row would be
# ineligible and fetching it would spend request budget to store nothing.
LISTING_PATH = "/en/jobs-services/"
SEEKING_CATEGORY = "jobs-wanted"


class DubizzleAdapter(BaseAdapter):
    source_type = "html_scrape"

    def __init__(
        self,
        *,
        name: str = "dubizzle",
        source_group: str = "dubizzle",
        base_url: str = "https://www.dubizzle.com.om",
        categories: tuple[str, ...] = (),
        client: PoliteClient | None = None,
        config: Config | None = None,
        robots: RobotsPolicy | None = None,
        is_known_unchanged=None,
    ) -> None:
        super().__init__(name=name, source_group=source_group)
        self.config = config or Config()
        self.base_url = base_url.rstrip("/")
        self.categories = categories
        self._client = client
        self._robots = robots
        self._is_known_unchanged = is_known_unchanged

    @property
    def listing_url(self) -> str:
        return f"{self.base_url}{LISTING_PATH}"

    def _ensure_client(self) -> PoliteClient:
        if self._client is None:
            self._client = PoliteClient(
                source=self.name,
                # Slower than the feed: this source needs one request per
                # listing PLUS one per detail page, so the same page budget
                # costs it far more requests.
                policy=SourcePolicy(min_interval_s=3.0, max_bytes=self.config.max_response_bytes),
                config=self.config,
            )
        if self._robots is None:
            self._robots = RobotsPolicy(self._client, user_agent=self.config.user_agent)
        return self._client

    # ------------------------------------------------------------------
    def _fetch(self, result: AdapterResult, *, limit: int | None) -> None:
        client = self._ensure_client()
        assert self._robots is not None

        pages = [self.listing_url] + [
            f"{self.base_url}{LISTING_PATH}{c}/" for c in self.categories if c != SEEKING_CATEGORY
        ]

        seen: set[str] = set()
        for page_url in pages:
            decision = self._robots.can_fetch(page_url)
            if not decision.allowed:
                result.fail(f"robots.txt refuses {page_url} ({decision.reason})")
                return

            try:
                html = client.get_text(page_url)
            except (Blocked, ResponseTooLarge) as exc:
                result.fail(str(exc))
                return
            except Exception as exc:  # noqa: BLE001
                result.fail(f"{type(exc).__name__}: {exc}")
                return

            result.pages_fetched += 1
            result.bytes_fetched = client.bytes_fetched

            cards = HTMLParser(html).css("article")
            if not cards:
                # A listing page always renders cards; zero means the markup moved,
                # not that Dubizzle has no jobs. Reporting that as a clean empty
                # fetch would age this source's whole inventory toward deletion.
                result.fail(
                    f"no <article> cards on {page_url} — the listing markup may have "
                    f"changed; not treating this as an empty listing"
                )
                return

            for card in cards:
                href = _ad_href(card)
                if not href:
                    result.skipped += 1
                    continue

                url = canonical_url(_absolute(self.base_url, href))
                if url in seen:
                    continue
                seen.add(url)

                stub = _from_card(card, url=url, source=self)
                if stub is None:
                    result.skipped += 1
                    continue
                if self._is_known_unchanged and self._is_known_unchanged(stub):
                    continue

                posting = self._with_detail(stub, client)
                if posting is None:
                    result.skipped += 1
                    continue

                result.postings.append(posting)
                result.bytes_fetched = client.bytes_fetched
                if limit is not None and len(result.postings) >= limit:
                    return

    # ------------------------------------------------------------------
    def _with_detail(self, stub: RawPosting, client: PoliteClient) -> RawPosting | None:
        """Fetch the ad page for its description.

        A listing card carries a title, price and location but no body text, and
        a posting with no description cannot be extracted from or scored — the
        legitimacy rules would see an empty string and report it as clean.
        Returns None rather than storing a title-only row.
        """
        assert self._robots is not None
        if not self._robots.can_fetch(stub.source_url):
            return None

        try:
            html = client.get_text(stub.source_url)
        except (Blocked, ResponseTooLarge):
            raise
        except Exception:  # noqa: BLE001 - one bad ad must not end the source
            return None

        tree = HTMLParser(html)
        description = _labelled_text(tree, "Description")
        if not description:
            return None

        heading = tree.css_first("h1")
        title = heading.text(strip=True) if heading else stub.title
        breadcrumb = _labelled_text(tree, "Breadcrumb")

        return RawPosting(
            source=stub.source,
            source_group=stub.source_group,
            source_type=stub.source_type,
            source_url=stub.source_url,
            title=title or stub.title,
            raw_description=description[:MAX_DESCRIPTION_CHARS],
            # No employer is stated anywhere on the page. Same rule as every
            # other adapter: absent means None, never approximated.
            company=None,
            location_text=stub.location_text,
            posted_date=None,
            posted_date_text=stub.posted_date_text,
            listing_intent=_intent_from(breadcrumb, stub.listing_intent),
            # Nothing on the page identifies the poster, so this cannot be
            # anything but unknown — which is what keeps the source out of
            # aggregation.
            poster_type="unknown",
            labels=stub.labels,
            outbound_links=(),
        )


# ---------------------------------------------------------------------------
def _ad_href(card: Node) -> str:
    link = card.css_first("a[href*='/ad/']")
    return (link.attributes.get("href") or "").strip() if link else ""


def _from_card(card: Node, *, url: str, source: DubizzleAdapter) -> RawPosting | None:
    """What the listing card states, before the detail page is fetched.

    Built first so a card that is already known and unchanged can be skipped
    WITHOUT spending a detail request on it. On a source costing two requests
    per posting, that is the difference between an incremental cycle and a full
    re-crawl every twelve hours.
    """
    title = _labelled_text(card, "Title") or _first_line(card)
    if not title:
        return None

    return RawPosting(
        source=source.name,
        source_group=source.source_group,
        source_type=source.source_type,
        source_url=url,
        title=title[:300],
        # Placeholder only; _with_detail replaces it or drops the posting.
        raw_description=title,
        location_text=_labelled_text(card, "Location") or None,
        # "2 days ago", "12 minutes ago" — never parsed into posted_date here.
        # The same string means a different day on every fetch, so resolving it
        # per cycle would move a posting's date every time it is seen.
        posted_date=None,
        posted_date_text=_labelled_text(card, "Creation date") or None,
    )


def _intent_from(breadcrumb: str, default: str) -> str:
    """Read intent from the publisher's own category, or leave it unknown.

    Only the "Jobs Wanted" category is a STATED claim that the poster is seeking
    work. Everything else stays unknown — deliberately, because the sample
    showed seekers throughout the vacancy categories too, so treating a vacancy
    category as evidence of a vacancy would be wrong in exactly the cases that
    matter.
    """
    if "jobs wanted" in (breadcrumb or "").casefold():
        return "seeking"
    return default


def _labelled_text(node: Node | HTMLParser, label: str) -> str:
    """Text of the first element carrying an aria-label.

    aria-labels are the only stable hooks on this site — the CSS classes are
    build hashes that change on every deploy.
    """
    found = node.css_first(f"[aria-label='{label}']")
    if found is None:
        return ""
    text = found.text(separator=" ", strip=True) or ""
    # The description block renders its own heading inside itself, and the
    # separator does not apply because heading and body share one text node —
    # so the split is on whitespace generally, not on a single space.
    stripped = _strip_heading(text, label)
    return stripped.strip()


def _strip_heading(text: str, label: str) -> str:
    if not text.startswith(label):
        return text
    remainder = text[len(label) :]
    return remainder if remainder[:1].isspace() else text


def _first_line(card: Node) -> str:
    text = card.text(separator="\n", strip=True) or ""
    for line in text.splitlines():
        cleaned = line.strip()
        if cleaned and cleaned.casefold() != "featured":
            return cleaned
    return ""


def _absolute(base: str, href: str) -> str:
    if "//" in href:
        return href
    parts = urlsplit(base)
    return urlunsplit((parts.scheme, parts.netloc, href if href.startswith("/") else f"/{href}", "", ""))
