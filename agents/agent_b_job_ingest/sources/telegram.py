"""Telegram adapter, reading the public logged-out preview at ``t.me/s/{channel}``.

``t.me`` serves no robots.txt — a live check returns 404, which the standard
reads as "no restrictions". **That is not consent**, and it is treated here as a
reason for more restraint rather than less: ``min_interval_s=5.0`` (the slowest
policy in the project), an identifying user agent, public channels only, no
MTProto and no login. The surface used is the one Telegram itself publishes for
logged-out viewing.

The remaining control is in ``config.py``: a channel stays unfetchable until a
human sets ``terms_reviewed=True``. This adapter is generic — pointing it at a
new channel is a one-line change — so the gate exists precisely because adding a
source is too easy to leave to good intentions.

**View counts must never enter ``raw_description``.** They change every cycle, so
including them would change the content hash of every posting on every cycle,
forcing a permanent full re-extraction and re-embedding of the entire channel —
LLM and embedding spend that grows forever and produces no new information.
Scoping the text to ``div.tgme_widget_message_text`` excludes them structurally
(the views span is a sibling in the footer, confirmed against live HTML), and a
test asserts it rather than trusting the selector to stay that way.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from selectolax.parser import HTMLParser, Node

from shared.config import Config

from .base import AdapterResult, BaseAdapter, RawPosting
from .el7far import canonical_url
from .http import (Blocked, PoliteClient, ResponseTooLarge, SourcePolicy,
                   build_client, build_robots)
from .robots import RobotsPolicy

BASE = "https://t.me"
MAX_DESCRIPTION_CHARS = 20_000

MSG_SELECTOR = "div.tgme_widget_message[data-post]"
TEXT_SELECTOR = "div.tgme_widget_message_text"
# Never read for content. Named so the exclusion is greppable and the reason
# travels with the name.
VIEWS_SELECTOR = "span.tgme_widget_message_views"

_POST_ID_RE = re.compile(r"^(?P<channel>[A-Za-z0-9_]+)/(?P<id>\d+)$")


class TelegramAdapter(BaseAdapter):
    source_type = "telegram"

    def __init__(
        self,
        *,
        name: str,
        handle: str,
        source_group: str,
        client: PoliteClient | None = None,
        config: Config | None = None,
        robots: RobotsPolicy | None = None,
        is_known_unchanged=None,
    ) -> None:
        super().__init__(name=name, source_group=source_group)
        from .config import normalize_handle

        self.config = config or Config()
        self.handle = normalize_handle(handle)
        self._client = client
        self._robots = robots
        self._is_known_unchanged = is_known_unchanged

    @property
    def channel_url(self) -> str:
        return f"{BASE}/s/{self.handle}"

    def _ensure_client(self) -> PoliteClient:
        if self._client is None:
            self._client = build_client(
                source=self.name,
                policy=SourcePolicy.for_telegram(),
                config=self.config,
            )
        if self._robots is None:
            self._robots = build_robots(source=self.name, config=self.config)
        return self._client

    # ------------------------------------------------------------------
    def _fetch(self, result: AdapterResult, *, limit: int | None) -> None:
        client = self._ensure_client()
        assert self._robots is not None
        self._robots.require(self.channel_url)

        before: str | None = None
        known_run = 0
        seen: set[str] = set()

        for page in range(self.config.telegram_max_pages):
            params = {"before": before} if before else None
            try:
                body = client.get_text(self.channel_url, params=params)
            except (Blocked, ResponseTooLarge) as exc:
                result.fail(str(exc))
                return
            except Exception as exc:  # noqa: BLE001
                result.fail(f"{type(exc).__name__}: {exc}")
                return

            result.pages_fetched += 1
            result.bytes_fetched = client.bytes_fetched

            tree = HTMLParser(body)
            messages = tree.css(MSG_SELECTOR)
            if not messages:
                # A channel with no messages left to page through returns an empty
                # page too — but only AFTER we have already read some. Finding
                # nothing on the very first request means the selector no longer
                # matches, and calling that "an empty day" ages (and eventually
                # deletes) the whole inventory on a site redesign.
                if page == 0:
                    result.fail(
                        f"no messages matched {MSG_SELECTOR!r} on the first page — the "
                        f"channel layout may have changed; not treating this as empty"
                    )
                return

            oldest: str | None = None
            fresh = 0
            for node in messages:
                data_post = (node.attributes.get("data-post") or "").strip()
                match = _POST_ID_RE.match(data_post)
                if not match:
                    result.skipped += 1
                    continue

                post_id = match.group("id")
                oldest = post_id if oldest is None else min(oldest, post_id, key=int)

                # Identity is checked BEFORE parsing, so a message that appears
                # on two pages is not counted twice. Counting it per appearance
                # would make `skipped` a function of pagination rather than of
                # the channel — and skipped is a health metric, so an inflated
                # number would read as a source going wrong.
                if post_id in seen:
                    continue
                seen.add(post_id)
                fresh += 1

                posting = self._parse_message(node, post_id)
                if posting is None:
                    # Photo-only or sticker posts carry no text. Counted, never
                    # OCR'd — running OCR on a channel's images would be a
                    # different and much more invasive kind of scraping.
                    result.skipped += 1
                    continue

                if self._is_known_unchanged and self._is_known_unchanged(posting):
                    known_run += 1
                    continue
                known_run = 0

                result.postings.append(posting)
                if limit is not None and len(result.postings) >= limit:
                    return

            # A page carrying nothing new means pagination is not advancing —
            # the cursor is being ignored, or the channel has been exhausted.
            # Either way the next request re-reads this page, so stop rather
            # than spending the page budget on it.
            if oldest is None or fresh == 0 or oldest == before:
                return
            before = oldest

    # ------------------------------------------------------------------
    def _parse_message(self, node: Node, post_id: str) -> RawPosting | None:
        text_node = node.css_first(TEXT_SELECTOR)
        if text_node is None:
            return None

        text = text_node.text(separator="\n", strip=True) or ""
        text = "\n".join(line for line in (ln.strip() for ln in text.splitlines()) if line)
        if not text:
            return None

        title = _select_title(text)

        return RawPosting(
            source=self.name,
            source_group=self.source_group,
            source_type=self.source_type,
            # The canonical permalink, not the /s/ preview path, so a link from
            # anywhere else to this post canonicalizes to the same string.
            source_url=f"{BASE}/{self.handle}/{post_id}",
            title=title,
            raw_description=text[:MAX_DESCRIPTION_CHARS],
            # Not set, for the same reason as the Atom adapter: the channel name
            # is the publisher, not the employer. Filling it would fabricate an
            # employer and suppress the no_employer_named legitimacy signal.
            company=None,
            location_text=None,
            posted_date=_parse_time(node),
            posted_date_text=None,
            labels=(),
            outbound_links=_links_in(text_node),
        )


# ---------------------------------------------------------------------------
MIN_TITLE_CHARS = 25


def _select_title(text: str) -> str:
    """Telegram posts have no title field, so one is SELECTED from the text.

    Selected, never composed: whatever is returned appears verbatim in the post,
    and the full text stays in ``raw_description`` so nothing is lost.

    Not simply the first line. Live posts on the verified channel open with an
    internal reference code (a short token of digits, ``#`` and dashes), and
    using it would fill the title column with codes — unreadable in the run log
    and useless to anything downstream that matches on titles. So the first
    *substantive* line wins, with the first line as the fallback when every line
    is short: returning something the source wrote beats returning nothing.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    for line in lines:
        if len(line.strip()) >= MIN_TITLE_CHARS:
            return line.strip()[:300]
    return lines[0].strip()[:300]


def _parse_time(node: Node) -> datetime | None:
    """``<time datetime="2026-07-19T11:01:31+00:00">`` — a real timestamp, so
    there is no relative-date text to re-resolve on every fetch."""
    time_node = node.css_first("time[datetime]")
    if time_node is None:
        return None
    raw = (time_node.attributes.get("datetime") or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _links_in(text_node: Node) -> tuple[str, ...]:
    """Outbound links, in document order.

    **Order is load-bearing and must be preserved.** Live posts on the verified
    channel carry TWO blog links: the job the post is about, followed by an
    unrelated next job. So "this post links to X, therefore it duplicates X" is
    wrong for the second link, and a dedup rule that merged on *any* outbound
    link would silently collapse two distinct jobs into one — deleting a real
    vacancy and undercounting demand. Phase 4 must treat the first link as the
    candidate and still corroborate; this adapter's job is to preserve the
    ordering that makes that possible, and to not decide.

    Tracking parameters are stripped by ``canonical_url`` so a link carrying
    ``?utm_source=telegram`` matches the blog's own permalink exactly — which is
    what makes the deterministic path work at all.
    """
    links: list[str] = []
    seen: set[str] = set()
    for anchor in text_node.css("a[href]"):
        href = (anchor.attributes.get("href") or "").strip()
        if not href or href.startswith(("#", "mailto:", "javascript:", "tel:")):
            continue
        if not href.startswith("http"):
            continue
        url = canonical_url(href)
        if url not in seen:
            seen.add(url)
            links.append(url)
    return tuple(links)
