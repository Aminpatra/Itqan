"""What every source adapter produces, and what it must not.

``RawPosting`` is deliberately *raw*: it holds what the source literally said,
not an interpretation of it. No adapter assigns a sector, a skill list or a
legitimacy score — those are downstream nodes' jobs, and an adapter that guessed
at them would be fabricating from a single source's conventions.

The one rule that matters more than the rest: **a field is None when the source
did not state it.** Every default here is empty or None for that reason. An
adapter filling `company` with something adjacent (a feed's author, a channel's
name) manufactures an employer that does not exist, and — because
`no_employer_named` is a legitimacy signal — simultaneously disables the check
that would have caught it. That is why `AdapterResult` counts `skipped` rather
than letting adapters improvise a value to avoid a gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Iterator, Protocol, runtime_checkable

# Source kinds, mirroring the CHECK constraint on job_postings.source_type.
# Duplicated deliberately: a mismatch here surfaces as a Python error at parse
# time rather than a constraint violation mid-transaction hours later.
SOURCE_TYPES = ("html_scrape", "blogger_feed", "telegram")

# Mirroring the CHECK constraints added in migration 0006. Only
# ('vacancy', 'company') is eligible for aggregation; both default to 'unknown'
# so a classifier that fails to run excludes a posting rather than publishing it.
LISTING_INTENTS = ("vacancy", "seeking", "service", "unknown")
POSTER_TYPES = ("company", "individual", "unknown")


@dataclass(frozen=True)
class RawPosting:
    """One posting exactly as published, before any interpretation.

    Frozen because the parse is the single point of truth for what the source
    said. Downstream nodes derive new objects; nothing edits a RawPosting after
    the adapter returns it, so "where did this value come from" always has one
    answer.
    """

    # --- identity ---
    source: str
    source_group: str
    source_type: str
    source_url: str

    # --- content ---
    title: str
    raw_description: str

    # --- what the source may or may not have stated ---
    company: str | None = None
    location_text: str | None = None
    posted_date: datetime | None = None
    # Only for sources that publish prose like "3 days ago". Kept as text and
    # resolved downstream against fetch time; never parsed into posted_date
    # here, because the same string means a different day on every fetch and
    # would re-hash the posting every cycle.
    posted_date_text: str | None = None

    # --- provenance the SOURCE stated, never the adapter's opinion ---
    # 'seeking' is set only when the publisher's own category says so (e.g. a
    # listing filed under "Jobs Wanted"). Everything else stays 'unknown' and is
    # classified downstream where the decision is measurable. An adapter that
    # guessed here would put an unverifiable judgement in the same column as a
    # stated fact, and nothing later could tell them apart.
    listing_intent: str = "unknown"
    poster_type: str = "unknown"

    # --- adapter telemetry, not content ---
    # Publisher-assigned tags. Telemetry ONLY in this phase: they are not fed to
    # the extraction prompt, because a model shown the label echoes it, and then
    # "label disagrees with extraction" can never fire. Measure agreement first.
    labels: tuple[str, ...] = ()
    # Links the posting points at. This is the deterministic dedup path: a
    # Telegram post linking to a blog post resolves that pair by URL rather than
    # by embedding similarity.
    outbound_links: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.source_type not in SOURCE_TYPES:
            raise ValueError(
                f"unknown source_type {self.source_type!r}; expected one of {SOURCE_TYPES}"
            )
        if not self.source_url:
            raise ValueError("source_url is required — it is the natural key for dedup")
        if not self.title and not self.raw_description:
            raise ValueError("a posting with neither title nor description carries no content")
        if self.listing_intent not in LISTING_INTENTS:
            raise ValueError(f"listing_intent {self.listing_intent!r} not in {LISTING_INTENTS}")
        if self.poster_type not in POSTER_TYPES:
            raise ValueError(f"poster_type {self.poster_type!r} not in {POSTER_TYPES}")


@dataclass
class AdapterResult:
    """The outcome of one source's fetch — including the ways it went wrong.

    A bare ``list[RawPosting]`` cannot distinguish "this source published nothing
    today" from "this source blocked us halfway through", and those must not be
    treated alike: the second means the inventory we hold is incomplete, so
    ageing it toward deletion would delete real jobs because of a 429.
    """

    source: str
    postings: list[RawPosting] = field(default_factory=list)

    # Fetched successfully but unusable — a photo-only Telegram post, an entry
    # with no text. Expected and healthy; counted so a sudden rise is visible.
    skipped: int = 0

    # Fatal for this source. The adapter returns rather than raises so one dead
    # source cannot abort a cycle that three healthy ones are mid-way through.
    error: str | None = None

    # True when the fetch stopped early — blocked, rate-limited, or a page that
    # failed after earlier pages succeeded. THE consequential flag: postings is
    # non-empty and error may be None, so anything asking "did this work?" by
    # truthiness gets the wrong answer. See healthy_sources in the runlog node.
    partial: bool = False
    blocked_after_n: int | None = None

    # True when the fetch stopped for a reason of OUR choosing rather than the
    # source's: a --limit cap, or the "ten known posts in a row" early stop. The
    # source was healthy and there was more to read, so the inventory we hold is
    # just as incomplete as after a block — and ageing it would age postings we
    # deliberately did not ask for.
    #
    # Separate from `partial` because it is not a failure: the run is fine, the
    # health record should stay green, and only the ageing decision changes. That
    # distinction was missing, and it cost real data — a sequence of `--limit 40`
    # runs pushed 139 live postings to missed_cycles=2, one cycle from stale.
    truncated: bool = False

    pages_fetched: int = 0
    bytes_fetched: int = 0

    @property
    def ok(self) -> bool:
        """Complete and clean. Note `partial` alone makes this False even with
        postings in hand and no error — that is the entire point of the flag."""
        return self.error is None and not self.partial

    @property
    def may_age_inventory(self) -> bool:
        """Is this fetch a trustworthy census of what the source still lists?

        Only a clean AND complete pass may age inventory toward deletion. A
        blocked, errored, or deliberately truncated fetch saw a subset, so
        everything it did not see is 'not observed', never 'gone'.
        """
        return self.ok and not self.truncated

    def extend(self, postings: Iterable[RawPosting]) -> None:
        self.postings.extend(postings)

    def fail(self, message: str, *, partial: bool | None = None) -> "AdapterResult":
        """Record a fatal error. Partial defaults to 'we already had some', which
        is the honest reading: postings in hand means the source was reachable
        and stopped, not that it was empty."""
        self.error = message
        self.partial = bool(self.postings) if partial is None else partial
        if self.partial and self.blocked_after_n is None:
            self.blocked_after_n = len(self.postings)
        return self


@runtime_checkable
class SourceAdapter(Protocol):
    """The whole interface. Adapters are fetch-and-parse and nothing else.

    Nothing here touches the database. That is structural, not stylistic: it is
    what makes every adapter testable offline against a saved fixture, and it is
    what prevents a scraper bug from being able to escalate into a write.
    """

    name: str
    source_group: str
    source_type: str

    def fetch(self, *, limit: int | None = None) -> AdapterResult: ...


class BaseAdapter:
    """Shared plumbing: identity, limit handling, and the error boundary.

    Subclasses implement ``_fetch`` and may raise freely; ``fetch`` converts any
    escape into an errored result. Adapters run inside a fan-out where a raised
    exception would take down the branch and lose the postings already parsed.
    """

    name: str = ""
    source_group: str = ""
    source_type: str = ""

    def __init__(self, *, name: str | None = None, source_group: str | None = None) -> None:
        self.name = name or self.name
        self.source_group = source_group or self.source_group
        if not self.name:
            raise ValueError("an adapter needs a name; it becomes job_postings.source")
        if not self.source_group:
            raise ValueError(
                "an adapter needs a source_group; near-dup merging is scoped by it, "
                "and an empty group would silently pool unrelated publishers"
            )
        if self.source_type not in SOURCE_TYPES:
            raise ValueError(f"{type(self).__name__}.source_type must be one of {SOURCE_TYPES}")

    def new_result(self) -> AdapterResult:
        return AdapterResult(source=self.name)

    def fetch(self, *, limit: int | None = None) -> AdapterResult:
        result = self.new_result()
        try:
            self._fetch(result, limit=limit)
        except Exception as exc:  # noqa: BLE001 - deliberate boundary, see docstring
            result.fail(f"{type(exc).__name__}: {exc}")
        # Hitting the cap means the source had more to give. Marked here, once,
        # rather than at each adapter's several `return` sites — every adapter
        # honours `limit` through this method, so this cannot be forgotten by a
        # new one. (Early stops that are the adapter's own idea set the flag
        # themselves; see el7far's known-run stop.)
        if limit is not None and len(result.postings) >= limit:
            result.truncated = True
            del result.postings[limit:]
        return result

    def _fetch(self, result: AdapterResult, *, limit: int | None) -> None:
        raise NotImplementedError

    @staticmethod
    def _take(items: Iterable[RawPosting], limit: int | None) -> Iterator[RawPosting]:
        for i, item in enumerate(items):
            if limit is not None and i >= limit:
                return
            yield item
