"""What every course adapter produces, and what it must not.

``RawCourse`` is deliberately raw: it holds what the source literally published,
not an interpretation of it. No adapter assigns taught skills, a level it had to
guess, or an ESCO code — those are downstream jobs. A field is None when the
source did not state it.

The transport and robots layers come from ``shared.scraping`` (shared with Agent
B); only the item shape and the thin adapter base live here, because a course and
a job posting have almost no fields in common.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol, runtime_checkable

SOURCE_TYPES = ("api", "html_scrape")


@dataclass(frozen=True)
class RawCourse:
    """One course exactly as published, before any interpretation."""

    # --- identity ---
    source: str
    source_group: str
    source_type: str
    source_url: str

    # --- content ---
    name: str
    raw_description: str

    # --- what the source may or may not have stated ---
    provider: str | None = None          # the platform or teaching institution
    level: str | None = None             # only if the source labels difficulty
    primary_language: str | None = None  # ISO-639-1, if stated
    # Licensing the source imposes on reuse. freeCodeCamp's curriculum is
    # CC-BY-SA-4.0, which requires attribution; carried so it is stored on the row.
    attribution: str | None = None
    license: str | None = None

    # --- volatile quality/price signals (Agent E tie-breaking) --------------
    # Deterministic, from the provider's response/page — NEVER inferred. Stored
    # in the provider's own native scale (a 4.5 on Coursera and a 4.5 elsewhere
    # are NOT normalized here — that is a consumer's query-time concern). A
    # missing signal is None, never a 0. These refresh every cycle regardless of
    # content_hash, because price and ratings drift while title/description do not.
    rating: float | None = None
    review_count: int | None = None
    enrollment_count: int | None = None
    last_updated: str | None = None      # ISO8601, the provider's own field if present
    # {"amount": float|None, "currency": str|None, "is_free": bool}, or None when
    # the provider reports no price at all. Free -> amount 0.0, is_free True.
    price: dict | None = None

    def __post_init__(self) -> None:
        if self.source_type not in SOURCE_TYPES:
            raise ValueError(f"unknown source_type {self.source_type!r}; expected {SOURCE_TYPES}")
        if not self.source_url:
            raise ValueError("source_url is required — it is the natural key for dedup")
        if not self.name and not self.raw_description:
            raise ValueError("a course with neither name nor description carries no content")


@dataclass
class AdapterResult:
    """The outcome of one source's fetch — including the ways it went wrong.

    Mirrors Agent B's: ``partial`` is False-making even with items in hand, so a
    blocked-halfway source is never treated as a clean empty result (which would
    let staleness age inventory it never re-fetched)."""

    source: str
    courses: list[RawCourse] = field(default_factory=list)
    skipped: int = 0
    error: str | None = None
    partial: bool = False
    blocked_after_n: int | None = None
    pages_fetched: int = 0
    bytes_fetched: int = 0

    @property
    def ok(self) -> bool:
        return self.error is None and not self.partial

    def fail(self, message: str, *, partial: bool | None = None) -> "AdapterResult":
        self.error = message
        self.partial = bool(self.courses) if partial is None else partial
        if self.partial and self.blocked_after_n is None:
            self.blocked_after_n = len(self.courses)
        return self


@runtime_checkable
class CourseAdapter(Protocol):
    name: str
    source_group: str
    source_type: str

    def fetch(self, *, limit: int | None = None) -> AdapterResult: ...


class BaseAdapter:
    """Shared plumbing: identity, limit handling, and the error boundary.

    Subclasses implement ``_fetch`` and may raise freely; ``fetch`` converts any
    escape into an errored result, so a raised exception in a fan-out branch does
    not lose the courses already parsed."""

    name: str = ""
    source_group: str = ""
    source_type: str = ""

    def __init__(self, *, name: str | None = None, source_group: str | None = None) -> None:
        self.name = name or self.name
        self.source_group = source_group or self.source_group
        if not self.name:
            raise ValueError("an adapter needs a name; it becomes courses.source")
        if not self.source_group:
            raise ValueError("an adapter needs a source_group; near-dup merging is scoped by it")
        if self.source_type not in SOURCE_TYPES:
            raise ValueError(f"{type(self).__name__}.source_type must be one of {SOURCE_TYPES}")

    def new_result(self) -> AdapterResult:
        return AdapterResult(source=self.name)

    def fetch(self, *, limit: int | None = None) -> AdapterResult:
        result = self.new_result()
        try:
            self._fetch(result, limit=limit)
        except Exception as exc:  # noqa: BLE001 - deliberate boundary
            result.fail(f"{type(exc).__name__}: {exc}")
        if limit is not None and len(result.courses) > limit:
            del result.courses[limit:]
        return result

    def _fetch(self, result: AdapterResult, *, limit: int | None) -> None:
        raise NotImplementedError

    @staticmethod
    def _dedupe(items: Iterable[RawCourse]) -> list[RawCourse]:
        seen: set[str] = set()
        out: list[RawCourse] = []
        for item in items:
            if item.source_url not in seen:
                seen.add(item.source_url)
                out.append(item)
        return out
