"""Which course sources exist, and the checks that must fail startup.

Two source kinds with DIFFERENT consent models, decided 2026-07-24:

* ``html_scrape`` (freeCodeCamp) honors **robots.txt**, fail-closed, via
  ``shared.scraping.robots`` — the standard web-crawler contract.
* ``api`` (Coursera) is governed by the **API's published terms**, not robots:
  robots.txt is the Robots Exclusion Protocol for crawlers, while a documented
  public API consumed within its rate limits is a different relationship. That
  override is gated by a human ``terms_reviewed=True`` — the same discipline as
  the Telegram channel gate. An unreviewed API source cannot go live by accident.

Handles/hosts must be unique after normalization, same as Agent B: two configs
for one source would run two rate limiters at one host and double-count courses.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit

from .base import SOURCE_TYPES


class SourceConfigError(Exception):
    """Raised at startup. Deliberately not recoverable."""


@dataclass(frozen=True)
class CourseSourceConfig:
    name: str
    source_group: str
    source_type: str
    enabled: bool = True
    base_url: str = ""
    # Required True (outside --dry-run) for an api source: a human has read the
    # API's terms and accepts consuming it under them rather than under robots.
    terms_reviewed: bool = False

    # Can ONE cycle enumerate this source's entire inventory?
    #
    # This governs whether absence from a fetch is evidence of removal. Agent B's
    # sources are recent-first feeds, so "not in today's feed" eventually does
    # mean delisted; a 23,101-course catalog fetched 800 at a time is a SAMPLE,
    # and treating absence from a sample as removal deletes live courses for no
    # reason but our own pagination. A non-census source is never aged: its rows
    # leave only by an explicit operator purge.
    #
    # Set True only when the adapter provably reads the whole list every cycle.
    census: bool = False

    def __post_init__(self) -> None:
        if self.source_type not in SOURCE_TYPES:
            raise SourceConfigError(f"{self.name}: source_type {self.source_type!r} not in {SOURCE_TYPES}")
        if not self.source_group:
            raise SourceConfigError(f"{self.name}: source_group is required")
        if not self.base_url:
            raise SourceConfigError(f"{self.name}: base_url is required")

    @property
    def identity(self) -> str:
        return f"{self.source_type}:{urlsplit(self.base_url).netloc.lower()}"


DEFAULT_SOURCES: tuple[CourseSourceConfig, ...] = (
    CourseSourceConfig(
        name="coursera",
        source_group="coursera",
        source_type="api",
        base_url="https://api.coursera.org",
        # Reviewed and approved by the user on 2026-07-24: the Catalog API is
        # documented public/no-auth and consumed within its rate limits under
        # its terms. robots.txt disallows /api/ for crawlers, which does not
        # govern a documented API client — the decision is recorded here, not
        # worked around silently.
        terms_reviewed=True,
        enabled=True,
        # 23,101 courses read 800 at a time: every cycle sees a slice, so a
        # course's absence says nothing about whether it still exists.
        census=False,
    ),
    CourseSourceConfig(
        name="freecodecamp",
        source_group="freecodecamp",
        source_type="html_scrape",
        base_url="https://www.freecodecamp.org",
        enabled=True,
        # The catalog is one server-rendered index read in full every cycle, so
        # a course missing from it really has been withdrawn. If this adapter is
        # ever broadened to a paginated or partial source, this MUST go back to
        # False — it is the flag that licenses deletion.
        census=True,
    ),
)


def validate_source_config(
    sources: tuple[CourseSourceConfig, ...] | list[CourseSourceConfig],
    *,
    dry_run: bool = False,
) -> list[CourseSourceConfig]:
    if not sources:
        raise SourceConfigError("no course sources configured")

    seen: dict[str, list[str]] = {}
    for s in sources:
        seen.setdefault(s.identity, []).append(s.name)
    collisions = {i: n for i, n in seen.items() if len(n) > 1}
    if collisions:
        detail = "; ".join(f"{i} claimed by {', '.join(n)}" for i, n in sorted(collisions.items()))
        raise SourceConfigError(f"duplicate source identities: {detail}")

    runnable: list[CourseSourceConfig] = []
    for s in sources:
        if not s.enabled:
            continue
        if s.source_type == "api" and not s.terms_reviewed and not dry_run:
            raise SourceConfigError(
                f"{s.name} is an api source with terms_reviewed=False. A human must review "
                "the API's terms and set it explicitly; --dry-run may inspect it first."
            )
        runnable.append(s)
    if not runnable:
        raise SourceConfigError("every configured course source is disabled")
    return runnable


def select_sources(
    names: str | None = None,
    *,
    sources: tuple[CourseSourceConfig, ...] | None = None,
    dry_run: bool = False,
) -> list[CourseSourceConfig]:
    registry = sources if sources is not None else DEFAULT_SOURCES
    runnable = validate_source_config(registry, dry_run=dry_run)
    if not names:
        return runnable
    wanted = [n.strip() for n in names.split(",") if n.strip()]
    by_name = {s.name: s for s in runnable}
    if dry_run:
        for s in registry:
            by_name.setdefault(s.name, s)
    unknown = [n for n in wanted if n not in by_name]
    if unknown:
        known = ", ".join(sorted(by_name)) or "(none runnable)"
        raise SourceConfigError(f"unknown source(s): {', '.join(unknown)}. Available: {known}")
    return [by_name[n] for n in wanted]
