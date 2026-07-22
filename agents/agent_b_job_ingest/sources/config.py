"""Which sources exist, and the checks that must fail startup rather than warn.

Two invariants are enforced here, both fatal by design.

**1. Handles are unique after normalization.** ``@OmanJob1``,
``https://t.me/s/OmanJob1/`` and ``omanjob1`` are one channel written three ways.
Two config entries that normalize to the same handle produce two
``PoliteClient``s — and therefore two independent rate limiters — pointed at one
host, halving the real interval while every log line still shows the polite
number. Downstream it is worse: depending on how ``source`` is derived you get
either a unique-index violation part-way through a cycle, or *distinct*
posting ids for identical postings, which double-counts every
``frequency_count`` in the table Agent C trusts. Both are fabricated numbers, and
neither appears in the run log. A warning would be read once and forgotten;
refusing to start cannot be.

**2. A Telegram channel is not live until a human says so.** ``terms_reviewed``
defaults to False, and the only thing that changes it is a person editing this
file — never a default, never inference from the channel loading, never a flag
at the command line where whoever runs the process could set it. The adapter is
generic, so pointing it at a new channel is a one-line change; the guard exists
precisely because adding a source is *too easy* to leave to review-by-intention.
An entry already set True says nothing about the next one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from .base import SOURCE_TYPES

# Telegram's own rule: 5-32 chars, alphanumerics and underscores, starting with
# a letter. Validated rather than assumed, so a typo'd handle fails at config
# load instead of as a 404 mid-cycle.
_HANDLE_RE = re.compile(r"^[a-z][a-z0-9_]{4,31}$")


class SourceConfigError(Exception):
    """Raised at startup. Deliberately not recoverable — see module docstring."""


def normalize_handle(raw: str) -> str:
    """Reduce any way of writing a Telegram handle to its canonical form.

    Accepts ``@Name``, ``Name``, ``t.me/Name``, ``https://t.me/s/Name/``, and
    the same with query strings. Case is folded because Telegram handles are
    case-insensitive — which is exactly why two entries differing only in case
    are the same channel and must collide rather than coexist.
    """
    value = (raw or "").strip()
    if not value:
        raise SourceConfigError("empty Telegram handle")

    if "//" in value or value.lower().startswith("t.me"):
        candidate = value if "//" in value else f"https://{value}"
        path = urlsplit(candidate).path
        segments = [s for s in path.split("/") if s]
        # /s/<handle> is the logged-out preview surface; /<handle> the canonical
        # permalink. Both name the same channel.
        if segments and segments[0].lower() == "s":
            segments = segments[1:]
        if not segments:
            raise SourceConfigError(f"no channel in Telegram URL {raw!r}")
        value = segments[0]

    value = value.lstrip("@").rstrip("/").casefold()

    if not _HANDLE_RE.match(value):
        raise SourceConfigError(
            f"{raw!r} normalizes to {value!r}, which is not a valid Telegram handle "
            "(5-32 chars, starts with a letter, letters/digits/underscore only)"
        )
    return value


@dataclass(frozen=True)
class SourceConfig:
    """A source Agent B may fetch."""

    name: str
    source_group: str
    source_type: str
    enabled: bool = True

    # blogger_feed / html_scrape
    base_url: str = ""

    # telegram
    handle: str = ""
    # Set by a human who has read the channel's terms and content. Never set in
    # code, never defaulted True, never inferred from the channel working.
    terms_reviewed: bool = False

    def __post_init__(self) -> None:
        if self.source_type not in SOURCE_TYPES:
            raise SourceConfigError(
                f"{self.name}: source_type {self.source_type!r} not in {SOURCE_TYPES}"
            )
        if not self.source_group:
            raise SourceConfigError(f"{self.name}: source_group is required")
        if self.source_type == "telegram":
            if not self.handle:
                raise SourceConfigError(f"{self.name}: telegram source needs a handle")
        elif not self.base_url:
            raise SourceConfigError(f"{self.name}: {self.source_type} source needs a base_url")

    @property
    def normalized_handle(self) -> str:
        return normalize_handle(self.handle) if self.handle else ""

    @property
    def identity(self) -> str:
        """What two entries must not share.

        For Telegram that is the normalized handle; for web sources the host.
        Deliberately not the ``name``: two entries named differently but
        pointing at one channel are the collision that matters, and comparing
        names would miss it entirely.
        """
        if self.source_type == "telegram":
            return f"telegram:{self.normalized_handle}"
        return f"web:{urlsplit(self.base_url).netloc.lower()}"


# ---------------------------------------------------------------------------
# The registry. Sequencing per the approved plan: the Atom feed and one verified
# Telegram channel first; Dubizzle lands in phase 3.
#
# el7far and omanjob1 share a source_group because the Telegram channel's posts
# link to the blog — confirmed by inspecting outbound links, not assumed from
# their subject matter. That grouping is what allows 0.93 auto-merge between
# them; an unrelated publisher would need 0.97 *and* human review.
# ---------------------------------------------------------------------------
DEFAULT_SOURCES: tuple[SourceConfig, ...] = (
    SourceConfig(
        name="el7far",
        source_group="el7far_network",
        source_type="blogger_feed",
        base_url="https://oman.el7far.com",
        enabled=True,
    ),
    SourceConfig(
        name="tg_omanjob1",
        source_group="el7far_network",
        source_type="telegram",
        handle="@omanjob1",
        # Reviewed and enabled by a human on 2026-07-21. The flag records that
        # judgement and nothing else — it is not evidence the channel works, and
        # it does not carry over to any other channel added later.
        terms_reviewed=True,
        enabled=True,
    ),
    SourceConfig(
        name="dubizzle",
        # Its own group: an unrelated publisher, so a near-duplicate against
        # el7far needs 0.97 AND human review, never an auto-merge.
        source_group="dubizzle",
        source_type="html_scrape",
        base_url="https://www.dubizzle.com.om",
        # Disabled deliberately, not for lack of an adapter.
        #
        # It is a classifieds site: a large share of its job listings are people
        # advertising themselves rather than employers hiring, and the page
        # exposes no poster identity at all — no seller name, no company name,
        # no business badge. So `poster_type` cannot be anything but 'unknown',
        # and migration 0006 makes 'unknown' ineligible for aggregation. Every
        # row it produced today would be stored and never counted.
        #
        # Running it would therefore spend two requests per posting to fill a
        # table nothing reads. Enable it once posters can actually be
        # classified; `--sources dubizzle --dry-run` inspects it meanwhile.
        enabled=False,
    ),
)


def validate_source_config(
    sources: tuple[SourceConfig, ...] | list[SourceConfig],
    *,
    dry_run: bool = False,
) -> list[SourceConfig]:
    """Check the registry and return the sources that may actually run.

    Raises on any duplicate identity — including among disabled entries, because
    a collision left in the file is a collision waiting for someone to flip
    ``enabled``.

    ``dry_run`` relaxes only the ``terms_reviewed`` gate, and only because a dry
    run makes no writes and is how a human inspects a channel in order to decide
    whether to review it. It does not relax uniqueness: a dry run with two
    clients on one host is impolite in exactly the way the check exists to
    prevent.
    """
    if not sources:
        raise SourceConfigError("no sources configured")

    seen: dict[str, list[str]] = {}
    for source in sources:
        seen.setdefault(source.identity, []).append(source.name)

    collisions = {identity: names for identity, names in seen.items() if len(names) > 1}
    if collisions:
        detail = "; ".join(
            f"{identity} claimed by {', '.join(names)}" for identity, names in sorted(collisions.items())
        )
        raise SourceConfigError(
            f"duplicate source identities: {detail}. Two entries for one host or channel "
            "run two rate limiters against it and double-count every posting."
        )

    runnable: list[SourceConfig] = []
    for source in sources:
        if not source.enabled:
            continue
        if source.source_type == "telegram" and not source.terms_reviewed and not dry_run:
            raise SourceConfigError(
                f"{source.name} (@{source.normalized_handle}) is enabled but terms_reviewed "
                "is False. A human must review the channel and set it explicitly; it is "
                "never set in code. Use --dry-run to inspect the channel first."
            )
        runnable.append(source)

    if not runnable:
        raise SourceConfigError("every configured source is disabled")
    return runnable


def select_sources(
    names: str | None = None,
    *,
    sources: tuple[SourceConfig, ...] | None = None,
    dry_run: bool = False,
) -> list[SourceConfig]:
    """Resolve --sources into validated configs.

    Validation runs over the WHOLE registry before filtering: a duplicate must
    fail startup whether or not this invocation happens to select the entries
    involved.
    """
    registry = sources if sources is not None else DEFAULT_SOURCES
    runnable = validate_source_config(registry, dry_run=dry_run)
    if not names:
        return runnable

    wanted = [n.strip() for n in names.split(",") if n.strip()]
    by_name = {s.name: s for s in runnable}

    # A disabled source can be named explicitly under --dry-run, which writes
    # nothing — that is how an operator inspects one before deciding to enable
    # it. It is never picked up by the DEFAULT set, so "disabled" still means
    # "does not run"; it only stops meaning "cannot be looked at".
    if dry_run:
        for source in registry:
            by_name.setdefault(source.name, source)

    unknown = [n for n in wanted if n not in by_name]
    if unknown:
        known = ", ".join(sorted(by_name)) or "(none runnable)"
        raise SourceConfigError(f"unknown source(s): {', '.join(unknown)}. Available: {known}")
    return [by_name[n] for n in wanted]
