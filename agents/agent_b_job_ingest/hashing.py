"""Two hashes, each answering a different question, each with a specific trap.

``posting_id`` — "is this the same posting?" Identity. Stable for the life of a
posting across every cycle, so a repost of the same URL updates one row instead
of creating a second and double-counting its skills.

``content_hash`` — "has this posting changed since we last saw it?" Freshness.
When it matches what the store already holds, the whole expensive tail —
extraction, embedding — is skipped. That is the mechanism behind the phase gate
("second run does zero LLM, zero embed"), so what goes INTO this hash is
load-bearing: any field that changes between cycles without the posting really
changing would defeat it and force a permanent full re-extract.

The fields deliberately excluded are the ones that vary on their own:
``posted_date_text`` ("2 days ago" → "3 days ago" tomorrow), view counts (never
reach the adapter output at all), and ``last_seen_at``. Including any of them
would re-hash every posting every cycle — the exact failure the Telegram adapter
already guards against for view counts, restated here for relative dates.
"""

from __future__ import annotations

import hashlib
import re

from .sources.base import RawPosting

# ASCII Unit Separator. A delimiter that cannot occur in a URL or a source name,
# so ("ab", "c") and ("a", "bc") cannot collide into one id.
_SEP = "\x1f"

ID_LENGTH = 32


def _sha256_hex(*parts: str) -> str:
    joined = _SEP.join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def posting_id(source: str, canonical_url: str) -> str:
    """Stable identity: ``sha256(source || US || canonical_url)[:32]``.

    Scoped by ``source`` rather than global, so the same URL surfacing from two
    different sources stays two rows — which is correct, because they are two
    independent observations of demand, and the cross-source dedup that might
    later merge them is a separate, reviewable decision.

    The URL MUST already be canonical (tracking params stripped, trailing slash
    normalised) — that is done in the adapters via ``canonical_url``. Hashing a
    raw URL here would let ``?utm_source=telegram`` mint a second id for one
    posting and inflate every count derived from it.
    """
    if not source or not canonical_url:
        raise ValueError("posting_id needs both a source and a canonical url")
    return _sha256_hex(source, canonical_url)[:ID_LENGTH]


def content_hash(posting: RawPosting) -> str:
    """Change detection over the STABLE content only.

    Title and description are what extraction reads, so if both are byte-for-byte
    unchanged the extraction would be identical and re-running it buys nothing.
    Everything volatile is excluded by omission — see the module docstring for
    why each one would otherwise re-hash the posting every cycle.
    """
    return _sha256_hex(
        (posting.title or "").strip(),
        (posting.raw_description or "").strip(),
    )


def id_for(posting: RawPosting) -> str:
    """The posting_id for a RawPosting, using its already-canonical source_url."""
    return posting_id(posting.source, posting.source_url)


_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def child_source_url(post_url: str, role_title: str | None, index: int) -> str:
    """The source_url for one vacancy SPLIT out of a multi-job post.

    ``post_url#slug`` so each split vacancy is a distinct, stable key under the
    ``UNIQUE (source, source_url)`` constraint — and so a vacancy reached both as
    its own posting and as part of a roundup would still differ only by the
    fragment. The slug is the normalised role title (stable across cycles when
    the title is stable); the index is the fallback when a role has no usable
    title, and also disambiguates two roles that normalise to the same slug.
    """
    slug = _SLUG_STRIP.sub("-", (role_title or "").strip().casefold()).strip("-")
    fragment = f"{slug}-{index}" if slug else f"job-{index}"
    return f"{post_url}#{fragment}"
