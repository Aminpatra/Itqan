"""The persisted shape of a posting — the row the pipeline builds and the store
writes.

Kept in its own module because two layers share it: the pipeline constructs it
from a RawPosting plus the extraction and legitimacy results, and the store maps
it to columns. Defining it in either layer would make the other import
"upward". The lifecycle columns (first_seen_at, last_seen_at, missed_cycles,
stale_since) are deliberately absent: those are owned by the store's SQL, which
is the only place that knows whether a row is new this cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional


@dataclass
class PersistedPosting:
    posting_id: str
    source: str
    source_group: str
    source_type: str
    source_url: str

    title: str
    raw_description: str
    content_hash: str

    status: str = "active"

    # The URL of the SOURCE POST this row came from. Empty means "same as
    # source_url" (the store falls back to source_url when writing). It differs
    # only for a vacancy SPLIT out of a multi-job roundup: then source_url is
    # post_url#role and this is the roundup's own URL. It is what lets change
    # detection work at the POST level — decide "has this post changed?" once,
    # before re-extracting all the vacancies it split into.
    source_post_url: str = ""

    # Extracted. All optional — a null is an honest "the posting did not say",
    # never a placeholder.
    sector: Optional[str] = None
    required_skills: list[str] = field(default_factory=list)
    # The employer, as the posting names it. Extracted and grounded since the
    # first version and then dropped for want of a column — see migration 0010.
    # None means the posting named none, which is a real answer.
    company: Optional[str] = None
    seniority_level: Optional[str] = None
    location: Optional[str] = None
    country: Optional[str] = None
    posted_date: Optional[date] = None

    legitimacy_score: Optional[float] = None
    review_reason: Optional[str] = None

    duplicate_of: Optional[str] = None

    listing_intent: str = "unknown"
    poster_type: str = "unknown"

    extraction_model: Optional[str] = None
    embedding: Optional[list[float]] = None

    @property
    def is_canonical(self) -> bool:
        return self.duplicate_of is None
