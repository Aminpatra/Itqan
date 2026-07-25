"""The persisted shape of a course — built by the pipeline, written by the store.

The supply-side mirror of Agent B's ``PersistedPosting``. Lifecycle columns
(first_seen_at, last_seen_at, missed_cycles, stale_since) are owned by the
store's SQL, so they are absent here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PersistedCourse:
    course_id: str
    source: str
    source_group: str
    source_type: str
    source_url: str

    name: str
    raw_description: str
    content_hash: str

    status: str = "active"

    taught_skills: list[str] = field(default_factory=list)
    provider: Optional[str] = None
    level: Optional[str] = None
    primary_language: Optional[str] = None
    subject: Optional[str] = None
    country: Optional[str] = None

    review_reason: Optional[str] = None
    duplicate_of: Optional[str] = None

    attribution: Optional[str] = None
    license: Optional[str] = None

    # --- volatile quality/price signals -------------------------------------
    # Refreshed every cycle by the store's lightweight path, independent of
    # content_hash. price is flattened into columns (not jsonb) so a consumer can
    # `WHERE price_is_free` / `ORDER BY rating` without unpacking. None throughout
    # when the provider reports nothing — never a fabricated default.
    rating: Optional[float] = None
    review_count: Optional[int] = None
    enrollment_count: Optional[int] = None
    last_updated: Optional[str] = None       # ISO8601 string; store parses to timestamptz
    price_amount: Optional[float] = None
    price_currency: Optional[str] = None
    price_is_free: Optional[bool] = None

    extraction_model: Optional[str] = None
    embedding: Optional[list[float]] = None

    @property
    def is_canonical(self) -> bool:
        return self.duplicate_of is None

    @classmethod
    def price_columns(cls, price: dict | None) -> dict:
        """Flatten a RawCourse price dict into the three stored columns. None
        price (provider reported nothing) -> all None; never a guessed is_free."""
        if not price:
            return {"price_amount": None, "price_currency": None, "price_is_free": None}
        return {
            "price_amount": price.get("amount"),
            "price_currency": price.get("currency"),
            "price_is_free": price.get("is_free"),
        }
