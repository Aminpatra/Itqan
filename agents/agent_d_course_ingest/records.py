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

    extraction_model: Optional[str] = None
    embedding: Optional[list[float]] = None

    @property
    def is_canonical(self) -> bool:
        return self.duplicate_of is None
