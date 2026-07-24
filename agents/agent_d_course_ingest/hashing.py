"""Course identity and change detection — the course analog of Agent B's hashing.

``course_id`` is stable identity (so a re-listed course updates one row);
``content_hash`` is over the stable content (name + description) so an unchanged
course skips the expensive extract/embed tail — the basis of the "second run
does nothing" property.
"""

from __future__ import annotations

import hashlib

from .sources.base import RawCourse

_SEP = "\x1f"
ID_LENGTH = 32


def _sha256_hex(*parts: str) -> str:
    return hashlib.sha256(_SEP.join(parts).encode("utf-8")).hexdigest()


def course_id(source: str, canonical_url: str) -> str:
    if not source or not canonical_url:
        raise ValueError("course_id needs both a source and a canonical url")
    return _sha256_hex(source, canonical_url)[:ID_LENGTH]


def content_hash(course: RawCourse) -> str:
    return _sha256_hex((course.name or "").strip(), (course.raw_description or "").strip())


def id_for(course: RawCourse) -> str:
    return course_id(course.source, course.source_url)
