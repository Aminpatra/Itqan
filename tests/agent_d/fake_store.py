"""In-memory CourseStore stand-in for the pipeline's offline tests."""

from __future__ import annotations

from typing import Any


def _cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


class FakeCourseStore:
    def __init__(self) -> None:
        self.rows: dict[str, Any] = {}
        self.touched: list[str] = []
        self.upsert_calls = 0

    def lookup_hashes(self, ids: list[str]) -> dict[str, str]:
        return {i: self.rows[i].content_hash for i in ids if i in self.rows}

    def find_neardup_candidates(self, embedding, *, recent_days, limit, exclude_id):
        scored = []
        for cid, row in self.rows.items():
            if cid == exclude_id or row.embedding is None:
                continue
            if row.duplicate_of is not None or row.status == "rejected":
                continue
            scored.append({"course_id": cid, "source_group": row.source_group,
                           "similarity": _cosine(embedding, row.embedding)})
        scored.sort(key=lambda c: c["similarity"], reverse=True)
        return scored[:limit]

    def get_status(self, cid):
        r = self.rows.get(cid)
        return r.status if r else None

    _VOLATILE = ("rating", "review_count", "enrollment_count", "last_updated",
                 "price_amount", "price_currency", "price_is_free")

    def refresh_volatile(self, rows: list[dict]) -> int:
        """Mirror the store: update volatile columns on existing rows, no
        re-embed. Only counts rows that actually exist (like the real UPDATE).

        Mirrors the real SQL's ``volatile_observed`` gate — an unobserved refresh
        touches the lifecycle columns but leaves the stored values alone."""
        n = 0
        for r in rows:
            row = self.rows.get(r["course_id"])
            if row is None:
                continue
            if r.get("volatile_observed"):
                for column in self._VOLATILE:
                    setattr(row, column, r[column])
            if row.status == "stale":
                row.status = "active"
            self.touched.append(r["course_id"])
            n += 1
        return n

    def upsert_batch(self, rows: list[Any]) -> None:
        self.upsert_calls += 1
        ordered = sorted(rows, key=lambda r: r.duplicate_of is not None)
        present = set(self.rows)
        for row in ordered:
            if row.duplicate_of is not None:
                assert row.duplicate_of in present, (
                    f"FK violation: {row.course_id} -> {row.duplicate_of} not yet inserted"
                )
            # Same gate as the real ON CONFLICT branch: a changed course whose
            # quality lookup failed keeps the stored signals.
            existing = self.rows.get(row.course_id)
            if existing is not None and not getattr(row, "volatile_observed", False):
                for column in self._VOLATILE:
                    setattr(row, column, getattr(existing, column, None))
            self.rows[row.course_id] = row
            present.add(row.course_id)
