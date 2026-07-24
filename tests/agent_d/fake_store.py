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

    def touch_seen(self, ids: list[str]) -> int:
        self.touched.extend(ids)
        for i in ids:
            if i in self.rows and self.rows[i].status == "stale":
                self.rows[i].status = "active"
        return len(ids)

    def upsert_batch(self, rows: list[Any]) -> None:
        self.upsert_calls += 1
        ordered = sorted(rows, key=lambda r: r.duplicate_of is not None)
        present = set(self.rows)
        for row in ordered:
            if row.duplicate_of is not None:
                assert row.duplicate_of in present, (
                    f"FK violation: {row.course_id} -> {row.duplicate_of} not yet inserted"
                )
            self.rows[row.course_id] = row
            present.add(row.course_id)
