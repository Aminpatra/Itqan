"""An in-memory stand-in for JobStore, for the pipeline's offline tests.

The pipeline's logic — change detection, the legitimacy gate, link and near-dup
dedup, FK-ordered writes — is all Python and can be tested without Postgres. This
fake models just enough persistence for that: it remembers what was written so a
second run sees the same rows the real store would.

The near-dup search mirrors the SQL predicate (canonicals only, not rejected,
not self) so the classification logic is exercised the same way it will be
against the database. It does NOT model HNSW or ef_search — it is a linear scan —
because those affect recall, not the classification being tested here; the DB
suite covers the real index.
"""

from __future__ import annotations

from typing import Any


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class FakeStore:
    def __init__(self) -> None:
        self.rows: dict[str, Any] = {}
        self.touched: list[str] = []
        self.upsert_calls = 0

    # --- reads -------------------------------------------------------
    def lookup_hashes(self, posting_ids: list[str]) -> dict[str, str]:
        return {pid: self.rows[pid].content_hash for pid in posting_ids if pid in self.rows}

    def lookup_posts(self, source_post_urls: list[str]) -> dict[str, dict[str, Any]]:
        wanted = set(source_post_urls)
        out: dict[str, dict[str, Any]] = {}
        for pid, row in self.rows.items():
            post_url = row.source_post_url or row.source_url
            if post_url not in wanted:
                continue
            entry = out.setdefault(post_url, {"content_hash": row.content_hash, "posting_ids": []})
            entry["posting_ids"].append(pid)
        return out

    def existing_ids(self, posting_ids: list[str]) -> set[str]:
        return {pid for pid in posting_ids if pid in self.rows}

    def find_by_source_urls(self, urls: list[str]) -> dict[str, str]:
        by_url = {r.source_url: pid for pid, r in self.rows.items()}
        return {u: by_url[u] for u in urls if u in by_url}

    def find_neardup_candidates(self, embedding, *, recent_days, limit, exclude_id):
        scored = []
        for pid, row in self.rows.items():
            if pid == exclude_id or row.embedding is None:
                continue
            if row.duplicate_of is not None or row.status == "rejected":
                continue
            scored.append(
                {
                    "posting_id": pid,
                    "source_group": row.source_group,
                    "similarity": _cosine(embedding, row.embedding),
                }
            )
        scored.sort(key=lambda c: c["similarity"], reverse=True)
        return scored[:limit]

    def get_status(self, posting_id: str) -> str | None:
        row = self.rows.get(posting_id)
        return row.status if row else None

    # --- writes ------------------------------------------------------
    def touch_seen(self, posting_ids: list[str]) -> int:
        self.touched.extend(posting_ids)
        for pid in posting_ids:
            if pid in self.rows and self.rows[pid].status == "stale":
                self.rows[pid].status = "active"
        return len(posting_ids)

    def upsert_batch(self, rows: list[Any]) -> None:
        self.upsert_calls += 1
        # Mirror the real store EXACTLY: canonicals first, then INSERT in that
        # order with the FK checked against what has actually been written so
        # far. The first live run failed on a duplicate CHAIN (A -> B where B ->
        # C), which a looser "target is somewhere in the batch" check waved
        # through — so this fake now enforces the same order-sensitive rule
        # Postgres does.
        ordered = sorted(rows, key=lambda r: r.duplicate_of is not None)
        present = set(self.rows)
        for row in ordered:
            if row.duplicate_of is not None:
                assert row.duplicate_of in present, (
                    f"FK violation: {row.posting_id} references {row.duplicate_of}, "
                    "which is not yet inserted — a duplicate chain was not flattened"
                )
            self.rows[row.posting_id] = row
            present.add(row.posting_id)
