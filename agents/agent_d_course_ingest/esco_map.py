"""Map course taught-skills to ESCO — the supply side of the shared vocabulary.

Reuses ``shared.job_market.map_skills_to_esco`` (the READ-ONLY tier resolver:
exact label -> alt label -> nearest-label embedding >= threshold -> unmapped)
against the SAME ``esco_labels`` tables Agents B and C use, and persists the
result into Agent D's OWN ``course_esco_map`` cache. It never writes Agent B's
``skill_esco_map`` — course skills and posting skills are different corpora, and
one agent's mappings must not silently change the other's, even though both point
at the same taxonomy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MapSummary:
    considered: int = 0
    exact: int = 0
    alt_label: int = 0
    embedding: int = 0
    unmapped: int = 0
    deferred: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


def map_new_course_skills(store: Any, embedder: Any, config: Any) -> MapSummary:
    """Resolve this cycle's not-yet-mapped course skill keys."""
    from shared.job_market import map_skills_to_esco

    summary = MapSummary()
    version = store.get_esco_version()
    if version is None:
        return summary  # taxonomy not synced; nothing to map against

    pending = store.pending_skill_keys(version=version)
    summary.considered = len(pending)
    if not pending:
        return summary

    mappings = map_skills_to_esco(pending, embedder=embedder, config=config)

    entries: list[dict[str, Any]] = []
    for m in mappings:
        if m.method == "deferred":
            # No embedder this run: "did not look" is not "looked and missed",
            # so it stays unwritten and is retried next cycle.
            summary.deferred += 1
            continue
        setattr(summary, m.method, getattr(summary, m.method) + 1)
        entries.append({
            "skill_key": m.skill_key,
            "esco_uri": m.esco_uri,
            "method": m.method,
            "similarity": m.similarity,
            "esco_version": version,
        })

    if entries:
        store.upsert_course_map(entries)
    return summary
