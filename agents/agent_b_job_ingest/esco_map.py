"""ESCO taxonomy loading and free-text skill mapping.

Two jobs, both deterministic and LLM-free by design:

``sync_taxonomy`` — one-off (and on new ESCO releases): parse the user-downloaded
skills CSV, load concepts and their labels, embed every label. ~13.9K concepts /
~40K labels in v1.2; the embedding spend is a few cents, once.

``map_new_skills`` — every cycle: take the skill keys the demand table is about
to aggregate and resolve each to an ESCO concept, in strict precedence:

    exact preferred-label match  >  exact alt-label match  >
    nearest label embedding >= threshold  >  honestly unmapped

The precedence is the design. Lexical matches are free and certain, so they go
first; the embedding step only pays for what lexical missed; and a miss is
RECORDED (method='unmapped') so no future cycle re-pays for it. No LLM anywhere:
a mapping either has a mechanical justification (equality, or a similarity score
stored beside it) or it does not happen — the same auditability rule as
grounding and legitimacy.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from shared.embeddings import embed_texts

EMBED_BATCH = 512


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------
def parse_esco_csv(path: Path) -> Iterator[dict[str, Any]]:
    """Yield one concept per row of ESCO's skills CSV.

    The real file's shape (verified against the v1.2 download): columns include
    conceptUri, skillType, preferredLabel, altLabels, description; altLabels is
    a NEWLINE-separated list inside one quoted cell. utf-8-sig because the
    bundle ships with a BOM.
    """
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "conceptUri" not in reader.fieldnames:
            raise ValueError(
                f"{path} does not look like an ESCO skills CSV (no conceptUri column). "
                "Expected the skills_en.csv file from the ESCO download bundle."
            )
        for row in reader:
            uri = (row.get("conceptUri") or "").strip()
            preferred = (row.get("preferredLabel") or "").strip()
            if not uri or not preferred:
                continue
            alts = [
                label.strip()
                for label in (row.get("altLabels") or "").splitlines()
                if label.strip()
            ]
            yield {
                "esco_uri": uri,
                "preferred_label": preferred,
                "skill_type": (row.get("skillType") or "").strip() or None,
                "description": (row.get("description") or "").strip() or None,
                "alt_labels": alts,
            }


def _label_key(label: str) -> str:
    """The same normalisation skill_demand_stats uses for skill_key, so an
    exact match is a plain equality."""
    return label.strip().lower()


# ---------------------------------------------------------------------------
# sync
# ---------------------------------------------------------------------------
@dataclass
class SyncSummary:
    skills: int = 0
    labels: int = 0
    embedded: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


def sync_taxonomy(store: Any, embedder: Any, *, path: Path, version: str) -> SyncSummary:
    """Load the taxonomy and embed its labels. Idempotent and resumable:
    concepts upsert, labels insert-if-absent, and the embedding pass only
    touches labels whose embedding is still NULL — so an interrupted run picks
    up where it stopped rather than re-paying."""
    summary = SyncSummary()

    batch_skills: list[dict[str, Any]] = []
    batch_labels: list[dict[str, Any]] = []

    def flush() -> None:
        # Each flush is ITS OWN transaction. The resumability promise — an
        # interrupted sync picks up where it stopped — only holds if progress
        # actually commits as it happens; the first real sync ran inside one
        # caller-supplied transaction and was invisible (and voidable) until the
        # very end, which is the opposite of what this loader was designed for.
        if batch_skills:
            with store.transaction():
                store.upsert_esco_skills(batch_skills, version=version)
            batch_skills.clear()
        if batch_labels:
            with store.transaction():
                store.insert_esco_labels(batch_labels)
            batch_labels.clear()

    for concept in parse_esco_csv(path):
        summary.skills += 1
        batch_skills.append({k: concept[k] for k in
                             ("esco_uri", "preferred_label", "skill_type", "description")})

        seen_keys = set()
        for label, preferred in [(concept["preferred_label"], True)] + [
            (alt, False) for alt in concept["alt_labels"]
        ]:
            key = _label_key(label)
            if not key or key in seen_keys:
                continue  # a concept whose alt label equals its preferred label
            seen_keys.add(key)
            summary.labels += 1
            batch_labels.append({
                "label_key": key,
                "label": label,
                "esco_uri": concept["esco_uri"],
                "is_preferred": preferred,
            })

        if len(batch_labels) >= 1000:
            flush()
    flush()

    # Embedding pass over whatever is still NULL — one transaction per batch,
    # so a crash mid-pass loses at most one batch's API spend.
    while True:
        missing = store.labels_missing_embeddings(limit=EMBED_BATCH)
        if not missing:
            break
        vectors = embed_texts(embedder, [m["label"] for m in missing])
        with store.transaction():
            store.set_label_embeddings([
                (m["label_key"], m["esco_uri"], vec) for m, vec in zip(missing, vectors)
            ])
        summary.embedded += len(missing)

    return summary


# ---------------------------------------------------------------------------
# mapping
# ---------------------------------------------------------------------------
@dataclass
class MapSummary:
    considered: int = 0
    exact: int = 0
    alt_label: int = 0
    embedding: int = 0
    unmapped: int = 0
    deferred: int = 0   # no embedder available; left for a future cycle

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


def map_new_skills(store: Any, embedder: Any, config: Any) -> MapSummary:
    """Resolve this cycle's not-yet-mapped skill keys.

    With no embedder (``--no-embed``) only the lexical steps run, and keys they
    miss are DEFERRED — not written as 'unmapped', because "we did not look" and
    "we looked and found nothing" must never share a row.
    """
    summary = MapSummary()
    version = store.get_esco_version()
    if version is None:
        return summary  # taxonomy not synced yet; nothing to map against

    pending = store.pending_skill_keys(version=version)
    summary.considered = len(pending)
    if not pending:
        return summary

    # 1+2 — lexical, free, certain.
    lexical = store.find_esco_by_labels(pending)
    entries: list[dict[str, Any]] = []
    for key in pending:
        hit = lexical.get(key)
        if hit is None:
            continue
        method = "exact" if hit["is_preferred"] else "alt_label"
        setattr(summary, method, getattr(summary, method) + 1)
        entries.append({
            "skill_key": key,
            "esco_uri": hit["esco_uri"],
            "method": method,
            "similarity": None,
            "esco_version": version,
        })

    remaining = [k for k in pending if k not in lexical]

    # 3 — embedding, only for what lexical missed.
    if remaining and embedder is not None:
        vectors = embed_texts(embedder, remaining)
        threshold = config.esco_map_threshold
        for key, vector in zip(remaining, vectors):
            candidates = store.nearest_esco_label(vector, limit=1)
            best = candidates[0] if candidates else None
            if best and float(best["similarity"]) >= threshold:
                summary.embedding += 1
                entries.append({
                    "skill_key": key,
                    "esco_uri": best["esco_uri"],
                    "method": "embedding",
                    "similarity": round(float(best["similarity"]), 4),
                    "esco_version": version,
                })
            else:
                summary.unmapped += 1
                entries.append({
                    "skill_key": key,
                    "esco_uri": None,
                    "method": "unmapped",
                    # The best similarity that WASN'T enough — stored so the
                    # threshold can be tuned on evidence later.
                    "similarity": round(float(best["similarity"]), 4) if best else None,
                    "esco_version": version,
                })
    elif remaining:
        summary.deferred = len(remaining)

    if entries:
        store.upsert_skill_map(entries)
    return summary
