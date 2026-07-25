"""The course ingestion tail: RawCourses in, rows written.

The supply-side mirror of Agent B's pipeline, stage-for-stage, with two
differences that fall out of "courses aren't scam-prone job postings":

  * NO legitimacy filter. Courses come from vetted platforms, so a fraud filter
    would spend LLM calls confirming what the source vetting guarantees. In its
    place a **quality gate**: a course that yields no extractable skill is
    ``rejected`` (kept for audit, out of stats) — thin records must not pollute
    the supply table.
  * NO link dedup. A job posting linking to its blog original is a Telegram/blog
    thing; courses have no such cross-source links. Near-dup by embedding still
    runs (the same course re-listed under two slugs).

Ordering still exists to make a warm cycle cheap: an unchanged course costs one
UPDATE and no LLM or embedding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .hashing import content_hash, id_for
from .records import PersistedCourse
from .schemas import CourseExtraction
from .sources.base import RawCourse


@dataclass
class IngestSummary:
    received: int = 0
    unchanged: int = 0
    changed: int = 0
    new: int = 0
    rejected: int = 0
    needs_review: int = 0
    embed_duplicates: int = 0
    extractions: int = 0
    embeddings: int = 0
    written: int = 0
    # Unchanged courses whose volatile quality/price signals were refreshed this
    # cycle without re-extracting or re-embedding.
    volatile_refreshed: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)

    def merge(self, other: "IngestSummary") -> "IngestSummary":
        for k, v in other.__dict__.items():
            setattr(self, k, getattr(self, k) + v)
        return self


@dataclass
class _Work:
    raw: RawCourse
    course_id: str
    content_hash: str
    row: PersistedCourse


class CoursePipeline:
    def __init__(self, *, store: Any, extractor: Any, embedder: Any, config: Any,
                 model_name: str = "unknown") -> None:
        self.store = store
        self.extractor = extractor
        self.embedder = embedder
        self.config = config
        self.model_name = model_name

    # ------------------------------------------------------------------
    def run(self, courses: list[RawCourse]) -> IngestSummary:
        summary = IngestSummary(received=len(courses))
        deduped = _dedupe(courses)
        stored = self.store.lookup_hashes([id_for(c) for c in deduped])

        unchanged_volatile: list[dict] = []
        work: list[_Work] = []
        for raw in deduped:
            cid = id_for(raw)
            chash = content_hash(raw)
            if stored.get(cid) == chash:
                # Content unchanged, but price/rating may have moved: refresh the
                # volatile signals every cycle WITHOUT re-extracting or embedding.
                unchanged_volatile.append(_volatile_row(cid, raw))
                continue
            summary.new += cid not in stored
            summary.changed += cid in stored
            work.append(self._new_work(raw, cid, chash))

        summary.unchanged = self.store.refresh_volatile(unchanged_volatile)
        summary.volatile_refreshed = summary.unchanged

        # extract, then the quality gate (needs the skills to judge them)
        for item in work:
            self._extract(item, summary)
            if not item.row.taught_skills:
                item.row.status = "rejected"
                item.row.review_reason = "no_extractable_skills"
                summary.rejected += 1

        live = [w for w in work if w.row.status != "rejected"]

        if self.embedder is not None:
            self._embed(live, summary)
            self._resolve_neardup(live, summary)

        self._flatten_duplicate_chains(work)

        rows = [w.row for w in work]
        self.store.upsert_batch(rows)
        summary.written = len(rows)
        return summary

    # ------------------------------------------------------------------
    def _new_work(self, raw: RawCourse, cid: str, chash: str) -> _Work:
        row = PersistedCourse(
            course_id=cid, source=raw.source, source_group=raw.source_group,
            source_type=raw.source_type, source_url=raw.source_url,
            name=raw.name, raw_description=raw.raw_description, content_hash=chash,
            provider=raw.provider, level=raw.level, primary_language=raw.primary_language,
            attribution=raw.attribution, license=raw.license,
            extraction_model=self.model_name,
            # Volatile signals travel with the full upsert for changed/new courses
            # (unchanged ones go through refresh_volatile instead). Deterministic,
            # no LLM — they came straight from the provider response/page.
            rating=raw.rating, review_count=raw.review_count,
            enrollment_count=raw.enrollment_count, last_updated=raw.last_updated,
            **PersistedCourse.price_columns(raw.price),
        )
        return _Work(raw=raw, course_id=cid, content_hash=chash, row=row)

    def _extract(self, item: _Work, summary: IngestSummary) -> None:
        summary.extractions += 1
        result: CourseExtraction = self.extractor.invoke({
            "name": item.raw.name,
            "provider": item.raw.provider or "unknown",
            "body": item.raw.raw_description,
        })
        item.row.taught_skills = result.taught_skills
        # Extraction may refine level/subject the adapter left blank; never
        # overwrite a value the source already stated.
        if item.row.level is None and result.level:
            item.row.level = result.level
        item.row.subject = result.subject

    def _embed(self, items: list[_Work], summary: IngestSummary) -> None:
        if not items:
            return
        vectors = self.embedder.embed_documents([_essence_text(w) for w in items])
        if len(vectors) != len(items):
            raise RuntimeError("embedder returned a different number of vectors than texts")
        summary.embeddings += len(items)
        for item, vec in zip(items, vectors):
            item.row.embedding = list(vec)

    def _resolve_neardup(self, embedded: list[_Work], summary: IngestSummary) -> None:
        """Same design as Agent B: in-group >= 0.97 auto-merges; cross-group
        >= 0.97 goes to needs_review, never auto-merges. Candidate set is the
        union of the store and this cycle."""
        recent = self.config.course_neardup_recent_days
        t_in = self.config.neardup_in_group_threshold
        t_cross = self.config.neardup_cross_group_threshold
        k = self.config.neardup_candidates

        for item in embedded:
            if item.row.duplicate_of is not None:
                continue
            best_in: tuple[float, str] | None = None
            best_cross: tuple[float, str] | None = None

            for other in embedded:
                if other is item or other.row.embedding is None or other.row.duplicate_of is not None:
                    continue
                sim = _cosine(item.row.embedding, other.row.embedding)
                same = other.raw.source_group == item.raw.source_group
                allow_cross = item.course_id > other.course_id
                best_in, best_cross = _acc(best_in, best_cross, sim, other.course_id,
                                           same_group=same, allow_cross=allow_cross)

            for cand in self.store.find_neardup_candidates(
                item.row.embedding, recent_days=recent, limit=k, exclude_id=item.course_id
            ):
                sim = float(cand["similarity"])
                best_in, best_cross = _acc(best_in, best_cross, sim, cand["course_id"],
                                           same_group=cand["source_group"] == item.raw.source_group,
                                           allow_cross=True)

            if best_in and best_in[0] >= t_in:
                item.row.duplicate_of = best_in[1]
                summary.embed_duplicates += 1
            elif best_cross and best_cross[0] >= t_cross:
                item.row.status = "needs_review"
                item.row.review_reason = "cross_group_duplicate"
                summary.needs_review += 1

    def _flatten_duplicate_chains(self, work: list[_Work]) -> None:
        by_id = {w.course_id: w for w in work}
        for item in work:
            if item.row.duplicate_of is None:
                continue
            trail = [item.course_id]
            target = item.row.duplicate_of
            for _ in range(len(work) + 1):
                node = by_id.get(target)
                if node is None or node.row.duplicate_of is None:
                    break
                if target in trail:
                    canonical = min(trail + [target])
                    for cid in trail + [target]:
                        m = by_id.get(cid)
                        if m is not None:
                            m.row.duplicate_of = None if cid == canonical else canonical
                    target = canonical
                    break
                trail.append(target)
                target = node.row.duplicate_of
            item.row.duplicate_of = None if target == item.course_id else target


# ---------------------------------------------------------------------------
def _volatile_row(cid: str, raw: RawCourse) -> dict:
    """The volatile-column payload for an unchanged course's cheap refresh."""
    return {
        "course_id": cid,
        "rating": raw.rating,
        "review_count": raw.review_count,
        "enrollment_count": raw.enrollment_count,
        "last_updated": raw.last_updated,
        **PersistedCourse.price_columns(raw.price),
    }


def _dedupe(courses: list[RawCourse]) -> list[RawCourse]:
    seen: set[str] = set()
    out: list[RawCourse] = []
    for c in courses:
        cid = id_for(c)
        if cid not in seen:
            seen.add(cid)
            out.append(c)
    return out


def _essence_text(item: _Work) -> str:
    """Title + taught skills + provider + level — the differentiating content,
    NOT the full description. Learned on Agent B: shared boilerplate in
    descriptions makes different items look similar and falsely merge."""
    row = item.row
    parts = [(item.raw.name or "").strip()]
    scalars = " ".join(p for p in (row.level, row.provider, row.subject) if p)
    if scalars:
        parts.append(scalars)
    if row.taught_skills:
        parts.append("teaches: " + ", ".join(row.taught_skills))
    return "\n".join(p for p in parts if p)


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _acc(best_in, best_cross, sim, cid, *, same_group, allow_cross):
    if same_group:
        if best_in is None or sim > best_in[0]:
            best_in = (sim, cid)
    elif allow_cross:
        if best_cross is None or sim > best_cross[0]:
            best_cross = (sim, cid)
    return best_in, best_cross
