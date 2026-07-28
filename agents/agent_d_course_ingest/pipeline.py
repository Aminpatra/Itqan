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

from dataclasses import dataclass, field
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
    # Courses whose extraction call raised. NOT the same as `rejected`: we did
    # not fail to find skills, we failed to look. They are dropped from the
    # cycle and retried next, leaving any stored row untouched.
    extraction_failed: int = 0
    extraction_errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
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
        extracted: list[_Work] = []
        for item in work:
            # A cheap gate BEFORE the call: a course with almost no text cannot
            # yield skills, and paying an LLM call to discover that is pure
            # waste at catalog scale. Deliberately generous — freeCodeCamp's
            # one-line descriptions are real and must pass.
            if not _has_enough_text(item.raw, self.config):
                item.row.status = "rejected"
                item.row.review_reason = "insufficient_text"
                summary.rejected += 1
                extracted.append(item)
                continue
            try:
                self._extract(item, summary)
            except Exception as exc:  # noqa: BLE001 - one course must not sink the batch
                # A 429, a timeout or a schema violation means we never looked,
                # which is NOT the same as looking and finding nothing. Writing
                # the row now would either publish a skill-less course or
                # overwrite a good stored row with an empty one, so this course
                # is dropped from the cycle and retried next.
                summary.extraction_failed += 1
                summary.extraction_errors.append(
                    f"{item.raw.source_url}: {type(exc).__name__}: {exc}")
                continue
            if not item.row.taught_skills:
                item.row.status = "rejected"
                item.row.review_reason = "no_extractable_skills"
                summary.rejected += 1
            extracted.append(item)

        work = extracted
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
            volatile_observed=raw.volatile_observed,
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
        union of the store and this cycle.

        The in-cycle half is one matrix product rather than a Python double
        loop. Agent B never needed this — a cycle brings in a few dozen postings
        — but a course catalog batch is thousands, and n^2 pure-Python dot
        products over 1536 dimensions is billions of operations. Same comparisons,
        same order, same results.
        """
        import numpy as np

        recent = self.config.course_neardup_recent_days
        t_in = self.config.neardup_in_group_threshold
        t_cross = self.config.neardup_cross_group_threshold
        k = self.config.neardup_candidates

        usable = [w for w in embedded if w.row.embedding is not None]
        sims = None
        if len(usable) > 1:
            matrix = np.asarray([w.row.embedding for w in usable], dtype=np.float64)
            # Vectors are L2-normalized by the embedder, so the dot product IS
            # cosine — the same assumption the old per-pair loop made.
            sims = matrix @ matrix.T
            index = {id(w): i for i, w in enumerate(usable)}
            groups = np.array([hash(w.raw.source_group) for w in usable])
            ids = [w.course_id for w in usable]
            # `active` mirrors the old loop's `other.row.duplicate_of is not None`
            # skip: an item marked duplicate stops being a merge target for the
            # items considered after it.
            active = np.ones(len(usable), dtype=bool)

        for item in embedded:
            if item.row.duplicate_of is not None:
                continue
            best_in: tuple[float, str] | None = None
            best_cross: tuple[float, str] | None = None

            if sims is not None and item.row.embedding is not None:
                i = index[id(item)]
                row = sims[i]
                valid = active.copy()
                valid[i] = False
                same = valid & (groups == groups[i])
                # Only the higher course_id of a cross-group pair reports it, so
                # one pair yields one review rather than two.
                cross = valid & (groups != groups[i]) & np.array(
                    [item.course_id > other for other in ids])
                for mask, is_same in ((same, True), (cross, False)):
                    if not mask.any():
                        continue
                    j = int(np.argmax(np.where(mask, row, -np.inf)))
                    best_in, best_cross = _acc(best_in, best_cross, float(row[j]), ids[j],
                                               same_group=is_same, allow_cross=True)

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
                if sims is not None and item.row.embedding is not None:
                    active[index[id(item)]] = False
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
        "volatile_observed": raw.volatile_observed,
        "rating": raw.rating,
        "review_count": raw.review_count,
        "enrollment_count": raw.enrollment_count,
        "last_updated": raw.last_updated,
        **PersistedCourse.price_columns(raw.price),
    }


def _has_enough_text(raw: RawCourse, config: Any) -> bool:
    """Is there enough here for skill extraction to be worth a call?

    Name AND description together, because a descriptive title alone is real
    evidence ("Machine Learning with Python" names its skills); it is the
    genuinely empty record we refuse to pay for.
    """
    text = f"{raw.name or ''} {raw.raw_description or ''}".strip()
    return len(text) >= getattr(config, "course_min_text_chars", 30)


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


def _acc(best_in, best_cross, sim, cid, *, same_group, allow_cross):
    if same_group:
        if best_in is None or sim > best_in[0]:
            best_in = (sim, cid)
    elif allow_cross:
        if best_cross is None or sim > best_cross[0]:
            best_cross = (sim, cid)
    return best_in, best_cross
