"""The course ingestion tail, offline against fakes.

The phase-4 gate is the first test: a second run over the same courses does zero
LLM and zero embedding work. The quality-gate and near-dup tests guard the two
course-specific behaviours.
"""

from __future__ import annotations

import pytest

from agents.agent_d_course_ingest.pipeline import CoursePipeline
from agents.agent_d_course_ingest.prompts.extraction import EXTRACTION_PROMPT
from agents.agent_d_course_ingest.schemas import CourseExtraction
from agents.agent_d_course_ingest.sources.base import RawCourse
from shared.config import Config
from shared.llm import structured
from tests.agent_d.fake_store import FakeCourseStore
from tests.fake_embedder import FakeEmbedder
from tests.fake_llm import FakeStructuredLLM


def make_pipeline(store=None, llm=None, embedder=None):
    llm = llm or FakeStructuredLLM()
    embedder = embedder or FakeEmbedder()
    return CoursePipeline(
        store=store or FakeCourseStore(),
        extractor=EXTRACTION_PROMPT | structured(llm, CourseExtraction),
        embedder=embedder,
        config=Config(),
        model_name="fake",
    ), llm, embedder


def course(url, *, source="coursera", group="coursera", stype="api",
           name="Python Basics", body=None, provider="Google Cloud"):
    return RawCourse(
        source=source, source_group=group, source_type=stype, source_url=url,
        name=name, raw_description=body or "Learn Python and SQL from the basics.",
        provider=provider, primary_language="en", license=None,
    )


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------
def test_second_run_does_zero_llm_and_embedding_work():
    store = FakeCourseStore()
    pipe, _, _ = make_pipeline(store=store)
    batch = [course("https://c.test/a"), course("https://c.test/b", name="SQL Deep Dive")]

    first = pipe.run(batch)
    assert first.extractions == 2 and first.embeddings == 2 and first.written == 2

    pipe2, llm2, emb2 = make_pipeline(store=store)
    second = pipe2.run(batch)
    assert second.unchanged == 2
    assert second.extractions == 0 and second.embeddings == 0
    assert llm2.calls == [] and emb2.embed_calls == 0


def test_a_changed_course_is_re_extracted():
    store = FakeCourseStore()
    make_pipeline(store=store)[0].run([course("https://c.test/a")])
    pipe2, _, _ = make_pipeline(store=store)
    s = pipe2.run([course("https://c.test/a", body="An entirely rewritten Python and machine learning syllabus.")])
    assert s.changed == 1 and s.extractions == 1


# ---------------------------------------------------------------------------
# quality gate (replaces legitimacy)
# ---------------------------------------------------------------------------
def test_a_course_with_no_extractable_skills_is_rejected():
    """No skill vocabulary in the text -> the fake extracts nothing -> the
    quality gate rejects it (kept for audit, no embedding, out of stats)."""
    store = FakeCourseStore()
    pipe, _, _ = make_pipeline(store=store)
    s = pipe.run([course("https://c.test/empty", name="An Inspiring Journey",
                         body="A motivational experience about personal growth.")])
    assert s.rejected == 1 and s.embeddings == 0
    row = next(iter(store.rows.values()))
    assert row.status == "rejected" and row.taught_skills == []


def test_a_course_with_skills_is_kept_and_embedded():
    store = FakeCourseStore()
    pipe, _, _ = make_pipeline(store=store)
    s = pipe.run([course("https://c.test/py")])
    assert s.rejected == 0 and s.embeddings == 1
    row = next(iter(store.rows.values()))
    assert row.status == "active" and "Python" in row.taught_skills


# ---------------------------------------------------------------------------
# near-dup
# ---------------------------------------------------------------------------
def test_in_group_reruns_of_one_course_auto_merge():
    """Identical essence within one provider group -> same vector -> merge."""
    store = FakeCourseStore()
    pipe, _, _ = make_pipeline(store=store)
    body = "Learn Python and SQL from the basics."
    s = pipe.run([
        course("https://c.test/one", body=body),
        course("https://c.test/two", body=body),
    ])
    canonical = [r for r in store.rows.values() if r.duplicate_of is None]
    merged = [r for r in store.rows.values() if r.duplicate_of is not None]
    assert len(canonical) == 1 and len(merged) == 1
    assert s.embed_duplicates == 1
    assert merged[0].duplicate_of == canonical[0].course_id


def test_cross_group_near_duplicates_go_to_review_never_merge():
    store = FakeCourseStore()
    pipe, _, _ = make_pipeline(store=store)
    body = "Learn Python and SQL from the basics."
    s = pipe.run([
        course("https://c.test/one", body=body),
        course("https://f.test/two", source="freecodecamp", group="freecodecamp",
               stype="html_scrape", body=body),
    ])
    assert s.needs_review == 1 and s.embed_duplicates == 0
    review = [r for r in store.rows.values() if r.status == "needs_review"][0]
    assert review.duplicate_of is None and review.review_reason == "cross_group_duplicate"


def test_three_way_reruns_collapse_to_one_root_not_a_chain():
    store = FakeCourseStore()
    pipe, _, _ = make_pipeline(store=store)
    body = "Learn Python and SQL from the basics."
    s = pipe.run([course(f"https://c.test/{i}", body=body) for i in range(3)])
    canonical = [r for r in store.rows.values() if r.duplicate_of is None]
    merged = [r for r in store.rows.values() if r.duplicate_of is not None]
    assert len(canonical) == 1 and len(merged) == 2
    root = canonical[0].course_id
    assert all(m.duplicate_of == root for m in merged)


def test_within_batch_duplicate_urls_collapse():
    store = FakeCourseStore()
    pipe, _, _ = make_pipeline(store=store)
    s = pipe.run([course("https://c.test/a"), course("https://c.test/a")])
    assert s.received == 2 and s.written == 1
