"""A single roundup post that advertises several vacancies becomes one row per
vacancy, each with its OWN skills — not one merged posting.

Drives the pipeline with a batch extractor (as the live runner does) and a fake
LLM that returns several jobs for one post. Uses THREE jobs deliberately: the
count is whatever the post yields, never assumed.
"""

from __future__ import annotations

from datetime import datetime, timezone

from agents.agent_b_job_ingest.pipeline import IngestPipeline
from agents.agent_b_job_ingest.prompts.extraction import EXTRACTION_PROMPT
from agents.agent_b_job_ingest.prompts.legitimacy import LEGITIMACY_PROMPT
from agents.agent_b_job_ingest.schemas import JobExtraction, JobExtractionBatch, LegitimacyVerdict
from agents.agent_b_job_ingest.sources.base import RawPosting
from shared.config import Config
from shared.llm import structured
from tests.agent_b.fake_store import FakeStore
from tests.fake_embedder import FakeEmbedder
from tests.fake_llm import FakeStructuredLLM

ROUNDUP_BODY = (
    "Digital Mall Oman is hiring for several roles. "
    "Accountant: requires bookkeeping and Excel. "
    "Storekeeper: requires inventory management. "
    "Driver: requires a valid driving licence. Apply by email."
)

THREE_JOBS = JobExtractionBatch(jobs=[
    JobExtraction(title="Accountant", sector="2", required_skills=["Bookkeeping", "Excel"],
                  listing_intent="vacancy", poster_type="unknown"),
    JobExtraction(title="Storekeeper", sector="4", required_skills=["Inventory Management"],
                  listing_intent="vacancy", poster_type="unknown"),
    JobExtraction(title="Driver", sector="8", required_skills=["Driving Licence"],
                  listing_intent="vacancy", poster_type="unknown"),
])


def make_pipeline(store, llm=None, embedder=None, **overrides):
    llm = llm or FakeStructuredLLM(**overrides)
    return IngestPipeline(
        store=store,
        extractor=EXTRACTION_PROMPT | structured(llm, JobExtractionBatch),
        adjudicator=LEGITIMACY_PROMPT | structured(llm, LegitimacyVerdict),
        embedder=embedder or FakeEmbedder(),
        config=Config(),
        model_name="fake",
    ), llm


def roundup(url="https://oman.el7far.com/2026/07/roundup.html", body=ROUNDUP_BODY):
    return RawPosting(
        source="el7far", source_group="el7far_network", source_type="blogger_feed",
        source_url=url, title="Digital Mall Oman Careers 2026: 3 Jobs",
        raw_description=body, posted_date=datetime(2026, 7, 20, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
def test_roundup_splits_into_one_row_per_vacancy_with_distinct_skills():
    store = FakeStore()
    pipe, _ = make_pipeline(store, JobExtractionBatch=THREE_JOBS)
    summary = pipe.run([roundup()])

    assert summary.written == 3, "three vacancies should become three rows"
    by_title = {r.title: r for r in store.rows.values()}
    assert set(by_title) == {"Accountant", "Storekeeper", "Driver"}
    # each row carries ONLY its own skills — not the merged union
    assert by_title["Accountant"].required_skills == ["Bookkeeping", "Excel"]
    assert by_title["Storekeeper"].required_skills == ["Inventory Management"]
    assert by_title["Driver"].required_skills == ["Driving Licence"]
    # one extraction call for the whole post, three rows out
    assert summary.extractions == 1 and summary.embeddings == 3
    # all three share the roundup as their source post; each has a distinct URL/id
    assert {r.source_post_url for r in store.rows.values()} == {roundup().source_url}
    assert len({r.source_url for r in store.rows.values()}) == 3
    assert all(r.source_url.startswith(roundup().source_url + "#") for r in store.rows.values())


def test_a_single_vacancy_post_is_one_row_identical_to_before():
    store = FakeStore()
    pipe, _ = make_pipeline(store)          # default fake -> one-job batch
    summary = pipe.run([roundup(body="Example Engineering Co seeks a Data Analyst. Requirements: SQL.")])
    assert summary.written == 1
    row = next(iter(store.rows.values()))
    # single-vacancy: source_url is the post's own URL (no #fragment), id stable
    assert row.source_url == roundup().source_url
    assert "#" not in row.source_url


def test_warm_cycle_reingests_a_roundup_with_zero_llm_and_embedding():
    store = FakeStore()
    make_pipeline(store, JobExtractionBatch=THREE_JOBS)[0].run([roundup()])

    embedder2 = FakeEmbedder()
    pipe2, llm2 = make_pipeline(store, embedder=embedder2, JobExtractionBatch=THREE_JOBS)
    summary = pipe2.run([roundup()])

    assert summary.unchanged == 3, "all three split rows touched, none rebuilt"
    assert summary.extractions == 0 and summary.embeddings == 0
    assert embedder2.embed_calls == 0 and llm2.calls == []


def test_changed_roundup_is_re_split():
    store = FakeStore()
    make_pipeline(store, JobExtractionBatch=THREE_JOBS)[0].run([roundup()])

    two_jobs = JobExtractionBatch(jobs=THREE_JOBS.jobs[:2])
    pipe2, _ = make_pipeline(store, JobExtractionBatch=two_jobs)
    summary = pipe2.run([roundup(body=ROUNDUP_BODY + " (updated: driver role filled)")])
    assert summary.changed == 1
    assert summary.extractions == 1
    # the two current roles are (re)written; the dropped Driver ages out via
    # staleness (its row is simply not touched this cycle).
    assert summary.written == 2
