"""Root-source enrichment: replace a posting's thin aggregator skills with the
real job page's — bounded, robots-respecting, enrich-only.

A FakeRootFetcher serves canned job-page text by URL (no network). The link
heuristic decides WHICH link is fetched; the batch extractor decides whether the
fetched page is a single job (enrich) or a listing (skip).
"""

from __future__ import annotations

from datetime import datetime, timezone

from agents.agent_b_job_ingest.pipeline import IngestPipeline
from agents.agent_b_job_ingest.prompts.extraction import EXTRACTION_PROMPT
from agents.agent_b_job_ingest.prompts.legitimacy import LEGITIMACY_PROMPT
from agents.agent_b_job_ingest.root_fetch import candidate_job_link
from agents.agent_b_job_ingest.schemas import (
    JobExtraction,
    JobExtractionBatch,
    LegitimacyVerdict,
)
from agents.agent_b_job_ingest.sources.base import RawPosting
from shared.config import Config
from shared.llm import structured
from tests.agent_b.fake_store import FakeStore
from tests.fake_embedder import FakeEmbedder
from tests.fake_llm import FakeStructuredLLM


class FakeRootFetcher:
    def __init__(self, pages: dict[str, str]):
        self.pages = pages
        self.fetched: list[str] = []

    def fetch(self, url: str) -> str | None:
        self.fetched.append(url)
        return self.pages.get(url)


def make_pipeline(store, *, root_fetcher=None, llm=None, config=None, **overrides):
    llm = llm or FakeStructuredLLM(**overrides)
    return IngestPipeline(
        store=store,
        extractor=EXTRACTION_PROMPT | structured(llm, JobExtractionBatch),
        adjudicator=LEGITIMACY_PROMPT | structured(llm, LegitimacyVerdict),
        embedder=FakeEmbedder(),
        config=config or Config(),
        model_name="fake",
        root_fetcher=root_fetcher,
    )


def posting(links=(), body=None):
    return RawPosting(
        source="el7far", source_group="el7far_network", source_type="blogger_feed",
        source_url="https://oman.el7far.com/2026/07/eni-well-engineer.html",
        title="Eni – Well Engineer",
        raw_description=body or "Eni is hiring a Well Engineer. Apply on the company site.",
        posted_date=datetime(2026, 7, 20, tzinfo=timezone.utc),
        outbound_links=links,
    )


JOB_URL = "https://jobs.eni.com/en/sites/CX_1004/job/33730"
ROOT_TEXT = "Well Engineer at Eni. Requirements: well planning, drilling operations, WELLPLAN."


# ---------------------------------------------------------------------------
def test_root_page_skills_replace_the_thin_aggregator_skills():
    store = FakeStore()
    fetcher = FakeRootFetcher({JOB_URL: ROOT_TEXT})
    # el7far extraction is thin; the root page extraction is rich.
    llm = FakeStructuredLLM(JobExtractionBatch=None)  # placeholder; overridden below

    # First call (el7far body) and the root re-extract both hit JobExtractionBatch;
    # return the ROOT skills for the root text, thin for the aggregator text.
    class TwoFaced(FakeStructuredLLM):
        def respond(self, schema, payload):
            if schema.__name__ == "JobExtractionBatch":
                body = str(payload)
                if "WELLPLAN" in body:
                    return JobExtractionBatch(jobs=[JobExtraction(
                        sector="2", required_skills=["well planning", "drilling operations", "WELLPLAN"],
                        location="Muscat", country="OM")])
                return JobExtractionBatch(jobs=[JobExtraction(sector="2", required_skills=["oil and gas"])])
            return super().respond(schema, payload)

    pipe = make_pipeline(store, root_fetcher=fetcher, llm=TwoFaced())
    summary = pipe.run([posting(links=(JOB_URL,))])

    assert fetcher.fetched == [JOB_URL]
    assert summary.root_enrichments == 1
    row = next(iter(store.rows.values()))
    assert row.required_skills == ["well planning", "drilling operations", "WELLPLAN"]
    assert row.location == "Muscat"


def test_a_hub_page_is_not_deep_crawled():
    """If the fetched page extracts to MULTIPLE jobs it was a listing/hub — skip,
    keep the aggregator's skills, never split external content."""
    store = FakeStore()
    fetcher = FakeRootFetcher({JOB_URL: "many jobs listed here"})

    class HubLLM(FakeStructuredLLM):
        def respond(self, schema, payload):
            if schema.__name__ == "JobExtractionBatch" and "many jobs" in str(payload):
                return JobExtractionBatch(jobs=[
                    JobExtraction(required_skills=["a"]), JobExtraction(required_skills=["b"]),
                ])
            return super().respond(schema, payload)

    pipe = make_pipeline(store, root_fetcher=fetcher, llm=HubLLM())
    summary = pipe.run([posting(links=(JOB_URL,))])

    assert summary.root_enrichments == 0
    row = next(iter(store.rows.values()))
    assert "well planning" not in row.required_skills   # unchanged, aggregator skills kept


def test_no_candidate_link_means_no_fetch():
    store = FakeStore()
    fetcher = FakeRootFetcher({})
    # links are a hub + social — neither is a single job page
    pipe = make_pipeline(store, root_fetcher=fetcher)
    pipe.run([posting(links=("https://jobs.caa.gov.om/", "https://wa.me/123"))])
    assert fetcher.fetched == []


def test_a_blocked_or_missing_page_leaves_skills_and_never_deletes():
    store = FakeStore()
    fetcher = FakeRootFetcher({})   # fetch(JOB_URL) -> None (robots-deny / 404 / error)
    pipe = make_pipeline(store, root_fetcher=fetcher)
    summary = pipe.run([posting(links=(JOB_URL,))])
    assert fetcher.fetched == [JOB_URL]
    assert summary.root_enrichments == 0
    assert summary.written == 1                 # row still written, never dropped


def test_disabled_config_skips_enrichment():
    store = FakeStore()
    fetcher = FakeRootFetcher({JOB_URL: ROOT_TEXT})
    pipe = make_pipeline(store, root_fetcher=fetcher,
                         config=Config(enrich_from_root_source=False))
    pipe.run([posting(links=(JOB_URL,))])
    assert fetcher.fetched == []


def test_no_fetcher_is_a_no_op():
    store = FakeStore()
    pipe = make_pipeline(store, root_fetcher=None)
    summary = pipe.run([posting(links=(JOB_URL,))])
    assert summary.root_fetches == 0 and summary.root_enrichments == 0


def test_split_children_are_not_root_enriched():
    """A roundup's split children share the post's hub links, not per-role pages,
    so enrichment must skip them."""
    store = FakeStore()
    fetcher = FakeRootFetcher({JOB_URL: ROOT_TEXT})
    three = JobExtractionBatch(jobs=[
        JobExtraction(title="A", required_skills=["x"]),
        JobExtraction(title="B", required_skills=["y"]),
        JobExtraction(title="C", required_skills=["z"]),
    ])
    pipe = make_pipeline(store, root_fetcher=fetcher, JobExtractionBatch=three)
    pipe.run([posting(links=(JOB_URL,), body="Hiring A, B and C. Apply online.")])
    # 3 split children; none should trigger a root fetch
    assert fetcher.fetched == []
    assert summary_written(store) == 3


def summary_written(store):
    return len(store.rows)


def test_candidate_link_prefers_job_detail_over_hub_and_social():
    links = ("https://wa.me/1", "https://acme.com/careers",
             "https://jobs.acme.com/job/42", "https://jobs.acme.com/job/99")
    assert candidate_job_link(links, "oman.el7far.com") == "https://jobs.acme.com/job/42"
