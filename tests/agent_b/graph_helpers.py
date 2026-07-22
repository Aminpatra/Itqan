"""Fakes for driving the ingestion graph without a network.

The graph's real adapters fetch over HTTP; these replace them with canned
``AdapterResult``s so the whole cycle — fan-out, fan-in, per-source-batch
transactions, staleness, aggregation, run log — can be exercised against a real
database but no live site.
"""

from __future__ import annotations

from datetime import datetime, timezone

from agents.agent_b_job_ingest.nodes import GraphDeps
from agents.agent_b_job_ingest.prompts.extraction import EXTRACTION_PROMPT
from agents.agent_b_job_ingest.prompts.legitimacy import LEGITIMACY_PROMPT
from agents.agent_b_job_ingest.schemas import JobExtraction, LegitimacyVerdict
from agents.agent_b_job_ingest.sources.base import AdapterResult, RawPosting
from agents.agent_b_job_ingest.sources.config import SourceConfig
from shared.config import Config
from shared.llm import structured
from tests.fake_embedder import FakeEmbedder
from tests.fake_llm import FakeStructuredLLM


def raw(url, *, source, group, stype, title="Software Engineer at Example Engineering Co",
        body=None, links=()):
    return RawPosting(
        source=source, source_group=group, source_type=stype, source_url=url,
        title=title,
        raw_description=body or (
            "Example Engineering Co is hiring a Software Engineer in Muscat. "
            "Requirements: Python and SQL. Apply at careers@example.test"
        ),
        posted_date=datetime(2026, 7, 20, tzinfo=timezone.utc),
        outbound_links=links,
    )


class FakeAdapter:
    def __init__(self, name, *, postings, error=None, partial=False, skipped=0):
        self.name = name
        self._postings = postings
        self._error = error
        self._partial = partial
        self._skipped = skipped

    def fetch(self, *, limit=None):
        result = AdapterResult(source=self.name)
        result.postings = list(self._postings)[: limit] if limit else list(self._postings)
        result.skipped = self._skipped
        result.pages_fetched = 1
        result.bytes_fetched = 1000
        if self._error is not None:
            result.error = self._error
            result.partial = self._partial
        return result


# Two valid, unique source configs the planner will accept.
BLOG = SourceConfig(name="blog", source_group="g1", source_type="blogger_feed",
                    base_url="https://blog.test")
TG = SourceConfig(name="tg", source_group="g1", source_type="telegram",
                  handle="@testchannel", terms_reviewed=True)


def make_deps(store, adapters: dict[str, FakeAdapter], *, source_configs=(BLOG, TG)):
    """A GraphDeps wired with a real store, fake adapters, fake LLM + embedder."""
    llm = FakeStructuredLLM()
    return GraphDeps(
        config=Config(),
        store=store,
        extractor=EXTRACTION_PROMPT | structured(llm, JobExtraction),
        adjudicator=LEGITIMACY_PROMPT | structured(llm, LegitimacyVerdict),
        embedder=FakeEmbedder(),
        model_name="fake",
        source_configs=source_configs,
        adapter_for=lambda cfg, config: adapters[cfg.name],
    ), llm
