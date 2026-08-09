"""A source can be told to contribute only postings that reach a job offer.

**User decision 2026-08-09: "I only want final destinations."** Measured by
tracing 30 el7far postings to their end — 16 carried no external link at all
("send your CV to hr@…"), 11 pointed at a careers hub or LinkedIn, 3 were
refused or unreachable, and **0 reached a vacancy page**. Across the full
backfill: 5 destinations from 179 postings.

**The rule lives at the write, and that is the whole point.** Deleting such rows
buys twelve hours — the next cycle finds the posts absent, treats them as new,
and writes every one of them back. `test_a_refused_posting_does_not_come_back_next_cycle`
is the test that matters.

GulfTalent is deliberately exempt: its own URL IS the job ad, and its terms
require us to link there rather than to an employer's site, so demanding a
`final_url` would delete the one source that already satisfies the rule.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agents.agent_b_job_ingest.pipeline import IngestPipeline
from agents.agent_b_job_ingest.prompts.extraction import EXTRACTION_PROMPT
from agents.agent_b_job_ingest.prompts.legitimacy import LEGITIMACY_PROMPT
from agents.agent_b_job_ingest.schemas import (JobExtraction, JobExtractionBatch,
                                               LegitimacyVerdict)
from agents.agent_b_job_ingest.sources.base import RawPosting
from agents.agent_b_job_ingest.sources.config import DEFAULT_SOURCES, SourceConfig
from shared.config import Config
from shared.llm import structured
from tests.agent_b.fake_store import FakeStore
from tests.fake_embedder import FakeEmbedder
from tests.fake_llm import FakeStructuredLLM

STRICT = SourceConfig(name="aggregator", source_group="agg", source_type="blogger_feed",
                      base_url="https://agg.test", terms_reviewed=True,
                      require_destination=True)
LENIENT = SourceConfig(name="board", source_group="board", source_type="html_scrape",
                       base_url="https://board.test", terms_reviewed=True)

JOB_PAGE = "https://careers.acme.test/jobs/welder-8812"


class FakeFetcher:
    def __init__(self, pages): self.pages = pages
    def fetch(self, url): return self.pages.get(url)


class RootLLM(FakeStructuredLLM):
    def respond(self, schema, payload):
        if schema.__name__ == "JobExtractionBatch":
            if "WELDING" in str(payload):          # the destination page
                return JobExtractionBatch(jobs=[JobExtraction(
                    sector="7", required_skills=["welding", "fabrication"])])
            return JobExtractionBatch(jobs=[JobExtraction(
                sector="7", required_skills=["general labour"])])
        return super().respond(schema, payload)


def posting(source="aggregator", links=()):
    return RawPosting(
        source=source, source_group=f"{source}_grp",
        source_type="blogger_feed" if source == "aggregator" else "html_scrape",
        source_url=f"https://{source}.test/2026/08/welder.html",
        title="Welder wanted",
        raw_description="A welder is needed in Muscat. Send your CV to hr@acme.test.",
        posted_date=datetime(2026, 8, 1, tzinfo=timezone.utc),
        outbound_links=links,
    )


def build(store, *, pages=None, configs=(STRICT, LENIENT)):
    llm = RootLLM()
    return IngestPipeline(
        store=store,
        extractor=EXTRACTION_PROMPT | structured(llm, JobExtractionBatch),
        adjudicator=LEGITIMACY_PROMPT | structured(llm, LegitimacyVerdict),
        embedder=FakeEmbedder(), config=Config(), model_name="fake",
        root_fetcher=FakeFetcher(pages or {}),
        source_configs=configs,
    )


# ---------------------------------------------------------------------------
def test_a_posting_that_reaches_no_job_page_is_never_stored():
    """The apply-by-email article: 53% of el7far's postings, measured."""
    store = FakeStore()
    summary = build(store).run([posting()])

    assert store.rows == {}
    assert summary.skipped_no_destination == 1


def test_a_posting_that_reaches_one_is_stored_with_it():
    store = FakeStore()
    summary = build(store, pages={JOB_PAGE: "WELDING role at Acme. Fabrication required."}
                    ).run([posting(links=(JOB_PAGE,))])

    row = next(iter(store.rows.values()))
    assert row.final_url == JOB_PAGE
    assert row.required_skills == ["welding", "fabrication"]
    assert summary.skipped_no_destination == 0


def test_a_refused_posting_does_not_come_back_next_cycle():
    """THE test. A delete alone buys twelve hours: the next cycle sees the post
    as new and writes it straight back. The rule has to live at the write."""
    store = FakeStore()
    for _ in range(3):
        build(store).run([posting()])

    assert store.rows == {}, "a refused posting must not reappear on any later cycle"


def test_a_source_without_the_flag_is_untouched():
    """GulfTalent's case: its own URL is the job ad, so requiring a `final_url`
    would delete the one source that fully satisfies the rule."""
    store = FakeStore()
    summary = build(store).run([posting(source="board")])

    assert len(store.rows) == 1
    assert next(iter(store.rows.values())).final_url is None
    assert summary.skipped_no_destination == 0


def test_the_rule_comes_from_the_injected_registry_not_a_global():
    """A pipeline built directly — every offline test — must not inherit
    production semantics from whatever fixture name it happened to pick. The
    24 tests using `source="el7far"` as a stand-in are why this is injected."""
    store = FakeStore()
    summary = build(store, configs=()).run([posting()])

    assert len(store.rows) == 1, "no registry supplied means no obligations"
    assert summary.skipped_no_destination == 0


def test_the_real_registry_says_what_we_think_it_says():
    """Pins the production decision itself, so flipping a flag is a visible
    change rather than a quiet one."""
    by_name = {c.name: c for c in DEFAULT_SOURCES}
    assert by_name["el7far"].require_destination is True
    assert by_name["tg_omanjob1"].require_destination is True
    # Exempt, and it must stay exempt — see the module docstring.
    assert by_name["gulftalent"].require_destination is False


def test_refusal_is_counted_not_silent():
    """A source contributing 0 postings and a source contributing 180 look
    identical in a row count. The counter is how an operator tells them apart."""
    store = FakeStore()
    summary = build(store).run([posting(), posting(source="board")])

    assert summary.skipped_no_destination == 1
    assert len(store.rows) == 1
