"""The whole ingestion tail against real Postgres, end to end.

The offline pipeline test proves the counters; this proves the same property
survives contact with the database — the ON CONFLICT path, the pgvector round
trip, and the FK-ordered write all participating in one run and then a second.
The gate is the second run doing no expensive work AND the row count not growing.
"""

from __future__ import annotations

from datetime import datetime, timezone

from agents.agent_b_job_ingest.pipeline import IngestPipeline
from agents.agent_b_job_ingest.prompts.extraction import EXTRACTION_PROMPT
from agents.agent_b_job_ingest.prompts.legitimacy import LEGITIMACY_PROMPT
from agents.agent_b_job_ingest.schemas import JobExtraction, LegitimacyVerdict
from agents.agent_b_job_ingest.sources.base import RawPosting
from shared.config import Config
from shared.llm import structured
from tests.fake_embedder import FakeEmbedder
from tests.fake_llm import FakeStructuredLLM


def _pipeline(store, llm, embedder):
    return IngestPipeline(
        store=store,
        extractor=EXTRACTION_PROMPT | structured(llm, JobExtraction),
        adjudicator=LEGITIMACY_PROMPT | structured(llm, LegitimacyVerdict),
        embedder=embedder,
        config=Config(),
        model_name="fake",
    )


def _posting(url, **kw):
    return RawPosting(
        source=kw.pop("source", "el7far"),
        source_group=kw.pop("source_group", "el7far_network"),
        source_type=kw.pop("source_type", "blogger_feed"),
        source_url=url,
        title=kw.pop("title", "Software Engineer at Example Engineering Co"),
        raw_description=kw.pop(
            "raw_description",
            "Example Engineering Co is hiring a Software Engineer in Muscat. "
            "Responsibilities include building services. Requirements: Python and SQL.",
        ),
        posted_date=datetime(2026, 7, 20, tzinfo=timezone.utc),
        outbound_links=kw.pop("outbound_links", ()),
    )


def test_two_runs_ingest_once_and_the_second_does_no_work(store):
    llm, embedder = FakeStructuredLLM(), FakeEmbedder()
    batch = [_posting("https://oman.el7far.com/a.html"),
             _posting("https://oman.el7far.com/b.html", title="Data Analyst role")]

    first = _pipeline(store, llm, embedder).run(batch)
    assert first.written == 2
    assert store.counts().get("active", 0) == 2

    llm2, embedder2 = FakeStructuredLLM(), FakeEmbedder()
    second = _pipeline(store, llm2, embedder2).run(batch)

    assert second.unchanged == 2
    assert second.extractions == 0
    assert second.embeddings == 0
    assert llm2.calls == []
    assert embedder2.embed_calls == 0
    # And crucially, no new rows.
    assert store.counts().get("active", 0) == 2


def test_a_three_way_near_duplicate_batch_survives_the_real_fk(store):
    """Regression for the first full live run: a duplicate CHAIN (A->B, B->C)
    passed the fake store but violated the real FK, killing the batch. Three
    identical postings must land as one canonical + two duplicates, committed."""
    llm, embedder = FakeStructuredLLM(), FakeEmbedder()
    body = "Example Co seeks an Analyst. Requirements: SQL. Muscat."
    batch = [
        _posting("https://oman.el7far.com/one.html", raw_description=body),
        _posting("https://oman.el7far.com/two.html", raw_description=body),
        _posting("https://oman.el7far.com/three.html", raw_description=body),
    ]

    summary = _pipeline(store, llm, embedder).run(batch)

    assert summary.written == 3
    conn = store.connect()
    with conn.cursor() as cur:
        cur.execute("SELECT posting_id, duplicate_of FROM job_postings")
        rows = cur.fetchall()
    roots = [r for r in rows if r["duplicate_of"] is None]
    dups = [r for r in rows if r["duplicate_of"] is not None]
    assert len(roots) == 1 and len(dups) == 2
    assert all(d["duplicate_of"] == roots[0]["posting_id"] for d in dups)


def test_the_telegram_blog_pair_resolves_by_link_not_embedding(store):
    """The phase-7 acceptance criterion, provable now: a Telegram post linking to
    its blog original is merged deterministically and never embedded."""
    llm, embedder = FakeStructuredLLM(), FakeEmbedder()
    _pipeline(store, llm, embedder).run([_posting("https://oman.el7far.com/job.html")])

    llm2, embedder2 = FakeStructuredLLM(), FakeEmbedder()
    summary = _pipeline(store, llm2, embedder2).run([
        _posting(
            "https://t.me/omanjob1/1",
            source="tg_omanjob1", source_group="el7far_network", source_type="telegram",
            outbound_links=("https://oman.el7far.com/job.html",),
        )
    ])

    assert summary.link_duplicates == 1
    assert summary.embeddings == 0

    conn = store.connect()
    with conn.cursor() as cur:
        cur.execute("SELECT posting_id FROM job_postings WHERE source = 'el7far'")
        blog_id = cur.fetchone()["posting_id"]
        cur.execute("SELECT duplicate_of FROM job_postings WHERE source = 'tg_omanjob1'")
        assert cur.fetchone()["duplicate_of"] == blog_id, "telegram row must point at the blog row"
