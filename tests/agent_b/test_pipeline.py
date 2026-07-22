"""The ingestion tail, offline against fakes.

The phase-4 gate lives in the first test: a second run over the same postings
must do zero LLM and zero embedding work. Everything else here guards a specific
way the tail could quietly do the wrong thing — reject-then-still-extract,
merge-then-embed, invert the duplicate direction.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agents.agent_b_job_ingest.pipeline import IngestPipeline
from agents.agent_b_job_ingest.prompts.extraction import EXTRACTION_PROMPT
from agents.agent_b_job_ingest.prompts.legitimacy import LEGITIMACY_PROMPT
from agents.agent_b_job_ingest.schemas import JobExtraction, LegitimacyVerdict
from agents.agent_b_job_ingest.sources.base import RawPosting
from shared.config import Config
from shared.llm import structured
from tests.agent_b.fake_store import FakeStore
from tests.fake_embedder import FakeEmbedder
from tests.fake_llm import FakeStructuredLLM


def make_pipeline(store=None, llm=None, embedder=None, **llm_overrides):
    llm = llm or FakeStructuredLLM(**llm_overrides)
    embedder = embedder or FakeEmbedder()
    return IngestPipeline(
        store=store or FakeStore(),
        extractor=EXTRACTION_PROMPT | structured(llm, JobExtraction),
        adjudicator=LEGITIMACY_PROMPT | structured(llm, LegitimacyVerdict),
        embedder=embedder,
        config=Config(),
        model_name="fake",
    ), llm, embedder


def posting(pid_url: str, *, source="el7far", group="el7far_network", stype="blogger_feed",
            title="Software Engineer at Example Engineering Co",
            body=None, links=(), intent="unknown", poster="unknown") -> RawPosting:
    return RawPosting(
        source=source,
        source_group=group,
        source_type=stype,
        source_url=pid_url,
        title=title,
        raw_description=body or (
            "Example Engineering Co is hiring a Software Engineer in Muscat. "
            "Responsibilities include building services. Requirements: three years "
            "of experience with Python and SQL. Apply at careers@example.test"
        ),
        posted_date=datetime(2026, 7, 20, tzinfo=timezone.utc),
        outbound_links=links,
        listing_intent=intent,
        poster_type=poster,
    )


# ---------------------------------------------------------------------------
# THE GATE
# ---------------------------------------------------------------------------
def test_a_second_run_does_zero_llm_and_zero_embedding_work():
    """The property the whole ordering exists to produce. Without it a 12-hour
    cycle would re-extract and re-embed the entire corpus every time."""
    store = FakeStore()
    pipe, llm, embedder = make_pipeline(store=store)
    batch = [posting("https://x.test/a"), posting("https://x.test/b", title="Analyst role")]

    first = pipe.run(batch)
    assert first.extractions == 2
    assert first.embeddings == 2
    assert first.written == 2

    pipe2, llm2, embedder2 = make_pipeline(store=store)
    second = pipe2.run(batch)

    assert second.unchanged == 2
    assert second.extractions == 0
    assert second.embeddings == 0
    assert llm2.calls == []
    assert embedder2.embed_calls == 0


def test_a_changed_posting_is_re_extracted_but_an_unchanged_one_is_not():
    store = FakeStore()
    pipe, _, _ = make_pipeline(store=store)
    pipe.run([posting("https://x.test/a"), posting("https://x.test/b")])

    pipe2, llm2, embedder2 = make_pipeline(store=store)
    summary = pipe2.run([
        posting("https://x.test/a"),  # unchanged
        posting("https://x.test/b", body="An entirely rewritten description with new duties."),
    ])

    assert summary.unchanged == 1
    assert summary.changed == 1
    assert summary.extractions == 1
    assert summary.embeddings == 1


# ---------------------------------------------------------------------------
# legitimacy gate
# ---------------------------------------------------------------------------
def test_a_rejected_posting_costs_no_extraction_and_no_embedding():
    """Rejected rows are stored for audit but empty of everything expensive —
    the concrete payoff for running legitimacy before extraction."""
    scam = FakeStructuredLLM(
        LegitimacyVerdict=LegitimacyVerdict(
            is_scam=True,
            evidence_quote="registration fee",
            reasoning="demands an upfront fee",
            scam_confidence=0.95,
        )
    )
    store = FakeStore()
    pipe, llm, embedder = make_pipeline(store=store, llm=scam)

    body = "Many vacancies available. Pay a registration fee to secure your position. WhatsApp only."
    summary = pipe.run([posting("https://x.test/scam", body=body)])

    assert summary.rejected == 1
    assert summary.extractions == 0
    assert summary.embeddings == 0
    assert store.rows["".join(store.rows.keys())].status == "rejected"
    row = next(iter(store.rows.values()))
    assert row.sector is None and row.embedding is None
    assert row.raw_description  # retained for audit


def test_a_clean_posting_is_never_sent_to_the_adjudicator():
    store = FakeStore()
    pipe, llm, _ = make_pipeline(store=store)
    summary = pipe.run([posting("https://x.test/clean")])

    assert summary.adjudications == 0
    assert "LegitimacyVerdict" not in llm.calls


def test_a_fabricated_adjudicator_quote_is_discarded():
    """A model verdict is trusted only if its quote is real; otherwise the
    deterministic rule score stands. The model does not get to reject a posting
    on evidence it cannot produce."""
    liar = FakeStructuredLLM(
        LegitimacyVerdict=LegitimacyVerdict(
            is_scam=True,
            evidence_quote="a quote that is nowhere in the posting",
            reasoning="invented",
            scam_confidence=0.99,
        )
    )
    store = FakeStore()
    # A body that lands in the adjudicate band: whatsapp-only + no role detail.
    body = "Urgent hiring now. Contact us on WhatsApp only."
    pipe, _, _ = make_pipeline(store=store, llm=liar)
    summary = pipe.run([posting("https://x.test/borderline", body=body)])

    # The rule risk alone did not exceed the reject threshold, so a discarded
    # fabricated verdict leaves the posting un-rejected.
    assert summary.rejected == 0


# ---------------------------------------------------------------------------
# link dedup — deterministic, before embedding
# ---------------------------------------------------------------------------
def test_a_posting_that_links_to_another_is_the_duplicate():
    """Direction rule: the linker is the duplicate, the link target is
    canonical. The blog posting is ingested first; the Telegram post that links
    to it resolves by URL and is never embedded."""
    store = FakeStore()
    pipe, _, _ = make_pipeline(store=store)
    pipe.run([posting("https://oman.el7far.com/job.html")])

    pipe2, _, embedder2 = make_pipeline(store=store)
    summary = pipe2.run([
        posting(
            "https://t.me/omanjob1/1",
            source="tg_omanjob1",
            group="el7far_network",
            stype="telegram",
            links=("https://oman.el7far.com/job.html",),
        )
    ])

    assert summary.link_duplicates == 1
    assert summary.embeddings == 0, "a link-confirmed duplicate must not be embedded"
    tg_id = next(pid for pid, r in store.rows.items() if r.source == "tg_omanjob1")
    blog_id = next(pid for pid, r in store.rows.items() if r.source == "el7far")
    assert store.rows[tg_id].duplicate_of == blog_id


def test_only_the_first_outbound_link_resolves_a_duplicate():
    """A Telegram post appends an unrelated next job as a second link. Merging on
    any link would collapse two distinct vacancies into one."""
    store = FakeStore()
    pipe, _, _ = make_pipeline(store=store)
    pipe.run([posting("https://oman.el7far.com/second-unrelated.html")])

    pipe2, _, _ = make_pipeline(store=store)
    summary = pipe2.run([
        posting(
            "https://t.me/omanjob1/2",
            source="tg_omanjob1",
            group="el7far_network",
            stype="telegram",
            links=(
                "https://oman.el7far.com/its-own-subject.html",   # not in store
                "https://oman.el7far.com/second-unrelated.html",  # in store, but second
            ),
        )
    ])

    assert summary.link_duplicates == 0, "merged on a non-subject link"


def test_the_canonical_is_inserted_before_its_in_cycle_duplicate():
    """When both are new this cycle, the FK requires the canonical to be written
    first. The FakeStore asserts this the same way the real transaction would."""
    store = FakeStore()
    pipe, _, _ = make_pipeline(store=store)

    blog = posting("https://oman.el7far.com/same-job.html")
    tg = posting(
        "https://t.me/omanjob1/3", source="tg_omanjob1", group="el7far_network",
        stype="telegram", links=("https://oman.el7far.com/same-job.html",),
    )
    # Duplicate listed first in the batch, to prove ordering is by FK not input.
    summary = pipe.run([tg, blog])

    assert summary.link_duplicates == 1
    assert summary.written == 2


# ---------------------------------------------------------------------------
# near-dup — embedding similarity
# ---------------------------------------------------------------------------
def test_in_group_near_duplicates_auto_merge():
    """Identical text within one publisher's group → same vector → merge."""
    store = FakeStore()
    pipe, _, _ = make_pipeline(store=store)
    body = "Example Engineering Co seeks a Data Analyst. Requirements: SQL and Python. Muscat."

    summary = pipe.run([
        posting("https://x.test/one", body=body),
        posting("https://x.test/two", body=body),  # same group, same text
    ])

    merged = [r for r in store.rows.values() if r.duplicate_of is not None]
    canonical = [r for r in store.rows.values() if r.duplicate_of is None]
    assert len(merged) == 1 and len(canonical) == 1, "exactly one should become the duplicate"
    assert summary.embed_duplicates == 1
    assert merged[0].duplicate_of == canonical[0].posting_id


def test_cross_group_near_duplicates_go_to_review_never_auto_merge():
    """A wrong cross-publisher merge erases a real independent demand signal, so
    even at high similarity it is flagged for a human and duplicate_of stays
    NULL."""
    store = FakeStore()
    pipe, _, _ = make_pipeline(store=store)
    body = "Example Engineering Co seeks a Data Analyst. Requirements: SQL and Python. Muscat."

    summary = pipe.run([
        posting("https://x.test/one", body=body),
        posting("https://y.test/two", source="dubizzle", group="dubizzle",
                stype="html_scrape", body=body),
    ])

    assert summary.needs_review == 1
    assert summary.embed_duplicates == 0
    review = [r for r in store.rows.values() if r.status == "needs_review"][0]
    assert review.duplicate_of is None
    assert review.review_reason == "cross_group_duplicate"


def test_three_way_near_duplicates_collapse_to_one_root_not_a_chain():
    """The bug the FIRST FULL LIVE RUN found. With three similar postings, the
    near-dup pass can set A->B before B itself merges into C, leaving A pointing
    at another duplicate. The upsert sorts canonicals first, so A's FK fires
    before B exists and the whole batch dies mid-cycle. Every duplicate must
    point at a ROOT."""
    store = FakeStore()
    pipe, _, _ = make_pipeline(store=store)
    body = "Example Engineering Co seeks a Data Analyst. Requirements: SQL and Python. Muscat."

    summary = pipe.run([
        posting("https://x.test/one", body=body),
        posting("https://x.test/two", body=body),
        posting("https://x.test/three", body=body),
    ])

    canonical = [r for r in store.rows.values() if r.duplicate_of is None]
    merged = [r for r in store.rows.values() if r.duplicate_of is not None]
    assert len(canonical) == 1 and len(merged) == 2
    root = canonical[0].posting_id
    for row in merged:
        assert row.duplicate_of == root, (
            f"{row.posting_id} points at {row.duplicate_of}, not the root — a chain survived"
        )
    assert summary.embed_duplicates == 2


def test_unrelated_postings_are_not_merged():
    """Distinct jobs differ in the essence that gets embedded — title, skills —
    so they must land as two canonicals. (Since the first live run, similarity
    runs on the extracted essence, not the full description: shared page
    template must never be what makes two jobs 'similar'.)"""
    store = FakeStore()
    pipe, _, _ = make_pipeline(store=store)
    summary = pipe.run([
        posting("https://x.test/one", title="Warehouse Supervisor",
                body="A warehouse role in Sohar handling logistics."),
        posting("https://x.test/two", title="Pastry Chef",
                body="A pastry chef position in a Muscat hotel kitchen."),
    ])

    assert summary.embed_duplicates == 0
    assert summary.needs_review == 0
    assert all(r.duplicate_of is None for r in store.rows.values())


# ---------------------------------------------------------------------------
# extraction grounding
# ---------------------------------------------------------------------------
def test_an_ungrounded_company_claim_does_not_make_a_posting_aggregable():
    """The model asserting poster_type='company' is trusted only if it also
    names an employer that appears in the text. A hallucinated employer must not
    quietly promote a posting into the aggregable set.

    company is not a stored column (the two-table contract omits it), so the
    observable effect is on poster_type."""
    liar = FakeStructuredLLM(
        JobExtraction=JobExtraction(
            sector="2",
            company="A Company Not In The Text",
            required_skills=["Python"],
            listing_intent="vacancy",
            poster_type="company",
        )
    )
    store = FakeStore()
    pipe, _, _ = make_pipeline(store=store, llm=liar)
    pipe.run([posting("https://x.test/a", body="Some role. Requirements: Python. Apply online.")])

    row = next(iter(store.rows.values()))
    assert row.poster_type == "unknown"


def test_a_grounded_company_is_accepted():
    """When the named employer really appears in the text, poster_type='company'
    stands."""
    ok = FakeStructuredLLM(
        JobExtraction=JobExtraction(
            sector="2",
            company="Example Engineering Co",
            required_skills=["Python"],
            listing_intent="vacancy",
            poster_type="company",
        )
    )
    store = FakeStore()
    pipe, _, _ = make_pipeline(store=store, llm=ok)
    pipe.run([posting("https://x.test/a")])  # body names Example Engineering Co

    row = next(iter(store.rows.values()))
    assert row.poster_type == "company"


def test_within_batch_duplicate_urls_collapse_to_one():
    store = FakeStore()
    pipe, _, _ = make_pipeline(store=store)
    summary = pipe.run([posting("https://x.test/a"), posting("https://x.test/a")])

    assert summary.received == 2
    assert summary.written == 1
