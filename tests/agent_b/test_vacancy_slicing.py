"""A vacancy split out of a roundup carries its own text, not the whole article.

MEASURED on the live corpus before this existed: **all 40** el7far roundups
stored ONE body on every child. A row titled *"Accounts Payable In-Charge"* held
all 20,000 characters of *"Majees Technical Services Careers — 38 Job
Opportunities in Oman and India"*; 245 rows, 164 of them pinned at the character
cap, **4.1 MB of duplicated text**, and `count(DISTINCT raw_description) = 1` for
every roundup.

The model quotes each vacancy's span and **the quote is checked against the
posting** — the same rule as `evidence_quote` and Agent A's `source_span`. The
tests that matter most here are the failure ones: every way a slice can be
untrustworthy must end with the row keeping the full body, because this logic
runs over live rows and its worst case has to be the status quo.
"""

from __future__ import annotations

import pytest

from agents.agent_b_job_ingest.pipeline import (MIN_SLICE_CHARS, IngestPipeline,
                                                _verified_slice)
from agents.agent_b_job_ingest.prompts.extraction import EXTRACTION_PROMPT
from agents.agent_b_job_ingest.prompts.legitimacy import LEGITIMACY_PROMPT
from agents.agent_b_job_ingest.schemas import (JobExtraction, JobExtractionBatch,
                                               LegitimacyVerdict)
from agents.agent_b_job_ingest.sources.base import RawPosting
from shared.config import Config
from shared.llm import structured
from tests.agent_b.fake_store import FakeStore
from tests.fake_embedder import FakeEmbedder
from tests.fake_llm import FakeStructuredLLM

ACCOUNTS = ("Accounts Payable In-Charge. Process supplier invoices, run monthly "
            "payment cycles and reconcile vendor statements. Five years in a "
            "similar role and a degree in accounting are required.")
WELDER = ("Welding Supervisor. Supervise a team of eight welders on an EPC site, "
          "sign off on weld quality and maintain the inspection log. Trade "
          "certification and offshore experience essential.")
ROUNDUP = (
    "Majees Technical Services Careers — 38 Job Opportunities in Oman and India\n"
    "Majees Technical Services LLC is recruiting across several disciplines.\n\n"
    f"{ACCOUNTS}\n\n{WELDER}\n\n"
    "To apply, send your CV to careers@majeestech.com quoting the role."
)


def posting() -> RawPosting:
    return RawPosting(
        source="el7far", source_group="el7far_network", source_type="blogger_feed",
        source_url="https://oman.el7far.com/2026/08/majees-38-jobs.html",
        title="Majees Technical Services Careers — 38 Job Opportunities",
        raw_description=ROUNDUP,
    )


# ---------------------------------------------------------------------------
# the verifier itself
# ---------------------------------------------------------------------------
def test_a_quoted_span_present_in_the_posting_is_accepted():
    assert _verified_slice(JobExtraction(vacancy_text=ACCOUNTS), ROUNDUP) == ACCOUNTS


def test_a_paraphrase_is_refused():
    """THE check. The model may POINT at text; it may never compose it. A span
    that is not in the posting is words the source never wrote, and storing it
    would put invented text in the field whose whole job is provenance."""
    invented = ("Accounts Payable In-Charge — an exciting opportunity to join a "
                "market-leading finance team in a fast-paced environment.")
    assert _verified_slice(JobExtraction(vacancy_text=invented), ROUNDUP) is None


def test_a_span_that_is_only_the_title_again_is_refused():
    """The row already stores its title. A span that repeats it and nothing else
    adds no information, so the full body is the better answer.

    This is the check a LENGTH floor was reaching for and getting wrong: at 80
    characters it rejected 37 of 38 correct answers on the real corpus, because
    a roundup that LISTS roles says only "Project Coordinator - ELV / Muscat,
    Oman" about each one — 40 characters and genuinely the whole of it.
    """
    title = "Accounts Payable In-Charge"
    assert _verified_slice(JobExtraction(title=title, vacancy_text=title),
                           ROUNDUP, title) is None


def test_a_terse_listing_entry_is_accepted_because_the_source_is_terse():
    """The measured shape of a real roundup: a list, not paragraphs. The span is
    short because the article is short about that role, and storing it still
    beats storing 38 unrelated jobs."""
    body = ("Majees Technical Services Careers - 38 Job Opportunities\n"
            "Project Coordinator - ELV\n- Muscat, Oman\n"
            "Procurement Assistant\n- Sohar, Oman\n")
    span = "Project Coordinator - ELV\n- Muscat, Oman"
    assert _verified_slice(JobExtraction(title="Project Coordinator - ELV",
                                         vacancy_text=span),
                           body, "Project Coordinator - ELV") == span


def test_returning_the_whole_article_is_refused():
    """The failure that would leave every row clustered while reporting success:
    the model echoes its input back and nothing has been sliced."""
    assert _verified_slice(JobExtraction(vacancy_text=ROUNDUP), ROUNDUP) is None


def test_nothing_quoted_is_the_normal_case_not_an_error():
    assert _verified_slice(JobExtraction(), ROUNDUP) is None
    assert _verified_slice(JobExtraction(vacancy_text="   "), ROUNDUP) is None


def test_whitespace_and_arabic_orthography_do_not_defeat_the_check():
    """`verify_quote` normalises, so a span differing only in spacing still
    verifies. Without this a real quote would be thrown away over a newline —
    and on an Arabic corpus, over a diacritic."""
    spaced = ACCOUNTS.replace(" ", "  ").replace("\n", " ")
    assert _verified_slice(JobExtraction(vacancy_text=spaced), ROUNDUP) is not None


# ---------------------------------------------------------------------------
# through the pipeline
# ---------------------------------------------------------------------------
class Roundup(FakeStructuredLLM):
    """Splits the post into two vacancies, each quoting its own span."""

    def __init__(self, *, first=ACCOUNTS, second=WELDER):
        super().__init__()
        self.first, self.second = first, second

    def respond(self, schema, payload):
        if schema.__name__ == "JobExtractionBatch":
            return JobExtractionBatch(jobs=[
                JobExtraction(title="Accounts Payable In-Charge", sector="2",
                              required_skills=["accounts payable"],
                              vacancy_text=self.first),
                JobExtraction(title="Welding Supervisor", sector="7",
                              required_skills=["welding"],
                              vacancy_text=self.second),
            ])
        return super().respond(schema, payload)


def run(llm) -> tuple[FakeStore, object]:
    store = FakeStore()
    pipe = IngestPipeline(
        store=store,
        extractor=EXTRACTION_PROMPT | structured(llm, JobExtractionBatch),
        adjudicator=LEGITIMACY_PROMPT | structured(llm, LegitimacyVerdict),
        embedder=FakeEmbedder(), config=Config(), model_name="fake",
    )
    summary = pipe.run([posting()])
    return store, summary


def test_each_split_vacancy_stores_only_its_own_text():
    store, summary = run(Roundup())
    by_title = {r.title: r.raw_description for r in store.rows.values()}

    assert by_title["Accounts Payable In-Charge"] == ACCOUNTS
    assert by_title["Welding Supervisor"] == WELDER
    # The point of the whole exercise: the accounts row says nothing about welding.
    assert "Welding Supervisor" not in by_title["Accounts Payable In-Charge"]
    assert "38 Job Opportunities" not in by_title["Accounts Payable In-Charge"]
    assert summary.unsliced_vacancies == 0


def test_an_unverifiable_span_leaves_the_row_exactly_as_it_was():
    """The safety property. Every failure path ends in "keep the full body",
    which is why this can run over live rows: the worst case is the status quo."""
    store, summary = run(Roundup(first="Invented prose the posting never contained at all."))
    by_title = {r.title: r.raw_description for r in store.rows.values()}

    assert by_title["Accounts Payable In-Charge"] == ROUNDUP   # unchanged
    assert by_title["Welding Supervisor"] == WELDER            # the good one still sliced
    assert summary.unsliced_vacancies == 1


def test_a_single_vacancy_post_keeps_its_whole_body():
    """Nothing to slice: the body IS about that vacancy, and cutting it could
    only lose context."""
    class Single(FakeStructuredLLM):
        def respond(self, schema, payload):
            if schema.__name__ == "JobExtractionBatch":
                return JobExtractionBatch(jobs=[JobExtraction(
                    sector="2", required_skills=["accounts payable"],
                    # Even if the model volunteers a span, a single-vacancy post
                    # must not be narrowed.
                    vacancy_text=ACCOUNTS)])
            return super().respond(schema, payload)

    store, _ = run(Single())
    assert next(iter(store.rows.values())).raw_description == ROUNDUP


def test_slicing_does_not_change_identity_or_the_change_gate():
    """`content_hash` is POST-level by design — the post is the unit of change.
    If slicing altered it, every row would re-mint and a warm cycle would look
    like a changed one, re-extracting the whole corpus every twelve hours."""
    store, first = run(Roundup())
    hashes = {r.content_hash for r in store.rows.values()}
    ids = set(store.rows)
    assert len(hashes) == 1, "children of one post share its hash"

    pipe = IngestPipeline(
        store=store,
        extractor=EXTRACTION_PROMPT | structured(Roundup(), JobExtractionBatch),
        adjudicator=LEGITIMACY_PROMPT | structured(Roundup(), LegitimacyVerdict),
        embedder=FakeEmbedder(), config=Config(), model_name="fake",
    )
    second = pipe.run([posting()])

    assert second.extractions == 0, "a re-slice must not look like a change"
    assert set(store.rows) == ids


def test_the_employer_still_grounds_against_the_whole_post():
    """The subtle one, and the reason grounding was NOT moved to the slice.

    A roundup names its employer once, at the top. Ground `company` against a
    vacancy's slice and it fails, `poster_type` falls back to 'unknown', and
    every split row becomes ineligible for aggregation under migration 0006 —
    turning a text cleanup into the silent deletion of 245 rows' worth of
    demand.
    """
    class WithEmployer(Roundup):
        def respond(self, schema, payload):
            if schema.__name__ == "JobExtractionBatch":
                batch = super().respond(schema, payload)
                for job in batch.jobs:
                    job.company = "Majees Technical Services LLC"
                    job.listing_intent = "vacancy"
                    job.poster_type = "company"
                return batch
            return super().respond(schema, payload)

    store, _ = run(WithEmployer())
    rows = list(store.rows.values())
    # The employer appears in the ROUNDUP header, not in either vacancy's slice.
    assert all("Majees Technical Services LLC" not in r.raw_description for r in rows)
    assert all(r.company == "Majees Technical Services LLC" for r in rows)
    assert all(r.poster_type == "company" for r in rows), (
        "grounding must still read the post body, or every split row silently "
        "stops counting as demand"
    )
