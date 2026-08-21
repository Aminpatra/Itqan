"""Ingesting the knowledge base, and retrieving from it.

A real database, because everything worth checking here is a property of SQL: the
content-hash gate that stops an unchanged document being re-embedded, the delete
that stops a shortened document leaving orphaned passages behind, and the
locale-first ordering.

The embedder is fake and deterministic — identical text gives an identical
vector — so a passage retrieved by its own text scores 1.0 and the plumbing can
be checked without a paid API or a flaky assertion about meaning.

`test_a_shortened_document_does_not_leave_orphans` is the test that carries this
file. An orphaned passage is invisible in the source and fully live in retrieval:
nobody can see it to correct it, and Hud will quote it as current fact.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from api import knowledge as knowledge_module
from shared.config import Config
from tests.fake_embedder import FakeEmbedder

EN = """# What Itqan is

## Who it is for

Anyone looking for work in Oman.

## What it does not do

It does not apply for jobs on your behalf.
"""

AR = """# ما هو إتقان

## لمن هو

لكل من يبحث عن عمل في عُمان.
"""


@pytest.fixture
def docs(tmp_path: Path) -> Path:
    (tmp_path / "what-is-itqan.en.md").write_text(EN, encoding="utf-8")
    (tmp_path / "what-is-itqan.ar.md").write_text(AR, encoding="utf-8")
    return tmp_path


@pytest.fixture
def clean(dsn: str):
    """The knowledge table is global, so it is not in the suite's TRUNCATE list —
    which is correct (it holds no user data) and means this file clears its own."""
    import psycopg

    with psycopg.connect(dsn) as conn:
        conn.execute("TRUNCATE app_knowledge_chunks")
        conn.commit()
    return dsn


def _ingest(dsn: str, directory: Path):
    return knowledge_module.ingest(dsn=dsn, directory=directory,
                                   embedder=FakeEmbedder(), config=Config())


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------
def test_documents_land_as_passages(clean, docs):
    """Asserted against the DATABASE, not the summary, and that distinction found
    a real bug the first time this ran.

    `_connection` was not autocommit, so the SELECT that reads existing hashes
    opened an implicit transaction and every later `with conn.transaction()`
    nested inside it as a savepoint. Releasing a savepoint commits nothing, the
    connection closed without a commit, and the entire ingest was silently rolled
    back — while this summary cheerfully reported three passages written. Agent B
    lost a census to the same mechanism, then a prune that printed "deleted 432
    postings" with all 432 still there.

    A test that reads back the counter it just incremented cannot see any of it.
    """
    import psycopg

    summary = _ingest(clean, docs)
    assert summary["documents"] == 2
    assert summary["chunks"] == 3          # two English sections, one Arabic
    assert summary["embedded"] == 3

    with psycopg.connect(clean) as conn:
        stored = conn.execute(
            "SELECT count(*) FROM app_knowledge_chunks").fetchone()[0]
        vectors = conn.execute(
            "SELECT count(*) FROM app_knowledge_chunks "
            "WHERE embedding IS NOT NULL").fetchone()[0]

    assert stored == 3, "the summary said three passages; the table disagrees"
    assert vectors == 3, "a passage without a vector can never be retrieved"


def test_a_second_run_embeds_nothing(clean, docs):
    """The warm-cycle rule Agents B and D already live by. An ingest that costs
    money every time is one nobody runs after editing a typo — and documentation
    the assistant has not been given is worse than none, because from the outside
    it is indistinguishable."""
    _ingest(clean, docs)
    again = _ingest(clean, docs)

    assert again["chunks"] == 3, "the same documents should still produce 3 passages"
    assert again["embedded"] == 0, "unchanged text was re-embedded"


def test_an_edit_re_embeds_only_what_changed(clean, docs):
    _ingest(clean, docs)
    (docs / "what-is-itqan.en.md").write_text(
        EN.replace("Anyone looking for work in Oman.",
                   "Anyone looking for work anywhere in Oman."), encoding="utf-8")

    again = _ingest(clean, docs)
    assert again["embedded"] == 1, "one section changed; one passage should be embedded"


def test_a_shortened_document_does_not_leave_orphans(clean, docs):
    """THE test. A passage deleted from the source but left in the table is
    invisible where it can be corrected and live where it is quoted."""
    _ingest(clean, docs)
    (docs / "what-is-itqan.en.md").write_text(
        "# What Itqan is\n\n## Who it is for\n\nAnyone looking for work in Oman.\n",
        encoding="utf-8")

    again = _ingest(clean, docs)
    assert again["deleted"] == 1

    passages = knowledge_module.search(
        "It does not apply for jobs on your behalf.",
        dsn=clean, embedder=FakeEmbedder(), locale="en", config=Config())
    assert all("does not apply for jobs" not in p.text for p in passages)


# ---------------------------------------------------------------------------
# retrieval
# ---------------------------------------------------------------------------
def test_a_passage_is_retrieved_by_its_own_text(clean, docs):
    _ingest(clean, docs)
    hits = knowledge_module.search(
        "What Itqan is\n\n## Who it is for\n\nAnyone looking for work in Oman.",
        dsn=clean, embedder=FakeEmbedder(), locale="en", config=Config())

    assert hits, "an exact passage retrieved nothing"
    assert hits[0].similarity == pytest.approx(1.0, abs=1e-6)
    assert hits[0].locale == "en"


def test_the_questions_language_is_preferred(clean, tmp_path):
    """A missing translation must not mean no answer, so locale ORDERS rather
    than filters — but when both exist, the reader's own language wins.

    Both files carry identical body text on purpose. The fake embedder is
    deterministic and not semantic, so identical text means identical vectors and
    both rows tie at 1.0 — which is exactly the condition under which the
    tie-break is the only thing deciding, and therefore the only thing under
    test. A pair of differently-worded documents would be testing the hash.
    """
    body = "# Title\n\n## Section\n\nThe very same words in both files.\n"
    (tmp_path / "twin.en.md").write_text(body, encoding="utf-8")
    (tmp_path / "twin.ar.md").write_text(body, encoding="utf-8")
    _ingest(clean, tmp_path)

    query = "Title\n\n## Section\n\nThe very same words in both files."
    for locale in ("ar", "en"):
        hits = knowledge_module.search(query, dsn=clean, embedder=FakeEmbedder(),
                                       locale=locale, config=Config())
        assert len(hits) == 2, "both language copies should be reachable"
        assert hits[0].similarity == pytest.approx(hits[1].similarity, abs=1e-9)
        assert hits[0].locale == locale, (
            f"asked in {locale}, got {hits[0].locale} first on an exact tie")


def test_nothing_similar_enough_returns_nothing(clean, docs):
    """The floor's real job: a turn with no question in it gets no ABOUT block,
    rather than three passages of noise."""
    _ingest(clean, docs)
    hits = knowledge_module.search(
        "zzzz unrelated gibberish", dsn=clean, embedder=FakeEmbedder(),
        locale="en", config=Config())
    assert hits == []
    assert knowledge_module.knowledge_block(hits) == ""


def test_an_empty_question_never_reaches_the_database(clean):
    assert knowledge_module.search("   ", dsn=clean, embedder=FakeEmbedder(),
                                   config=Config()) == []


def test_the_table_holds_nothing_about_any_user(clean, docs):
    """The isolation question this project asks of every read surface. It does
    not arise here, and this is what makes that checkable rather than asserted:
    there is no column that could carry a user."""
    import psycopg

    _ingest(clean, docs)
    with psycopg.connect(clean) as conn:
        cols = {r[0] for r in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'app_knowledge_chunks'").fetchall()}

    assert not {c for c in cols if "user" in c}, f"knowledge table carries {cols}"
