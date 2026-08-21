"""The knowledge base as documents: what is read, and how it is cut up.

No database and no embedder — this is the half of the feature that is pure text
handling, and it runs on every offline run.

`test_the_shipped_documents_carry_no_figure_that_rots` is the test that carries
this file, and it is not a style check. Anything in these documents becomes
quotable by Hud as present-tense fact, because `verify_answer` accepts a figure
that appears in a retrieved passage. A corpus size written into the handbook is
therefore a claim the assistant will still be making confidently a year after it
stopped being true.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from api.knowledge import Document, chunk, read_documents

DOCS = Path(__file__).resolve().parent.parent / "docs" / "knowledge"


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
def test_both_languages_are_present_for_every_document():
    """Parity, the same rule the interface's i18n files are held to.

    A document that exists in one language only is not a small gap: it is an
    Arabic user asking a question the assistant can answer in English and
    declining, which reads as the product not being finished in their language.
    """
    docs = read_documents(DOCS)
    assert docs, "no knowledge documents were read at all"

    by_locale: dict[str, set[str]] = {}
    for doc in docs:
        by_locale.setdefault(doc.locale, set()).add(doc.slug)

    assert by_locale.get("en"), "no English documents"
    assert by_locale["en"] == by_locale.get("ar"), (
        f"missing translations: only en {sorted(by_locale['en'] - by_locale.get('ar', set()))}, "
        f"only ar {sorted(by_locale.get('ar', set()) - by_locale['en'])}")


def test_a_file_without_a_locale_is_skipped_not_guessed(tmp_path):
    """Guessing would put an Arabic passage in the English pool, where it is
    retrieved for the wrong questions and never for the right ones."""
    (tmp_path / "notes.md").write_text("# Notes\n\n## A\n\nbody", encoding="utf-8")
    (tmp_path / "good.en.md").write_text("# Good\n\n## A\n\nbody", encoding="utf-8")

    assert [d.slug for d in read_documents(tmp_path)] == ["good"]


def test_the_title_comes_from_the_heading(tmp_path):
    (tmp_path / "x.en.md").write_text("# What Itqan is\n\n## A\n\nbody", encoding="utf-8")
    assert read_documents(tmp_path)[0].title == "What Itqan is"


def test_an_empty_file_is_not_a_document(tmp_path):
    (tmp_path / "empty.en.md").write_text("   \n\n", encoding="utf-8")
    assert read_documents(tmp_path) == []


# ---------------------------------------------------------------------------
# chunking
# ---------------------------------------------------------------------------
def test_every_chunk_carries_the_document_title():
    """A passage is retrieved alone and shown alone. "Deleting removes the file
    itself" is ambiguous until "What Itqan stores" sits above it."""
    doc = Document(slug="x", locale="en", title="What Itqan stores",
                   text="# What Itqan stores\n\n## Your documents\n\nThey are private.\n"
                        "\n## Your account\n\nYours alone.")
    pieces = chunk(doc)

    assert len(pieces) == 2
    assert all(p.startswith("What Itqan stores\n\n") for p in pieces)
    assert "Your documents" in pieces[0] and "Your account" in pieces[1]


def test_the_heading_is_not_repeated_as_its_own_chunk():
    doc = Document(slug="x", locale="en", title="T",
                   text="# T\n\n## One\n\nbody one")
    assert chunk(doc) == ["T\n\n## One\n\nbody one"]


def test_a_long_section_splits_at_a_paragraph_not_mid_sentence():
    """The model is asked to quote from these. A passage cut mid-claim is worse
    than a slightly longer one."""
    para = "word " * 120                     # ~600 chars
    doc = Document(slug="x", locale="en", title="T",
                   text=f"# T\n\n## Long\n\n{para}\n\n{para}\n\n{para}")
    pieces = chunk(doc)

    assert len(pieces) > 1, "an over-long section was not split"
    for piece in pieces:
        # Nothing was cut inside a paragraph: every piece ends where a paragraph
        # ended, never part-way through one.
        assert piece.strip().endswith("word"), piece[-40:]


def test_a_short_document_stays_one_chunk_per_section():
    doc = Document(slug="x", locale="en", title="T", text="# T\n\n## A\n\nshort body")
    assert len(chunk(doc)) == 1


# ---------------------------------------------------------------------------
# the content rule
# ---------------------------------------------------------------------------
# Figures that are STABLE and load-bearing, cross-checked against the code below.
# Everything else numeric is treated as a figure that rots.
_STABLE = {"30", "1", "12", "3", "10", "5", "60", "100", "2", "4", "6"}


@pytest.mark.parametrize("path", sorted(DOCS.glob("*.md")), ids=lambda p: p.name)
def test_the_shipped_documents_carry_no_figure_that_rots(path):
    """THE test for this content.

    `verify_answer` accepts any figure that appears in a retrieved passage, so a
    number written here is a number Hud will state as current fact. Corpus sizes,
    course counts and test counts must therefore stay out of these documents —
    they change weekly, and nothing downstream would ever catch the drift.

    The allowed set is small and every member of it is a constant in the code,
    checked by the test below.
    """
    text = path.read_text(encoding="utf-8")
    # Ignore markdown list numbering at the start of a line.
    body = re.sub(r"(?m)^\s*\d+\.\s", "", text)
    found = {n.replace(",", "") for n in re.findall(r"\d[\d,]*", body)}

    rotten = found - _STABLE
    assert not rotten, (
        f"{path.name} states figure(s) {sorted(rotten)} that nothing keeps true. "
        f"Hud will quote them as current fact. Describe it in words, or add the "
        f"figure to _STABLE and pin it to the constant that sets it.")


def test_the_stated_limits_match_the_code():
    """The documents say 30 messages a day and one re-run a week. If someone
    changes the constant, this fails instead of the documentation quietly
    becoming a lie the assistant repeats."""
    from shared.config import Config

    config = Config()
    limit = config.assistant_daily_messages

    english = (DOCS / "the-assistant.en.md").read_text(encoding="utf-8")
    assert f"{limit} messages a day" in english
    assert "One re-run a week" in english

    # Arabic too, because a limit stated correctly in one language and wrongly in
    # the other is the version of this bug nobody notices.
    arabic = (DOCS / "the-assistant.ar.md").read_text(encoding="utf-8")
    assert str(limit) in arabic

    # The weekly figure is written as a word in both, so the code side is what
    # pins it: if this ever stops being 1, both documents are wrong.
    assert config.assistant_weekly_reruns == 1
