"""The fence, now that Agent S has a second source to draw on.

`verify_answer` used to check every figure against the fact sheet alone. Asked
what Itqan is, Hud could only decline — and if it had answered from Itqan's own
handbook, the answer would have been thrown away for stating a figure nobody
measured, and the person told we could not help.

**The rule has not changed.** It was never "only the fact sheet"; it is "only
what we actually showed the model". These tests pin the widened evidence set and,
just as importantly, pin that it was WIDENED and not opened.
"""

from __future__ import annotations

from agents.agent_s_assistant.facts import verify_answer

SHEET = "Readiness score: 60/100\nMatched jobs: none yet"
DOCS = ("How Itqan works\n\n## Where the jobs come from\n\n"
        "Public job postings from Omani sources, refreshed every 12 hours.")


# ---------------------------------------------------------------------------
# the widened evidence set
# ---------------------------------------------------------------------------
def test_a_figure_from_the_documentation_is_published():
    """THE test. Before this it was rejected, and the person got "I could not
    answer that from your results" to a question about the product."""
    assert verify_answer("Job postings refresh every 12 hours.", SHEET, DOCS) is None


def test_the_same_figure_is_still_rejected_without_the_documentation():
    """The fence was widened, not opened: the passage has to have been retrieved
    for this turn. A figure the model remembers from somewhere else is exactly
    what this check exists to catch."""
    problem = verify_answer("Job postings refresh every 12 hours.", SHEET, "")
    assert problem and "12" in problem


def test_a_figure_in_neither_block_is_rejected():
    assert verify_answer("Itqan holds 40,000 job postings.", SHEET, DOCS) is not None


def test_a_figure_from_the_fact_sheet_still_works():
    """The behaviour that already existed must not have been traded away."""
    assert verify_answer("You are at 60 out of 100.", SHEET, DOCS) is None


def test_knowledge_defaults_to_empty_so_existing_callers_are_unchanged():
    """The parameter is optional on purpose — the CLI and every existing test
    call this with two arguments."""
    assert verify_answer("You are at 60 out of 100.", SHEET) is None


# ---------------------------------------------------------------------------
# the vocabulary that the documentation can unlock
# ---------------------------------------------------------------------------
def test_esco_may_be_named_when_the_documentation_names_it():
    """In a sentence about someone's results, "ESCO" is plumbing and means
    nothing to them. In an answer to "how does the matching work?", it is the
    honest word — a real public vocabulary, which our own handbook names."""
    docs = ("How Itqan works\n\n## How skills are compared\n\n"
            "Itqan maps both onto ESCO, the European Union's standard skills "
            "vocabulary.")
    assert verify_answer("Itqan maps skills onto ESCO, the EU's vocabulary.",
                         SHEET, docs) is None


def test_esco_is_still_rejected_when_nothing_showed_it():
    problem = verify_answer("Your skills were mapped to ESCO codes.", SHEET, DOCS)
    assert problem and "esco" in problem


def test_the_genuinely_unpublishable_words_stay_banned_whatever_the_documents_say():
    """`gap_score` runs backwards — 0.0 is the BEST — so a person shown it reads
    it inverted. No document can make that safe, and one that mentioned it must
    not unlock it."""
    docs = DOCS + "\n\nInternally this is called the gap_score."
    problem = verify_answer("Your gap_score is 0.4.", SHEET, docs)
    assert problem and "gap_score" in problem


def test_a_card_handle_is_still_caught():
    assert verify_answer("These matched you: [J1].", SHEET, DOCS) is not None


# ---------------------------------------------------------------------------
# Arabic — a pre-existing rejection, found while building this
# ---------------------------------------------------------------------------
def test_an_arabic_answer_may_use_arabic_numerals():
    """Found 2026-08-21, and it predates the knowledge base entirely.

    The fact sheet is built in code and is always ASCII, while the digit class
    matches Unicode decimal digits — so an answer written in Arabic-Indic
    numerals had every figure read as invented, and the person silently got the
    deterministic fallback. `fold_digits` is the helper Agent A's grounding
    already uses for the same reason.
    """
    assert verify_answer("جاهزيتك "
                         "٦٠ من ١٠٠.", SHEET) is None


def test_folding_does_not_let_an_invented_arabic_figure_through():
    """The fold makes the comparison fair; it must not make it toothless."""
    assert verify_answer("لديك ٩٩ "
                         "وظيفة.", SHEET) is not None
