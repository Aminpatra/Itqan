"""What the candidate asked for, and how far it is allowed to act.

The web app asks four questions while Agent A reads the documents. Until now the
answers were saved to a table and never read by anything, so the tests here are
about one thing: that a stated preference reaches the number it is supposed to
reach, and stops well short of the ones it must not.

The preferences change RETRIEVAL — which postings this candidate is measured
against. They must never change whether a requirement counts as satisfied: that
is grounded evidence, and a preference cannot make a skill true.
"""

from __future__ import annotations

import json

import pytest

from tests.agent_c.test_gap_analysis import _posting, profile_path, run_graph  # noqa: F401


def _essence(profile_path, tmp_path, **prefs) -> str:
    state = run_graph(profile_path, tmp_path,
                      postings=[_posting(f"p{i}", 0.85) for i in range(5)],
                      stats=[], **prefs)
    return state["essence"]


# ---------------------------------------------------------------------------
# the role takes the title slot
# ---------------------------------------------------------------------------
def test_no_preference_leaves_the_essence_exactly_as_it_was(profile_path, tmp_path):
    """The baseline that makes the rest meaningful: an unanswered question must
    not silently reshape the query."""
    assert _essence(profile_path, tmp_path).splitlines()[0] == "Test Candidate"


def test_a_preferred_role_joins_the_headline(profile_path, tmp_path):
    """Open to other roles: the CV headline is what they have been, the role is
    what they want, and retrieval should see both."""
    first = _essence(profile_path, tmp_path,
                     preferred_role="Data Analyst").splitlines()[0]
    assert first == "Test Candidate / Data Analyst"


def test_roles_only_replaces_the_headline(profile_path, tmp_path):
    """'Not open to other roles' is a real narrowing, so the headline is dropped
    rather than pooled with the role the candidate actually wants."""
    first = _essence(profile_path, tmp_path, preferred_role="Data Analyst",
                     roles_only=True).splitlines()[0]
    assert first == "Data Analyst"


def test_roles_only_without_a_role_is_a_no_op(profile_path, tmp_path):
    """"Not open to other roles" with no role named states nothing, and must not
    be allowed to empty the title slot — which would leave retrieval resting on
    the skills line alone."""
    assert _essence(profile_path, tmp_path,
                    roles_only=True).splitlines()[0] == "Test Candidate"


def test_the_role_changes_the_embedded_query(profile_path, tmp_path):
    """The point of putting it in the title slot: a different query vector, and
    therefore potentially different postings. A preference recorded but not
    embedded would be decorative."""
    plain = run_graph(profile_path, tmp_path, postings=[_posting("p", 0.85)], stats=[])
    role = run_graph(profile_path, tmp_path, postings=[_posting("p", 0.85)], stats=[],
                     preferred_role="Data Analyst")
    assert plain["query_embedding"] != role["query_embedding"]


# ---------------------------------------------------------------------------
# arrangement: a text bias, and it says so
# ---------------------------------------------------------------------------
def test_arrangement_is_a_line_in_the_essence(profile_path, tmp_path):
    assert "preferred work arrangement: remote" in _essence(
        profile_path, tmp_path, preferred_arrangement="remote")


def test_arrangement_is_published_as_a_bias_not_a_filter(profile_path, tmp_path):
    """No posting records its work arrangement, so there is nothing to filter on.
    Recording the weakness is what stops a reader assuming the stronger claim."""
    state = run_graph(profile_path, tmp_path, postings=[_posting("p", 0.85)], stats=[],
                      preferred_arrangement="remote")
    prefs = json.loads(
        (tmp_path / "t" / "skill_gap.json").read_text(encoding="utf-8")
    )["calibration"]["preferences"]
    assert prefs["arrangement_applied"] == "retrieval_bias"
    assert prefs["preferred_arrangement"] == "remote"
    assert state["output_path"].endswith("skill_gap.json")


# ---------------------------------------------------------------------------
# reproducibility
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("prefs,expected", [
    ({}, None),
    ({"preferred_role": "Data Analyst"}, "joined_headline"),
    ({"preferred_role": "Data Analyst", "roles_only": True}, "replaced_headline"),
])
def test_calibration_records_how_the_role_was_applied(profile_path, tmp_path,
                                                     prefs, expected):
    """A gap file has to be explainable by the inputs that produced it — the same
    reason every threshold in this agent is published beside its output."""
    run_graph(profile_path, tmp_path, postings=[_posting("p", 0.85)], stats=[], **prefs)
    written = json.loads((tmp_path / "t" / "skill_gap.json").read_text(encoding="utf-8"))
    assert written["calibration"]["preferences"]["role_applied"] == expected


def test_a_preference_cannot_close_a_gap(profile_path, tmp_path):
    """The fence. Wanting to be a Data Analyst does not mean the candidate has the
    skills one needs, and no amount of preference may move a requirement from
    missing to matched — only evidence does that.
    """
    posting = _posting("p", 0.85, skills=["Power BI"])
    plain = run_graph(profile_path, tmp_path, postings=[posting], stats=[])
    biased = run_graph(profile_path, tmp_path, postings=[posting], stats=[],
                       preferred_role="Power BI Developer", roles_only=True,
                       preferred_arrangement="remote")

    def verdicts(state):
        return [(r["skill"], r["verdict"])
                for r in state["matched_jobs"][0]["skill_resolution"]]

    assert verdicts(plain) == verdicts(biased) == [("Power BI", "missing")]
    assert plain["matched_jobs"][0]["gap_score"] == biased["matched_jobs"][0]["gap_score"]
