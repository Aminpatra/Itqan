"""Agent C -> Agent E, on a REAL envelope rather than a hand-built dict.

Every other Agent E test constructs the gap dictionary itself, which means they
all agree with each other and none of them agrees with Agent C. That is exactly
where contract drift hides: Agent C could rename `missing_skill_details`
tomorrow and every Agent E test would still pass, while production quietly fell
through to the far lossier `most_common_missing_skills` path.

So this suite runs Agent C's real graph to produce a real `skill_gap.json`, then
feeds that FILE to Agent E's real graph. The load-bearing assertion is the drift
canary: Agent E must not emit its legacy-fallback warning.
"""

from __future__ import annotations

import json

import pytest

from agents.agent_e_course_recommend.graph import build_recommend_graph
from agents.agent_e_course_recommend.nodes import Deps as RecommendDeps
from shared.config import Config
from shared.contracts import SkillGap, load_gap
from tests.agent_c.test_gap_analysis import (  # noqa: F401 - profile_path is a fixture
    CANDIDATE_AXIS,
    PinnedEmbedder,
    _pin,
    _posting,
    _stat,
    profile_path,
    run_graph,
)
from tests.agent_e.test_recommend import FakeReader, cc


def _real_gap_file(profile_path, tmp_path):  # noqa: F811
    """Drive Agent C end to end and return the path it actually wrote."""
    embedder = PinnedEmbedder({"alpha": CANDIDATE_AXIS, "welding": _pin(0.10),
                               "python": _pin(0.10)})
    postings = [_posting(f"p{i}", 0.85, skills=["welding", "python"]) for i in range(5)]
    state = run_graph(profile_path, tmp_path, postings=postings,
                      stats=[_stat("welding", freq=9), _stat("python", freq=4)],
                      embedder=embedder)
    return state["output_path"]


def test_agent_c_writes_something_the_published_contract_accepts(profile_path, tmp_path):  # noqa: F811
    """Agent C validates on write, so this cannot regress silently — but assert
    it from the outside too, against the file on disk."""
    gap = load_gap(_real_gap_file(profile_path, tmp_path))
    assert gap.aggregate.missing_skill_details, "the enriched aggregate is empty"
    assert gap.schema_version.startswith("itqan.skill_gap/")


def test_agent_e_takes_the_rich_path_on_a_real_agent_c_envelope(profile_path, tmp_path):  # noqa: F811
    """THE drift canary. If Agent C renames or drops a field Agent E reads, the
    consumer falls back to a skill list with no ESCO codes and no weights — worse
    recommendations, produced silently. This fails instead."""
    gap_path = _real_gap_file(profile_path, tmp_path)
    gap = load_gap(gap_path)
    skills = [d.skill for d in gap.aggregate.missing_skill_details]

    reader = FakeReader(by_key={s: [cc(f"c-{s}", f"Course for {s}")] for s in skills},
                        by_esco={})
    state = build_recommend_graph(
        RecommendDeps(config=Config(), llm=None, courses_reader=reader)
    ).invoke({"gap_path": gap_path, "output_dir": str(tmp_path), "run_id": "chain-e"})

    legacy = [w for w in state.get("warnings", []) if "most_common_missing_skills" in w]
    assert not legacy, (
        "Agent E fell back to the legacy path on a REAL Agent C envelope — the two "
        f"sides have drifted: {legacy}")
    assert state["recommendations"], "the chain produced no recommendations"


def test_every_field_agent_e_reads_is_actually_produced(profile_path, tmp_path):  # noqa: F811
    """Names the coupling explicitly, so a rename fails with a message that says
    what broke rather than a mysteriously emptier output."""
    gap = load_gap(_real_gap_file(profile_path, tmp_path))
    detail = gap.aggregate.missing_skill_details[0]
    raw = json.loads(open(_real_gap_file(profile_path, tmp_path), encoding="utf-8").read())
    produced = set(raw["aggregate"]["missing_skill_details"][0])

    for field in ("skill", "esco_code", "priority_score", "jobs_missing_in",
                  "low_confidence"):
        assert field in produced, f"Agent E reads {field!r}; Agent C no longer emits it"
    assert detail.skill


def test_the_contract_ignores_fields_it_does_not_know(tmp_path):
    """Additive tolerance is what makes it safe to put a model over envelopes
    already in the wild: a producer adding a key must never break a consumer."""
    envelope = {
        "schema_version": "itqan.skill_gap/9.9",
        "user_id": "u", "used_fallback": False,
        "a_field_from_the_future": {"nested": [1, 2, 3]},
        "aggregate": {
            "missing_skill_details": [
                {"skill": "python", "priority_score": 1.0, "invented_later": True},
            ],
            "another_new_key": 5,
        },
    }
    path = tmp_path / "future.json"
    path.write_text(json.dumps(envelope), encoding="utf-8")

    gap = load_gap(str(path))
    assert gap.aggregate.missing_skill_details[0].skill == "python"


def test_a_field_that_changes_type_is_caught():
    """The failure this exists for: not an added key, but a broken one."""
    with pytest.raises(ValueError):
        SkillGap.model_validate({
            "aggregate": {"missing_skill_details": [
                {"skill": "python", "priority_score": "not a number"}]},
        })
