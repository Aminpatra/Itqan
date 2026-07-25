"""Agent C offline: the whole graph driven by fakes injected through Deps.

Similarities are PINNED — each phrase gets a hand-built vector whose dot
product with the candidate's is an exact chosen number — so every band boundary
is tested at its actual edge, not wherever a hash embedder happens to land.
"""

from __future__ import annotations

import json
import math

import pytest

from agents.agent_c_gap_analysis.graph import build_gap_graph
from agents.agent_c_gap_analysis.nodes import Deps
from shared.config import Config
from shared.contracts import CandidateProfile, JobPostingExport, SkillDemandStatRow
from shared.job_market import SkillMapping
from tests.fake_embedder import _vector_for


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------
def _pin(dot: float) -> list[float]:
    """A unit vector whose dot with the candidate axis [1,0,0,0] is exactly `dot`."""
    return [dot, math.sqrt(max(0.0, 1 - dot * dot)), 0.0, 0.0]


CANDIDATE_AXIS = [1.0, 0.0, 0.0, 0.0]


class PinnedEmbedder:
    def __init__(self, table: dict[str, list[float]]):
        self.table = {k.lower(): v for k, v in table.items()}

    def embed_query(self, text: str) -> list[float]:
        return self.table.get(text.lower(), _vector_for(text.lower()))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]


def _posting(pid: str, similarity: float, *, sector="2", skills=()) -> JobPostingExport:
    return JobPostingExport(
        posting_id=pid, source="s", source_group="g", source_type="blogger_feed",
        source_url=f"https://e.test/{pid}", title=f"Job {pid}", raw_description="d",
        sector=sector, required_skills=list(skills), country="OM",
        first_seen_at="2026-07-22T00:00:00+00:00", last_seen_at="2026-07-22T00:00:00+00:00",
        similarity=similarity,
    )


def _stat(skill: str, *, esco=None, freq=1, sector="2") -> SkillDemandStatRow:
    return SkillDemandStatRow(
        sector=sector, skill=skill, skill_key=skill.lower(), esco_code=esco,
        window_start="2026-06-22", window_end="2026-07-22",
        frequency_count=freq, computed_at="2026-07-22T00:00:00+00:00",
    )


def make_exporter(postings, stats):
    def exporter(query_embedding, *, top_k=15, config=None, **_):
        return {"job_postings": postings[:top_k], "skill_demand_stats": stats}
    return exporter


def make_mapper(mappings):
    def mapper(skills, *, embedder, config):
        by_key = {m.skill_key: m for m in mappings}
        return [by_key.get(s.lower(), SkillMapping(skill=s, skill_key=s.lower()))
                for s in skills]
    return mapper


@pytest.fixture
def profile_path(tmp_path):
    profile = CandidateProfile(
        run_id="cand-1", generated_at="2026-07-23T00:00:00+00:00",
        candidate={"contact": {"location": "Muscat"}},
        skills={"accepted": [{"name": "Alpha"}], "rejected": []},
        summary={"headline": "Test Candidate"},
    )
    path = tmp_path / "candidate_profile.json"
    path.write_text(profile.model_dump_json(), encoding="utf-8")
    return str(path)


def run_graph(profile_path, tmp_path, *, postings, stats, mappings=(),
              embedder=None, **initial):
    deps = Deps(
        config=Config(),
        embedder=embedder or PinnedEmbedder({"alpha": CANDIDATE_AXIS}),
        exporter=make_exporter(list(postings), list(stats)),
        mapper=make_mapper(list(mappings)),
    )
    state = build_gap_graph(deps).invoke({
        "profile_path": profile_path, "top_k": 15,
        "output_dir": str(tmp_path), "run_id": "t", **initial,
    })
    return state


# ---------------------------------------------------------------------------
# fallback boundary
# ---------------------------------------------------------------------------
def test_five_usable_postings_take_the_direct_path(profile_path, tmp_path):
    postings = [_posting(f"p{i}", 0.85) for i in range(5)]
    state = run_graph(profile_path, tmp_path, postings=postings, stats=[])

    assert state["used_fallback"] is False
    assert len(state["matched_jobs"]) == 5
    assert state["fallback_sector_gap"] is None


def test_exactly_four_usable_postings_trigger_the_fallback(profile_path, tmp_path):
    """The boundary is <5 USABLE — sub-threshold retrievals do not count, however
    many there are."""
    postings = [_posting(f"p{i}", 0.85) for i in range(4)] + [
        _posting(f"low{i}", 0.30, sector="5") for i in range(11)
    ]
    stats = [_stat("Welding", sector="5", freq=3)]
    state = run_graph(profile_path, tmp_path, postings=postings, stats=stats)

    assert state["used_fallback"] is True
    assert state["matched_jobs"] == []
    # modal sector over ALL retrievals: eleven 5s beat four 2s
    assert state["inferred_sector"] == "5"
    assert state["fallback_sector_gap"]["sector"] == "5"


def test_sector_override_beats_modal_inference(profile_path, tmp_path):
    postings = [_posting("p0", 0.30, sector="5")]
    stats = [_stat("Welding", sector="7")]
    state = run_graph(profile_path, tmp_path, postings=postings, stats=stats,
                      sector_override="7")

    assert state["inferred_sector"] == "7"
    assert state["fallback_sector_gap"]["sector"] == "7"


def test_fallback_ignores_freq_one_noise(profile_path, tmp_path):
    """A phrase aggregated once is not sector demand. Measured live: without the
    floor the fallback compared against 463 rows (~87% freq-1) and the gap
    saturated at ~1.0 — a number that says nothing."""
    postings = [_posting("p0", 0.30, sector="5")]
    stats = [
        _stat("Real Demand", sector="5", freq=4),
        _stat("One Off Phrase", sector="5", freq=1),
    ]
    state = run_graph(profile_path, tmp_path, postings=postings, stats=stats)

    fallback = state["fallback_sector_gap"]
    everything = (fallback["matched_skills"] + fallback["missing_skills"]
                  + fallback["possible_match_skills"])
    assert "Real Demand" in everything
    assert "One Off Phrase" not in everything


def test_zero_retrievals_and_no_sector_skips_fallback_with_a_warning(profile_path, tmp_path):
    state = run_graph(profile_path, tmp_path, postings=[], stats=[])

    assert state["used_fallback"] is True
    assert state["fallback_sector_gap"] is None
    assert any("skipped rather than guessed" in w for w in state["warnings"])


# ---------------------------------------------------------------------------
# the possible_match band never resolves
# ---------------------------------------------------------------------------
def test_band_boundaries_are_exact_and_the_middle_never_resolves(profile_path, tmp_path):
    """0.601 and 0.799 stay possible_match; 0.80 exactly is matched; 0.30 is
    missing. gap_score counts the band in the denominator only:
    missing(1) / total(4) = 0.25 at uniform floor weights."""
    embedder = PinnedEmbedder({
        "alpha": CANDIDATE_AXIS,
        "low-band": _pin(0.601),
        "high-band": _pin(0.799),
        "at-match": _pin(0.80),
        "far-off": _pin(0.30),
    })
    postings = [_posting(f"p{i}", 0.85,
                         skills=["low-band", "high-band", "at-match", "far-off"])
                for i in range(5)]
    state = run_graph(profile_path, tmp_path, postings=postings, stats=[],
                      embedder=embedder)

    job = state["matched_jobs"][0]
    assert sorted(job["possible_match_skills"]) == ["high-band", "low-band"], (
        "the [0.6, 0.8) band must stay unresolved — never matched, never missing"
    )
    assert job["matched_skills"] == ["at-match"]
    assert job["missing_skills"] == ["far-off"]
    assert job["gap_score"] == pytest.approx(0.25)


def test_esco_identity_beats_a_low_phrase_similarity(profile_path, tmp_path):
    """A job skill sharing the candidate's esco_code is matched even when the
    raw phrases are embedding-distant — canonical identity outranks fuzz."""
    embedder = PinnedEmbedder({"alpha": CANDIDATE_AXIS, "totally different words": _pin(0.10)})
    postings = [_posting(f"p{i}", 0.85, skills=["totally different words"]) for i in range(5)]
    stats = [_stat("totally different words", esco="uri:X")]
    mappings = [SkillMapping(skill="Alpha", skill_key="alpha",
                             esco_uri="uri:X", esco_label="x", method="exact")]
    state = run_graph(profile_path, tmp_path, postings=postings, stats=stats,
                      mappings=mappings, embedder=embedder)

    assert state["matched_jobs"][0]["matched_skills"] == ["totally different words"]


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------
def test_demand_frequency_weights_the_gap_score(profile_path, tmp_path):
    """One missing skill with freq 9 vs matched(1) + two possible(1 each):
    gap = 9 / 12 = 0.75. The floor keeps un-aggregated skills alive at 1."""
    embedder = PinnedEmbedder({
        "alpha": CANDIDATE_AXIS,
        "heavy-missing": _pin(0.10),
        "band-a": _pin(0.70), "band-b": _pin(0.70),
        "hit": _pin(0.90),
    })
    postings = [_posting(f"p{i}", 0.85,
                         skills=["heavy-missing", "band-a", "band-b", "hit"])
                for i in range(5)]
    stats = [_stat("heavy-missing", freq=9)]
    state = run_graph(profile_path, tmp_path, postings=postings, stats=stats,
                      embedder=embedder)

    assert state["matched_jobs"][0]["gap_score"] == pytest.approx(9 / 12)
    assert state["aggregate"]["most_common_missing_skills"][0] == "heavy-missing"


def test_missing_skill_details_carries_esco_and_priority_for_agent_e(profile_path, tmp_path):
    """The additive per-skill detail Agent E inherits: same skills and order as
    most_common_missing_skills, each with its esco_code (or None when the skill
    never mapped — never guessed) and its summed-demand priority_score."""
    embedder = PinnedEmbedder({
        "alpha": CANDIDATE_AXIS,
        "heavy-missing": _pin(0.10),   # esco-coded, freq 9 over 5 jobs -> priority 45
        "light-missing": _pin(0.10),   # unmapped, freq 2 over 5 jobs -> priority 10, esco None
        "hit": _pin(0.90),
    })
    postings = [_posting(f"p{i}", 0.85, skills=["heavy-missing", "light-missing", "hit"])
                for i in range(5)]
    stats = [_stat("heavy-missing", esco="uri:HM", freq=9),
             _stat("light-missing", freq=2)]
    state = run_graph(profile_path, tmp_path, postings=postings, stats=stats,
                      embedder=embedder)

    details = state["aggregate"]["missing_skill_details"]
    # order + membership mirror most_common_missing_skills
    assert [d["skill"] for d in details] == state["aggregate"]["most_common_missing_skills"]
    by_skill = {d["skill"]: d for d in details}
    # priority_score is the demand weight summed across every job the skill is
    # missing in (freq 9 x 5 postings = 45; freq 2 x 5 = 10).
    assert by_skill["heavy-missing"] == {
        "skill": "heavy-missing", "esco_code": "uri:HM", "priority_score": 45,
    }
    assert by_skill["light-missing"] == {
        "skill": "light-missing", "esco_code": None, "priority_score": 10,
    }
    # 'hit' matched -> never a missing detail
    assert "hit" not in by_skill


# ---------------------------------------------------------------------------
# output contract
# ---------------------------------------------------------------------------
def test_output_json_carries_the_contract_keys_and_mapping_evidence(profile_path, tmp_path):
    mappings = [SkillMapping(skill="Alpha", skill_key="alpha",
                             method="unmapped", similarity=0.71)]
    postings = [_posting(f"p{i}", 0.85, skills=["alpha"]) for i in range(5)]
    state = run_graph(profile_path, tmp_path, postings=postings, stats=[],
                      mappings=mappings)

    out = json.loads(
        (tmp_path / "t" / "skill_gap.json").read_text(encoding="utf-8")
    )
    for key in ("user_id", "used_fallback", "matched_jobs",
                "fallback_sector_gap", "aggregate"):
        assert key in out
    assert out["user_id"] == "cand-1"       # defaults to the profile run_id
    job = out["matched_jobs"][0]
    for key in ("job_id", "job_title", "matched_skills", "missing_skills",
                "possible_match_skills", "gap_score"):
        assert key in job
    # The near-miss score travels in C's own output — the user-decided home for
    # candidate-side tuning evidence (never Agent B's tables).
    assert out["candidate_skill_mappings"][0]["similarity"] == 0.71
    assert out["retrieval"]["usable"] == 5
    assert state["output_path"].endswith("skill_gap.json")
