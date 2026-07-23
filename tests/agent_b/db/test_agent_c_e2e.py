"""Agent C end to end against real Postgres: real export, real read-only ESCO
mapping, the whole graph — only the embedder is fake (deterministic).

The load-bearing assertions: the tier fallthrough works through real SQL, and
``skill_esco_map`` has EXACTLY as many rows after the run as before — Agent C's
read-only promise proven, not assumed.
"""

from __future__ import annotations

import json
from pathlib import Path

from agents.agent_b_job_ingest.esco_map import sync_taxonomy
from agents.agent_b_job_ingest.records import PersistedPosting
from agents.agent_c_gap_analysis.graph import build_gap_graph
from agents.agent_c_gap_analysis.nodes import Deps
from shared.config import Config
from shared.contracts import CandidateProfile
from shared.job_market import map_skills_to_esco
from tests.fake_embedder import FakeEmbedder

FIXTURE = Path(__file__).parents[1] / "fixtures" / "esco_skills_sample.csv"

HEADLINE = "Test Candidate"
SKILLS = ["manage time", "quantum basket weaving"]
# Must mirror nodes.make_build_query_embedding's essence construction.
ESSENCE = f"{HEADLINE}\nMuscat\nskills: {', '.join(SKILLS)}"


def _profile(tmp_path) -> str:
    profile = CandidateProfile(
        run_id="e2e-cand", generated_at="2026-07-23T00:00:00+00:00",
        candidate={"contact": {"location": "Muscat"}},
        skills={"accepted": [{"name": s} for s in SKILLS], "rejected": []},
        summary={"headline": HEADLINE},
    )
    path = tmp_path / "candidate_profile.json"
    path.write_text(profile.model_dump_json(), encoding="utf-8")
    return str(path)


def _seed_postings(store, embedder):
    # Five postings whose essence embedding EQUALS the candidate's, so all five
    # clear the 0.80 usable threshold at similarity 1.0 -> direct path.
    query_vec = embedder.embed_query(ESSENCE)
    rows = [
        PersistedPosting(
            posting_id=f"job{i}", source="el7far", source_group="g",
            source_type="blogger_feed", source_url=f"https://e.test/job{i}",
            title=f"Job {i}", raw_description="d", content_hash=f"h{i}",
            sector="2", country="OM", listing_intent="vacancy",
            poster_type="company", required_skills=["manage time", "perform welding"],
            embedding=list(query_vec),
        )
        for i in range(5)
    ]
    store.upsert_batch(rows)
    conn = store.connect()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO skill_demand_stats (sector, skill, skill_key, esco_code, "
            "window_start, window_end, frequency_count, trend) VALUES "
            "('2','manage time','manage time','http://example.test/esco/skill/0001',"
            "'2026-06-22','2026-07-22',7,'stable')"
        )
    conn.commit()


def _map_rows(store) -> int:
    conn = store.connect()
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) AS n FROM skill_esco_map")
        return cur.fetchone()["n"]


def test_full_gap_run_against_real_tables(store, tmp_path):
    embedder = FakeEmbedder()
    sync_taxonomy(store, FakeEmbedder(), path=FIXTURE, version="test-1")
    _seed_postings(store, embedder)
    before = _map_rows(store)

    config = Config(database_url=store.dsn)
    graph = build_gap_graph(Deps(config=config, embedder=embedder))
    state = graph.invoke({
        "profile_path": _profile(tmp_path),
        "top_k": 15, "output_dir": str(tmp_path), "run_id": "e2e",
    })

    # Direct path: five usable postings at similarity 1.0.
    assert state["used_fallback"] is False
    assert len(state["matched_jobs"]) == 5
    job = state["matched_jobs"][0]

    # 'manage time': candidate maps to concept 0001 via the real exact tier, the
    # job skill resolves to 0001 via the real stats row -> matched by identity.
    assert "manage time" in job["matched_skills"]
    # 'perform welding' shares no esco with the candidate and its FakeEmbedder
    # vector is unrelated to both candidate phrases -> missing.
    assert "perform welding" in job["missing_skills"]
    # weights: missing 'perform welding' floor 1; matched 'manage time' freq 7
    assert job["gap_score"] == round(1 / 8, 4)

    # Tier fallthrough through real SQL: exact hit + honest unmapped-with-score.
    methods = {m.skill_key: m.method for m in state["candidate_mappings"]}
    assert methods["manage time"] == "exact"
    assert methods["quantum basket weaving"] == "unmapped"

    # THE read-only proof: Agent C added nothing to Agent B's mapping table.
    assert _map_rows(store) == before, "Agent C wrote into skill_esco_map"

    out = json.loads((tmp_path / "e2e" / "skill_gap.json").read_text(encoding="utf-8"))
    assert out["used_fallback"] is False and len(out["matched_jobs"]) == 5


def test_alt_label_tier_through_real_sql(store):
    sync_taxonomy(store, FakeEmbedder(), path=FIXTURE, version="test-1")

    mappings = map_skills_to_esco(
        ["prioritise tasks", "bookkeeping"],
        embedder=FakeEmbedder(), config=Config(database_url=store.dsn),
    )
    by_key = {m.skill_key: m for m in mappings}

    assert by_key["prioritise tasks"].method == "alt_label"
    assert by_key["prioritise tasks"].esco_uri.endswith("/0001")
    assert by_key["bookkeeping"].esco_label == "accounting"
