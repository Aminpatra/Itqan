"""Agent E end to end against real Postgres: the real course_market reader wired
into the graph, only the LLM faked. Proves selection over live-shaped data.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from agents.agent_d_course_ingest.records import PersistedCourse
from agents.agent_e_course_recommend.graph import build_recommend_graph
from agents.agent_e_course_recommend.nodes import Deps
from shared.config import Config


class FakePlainLLM:
    def __init__(self):
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages[1].content)
        return SimpleNamespace(content="Recommended because it teaches the skill.")


def _course(cid, *, skills, rating=None, provider="IBM"):
    return PersistedCourse(
        course_id=cid, source="coursera", source_group="coursera", source_type="api",
        source_url=f"https://c.test/{cid}", name=f"Course {cid}", raw_description="d",
        content_hash=f"h_{cid}", taught_skills=skills, provider=provider, rating=rating,
    )


def test_full_graph_over_real_courses_picks_and_grounds(store, tmp_path):
    # supply: two courses teach 'sql'; one also teaches 'python'
    store.upsert_batch([
        _course("c1", skills=["sql"], rating=4.0),
        _course("c2", skills=["sql", "python"], rating=4.9),
    ])
    store.connect().commit()
    store.upsert_course_map([
        {"skill_key": "sql", "esco_uri": "uri:SQL", "method": "exact",
         "similarity": None, "esco_version": "test-1"},
        {"skill_key": "python", "esco_uri": "uri:PY", "method": "exact",
         "similarity": None, "esco_version": "test-1"},
    ])
    store.connect().commit()                 # reader uses a SEPARATE connection

    gap = {
        "user_id": "cand-1", "used_fallback": False,
        "aggregate": {"missing_skill_details": [
            {"skill": "sql", "esco_code": "uri:SQL", "priority_score": 5.0},
            {"skill": "python", "esco_code": "uri:PY", "priority_score": 3.0},
            {"skill": "welding", "esco_code": "uri:WELD", "priority_score": 2.0},
        ]},
    }
    (tmp_path / "skill_gap.json").write_text(json.dumps(gap), encoding="utf-8")

    llm = FakePlainLLM()
    deps = Deps(config=Config(database_url=store.dsn), llm=llm)
    state = build_recommend_graph(deps).invoke({
        "gap_path": str(tmp_path / "skill_gap.json"),
        "output_dir": str(tmp_path), "run_id": "e2e",
    })

    out = json.loads((tmp_path / "e2e" / "course_recommendations.json").read_text(encoding="utf-8"))
    by_skill = {r["skill"]: r for r in out["recommendations"]}

    # c2 covers sql AND python (higher rating too) -> one entry, python folded in
    assert by_skill["sql"]["course"]["course_id"] == "c2"
    assert by_skill["sql"]["course"]["covers_other_skills"] == ["python"]
    assert "python" not in by_skill                      # folded into c2's entry
    # welding has no course anywhere
    assert by_skill["welding"]["no_course_found"] is True
    assert by_skill["welding"]["course"] is None
    # the LLM ran once (for c2), skipped for welding
    assert len(llm.calls) == 1
    assert out["user_id"] == "cand-1"
