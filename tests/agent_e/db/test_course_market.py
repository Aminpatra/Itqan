"""shared.course_market against real Postgres: eligibility, the esco path, the
exact-key fallback, and a course teaching two requested concepts.
"""

from __future__ import annotations

from agents.agent_d_course_ingest.records import PersistedCourse
from shared.config import Config
from shared.contracts import CourseCandidate
from shared.course_market import courses_for_skills


def _cfg(store):
    return Config(database_url=store.dsn)


def _course(cid, *, skills, status="active", duplicate_of=None, rating=None,
            review_count=None, price_amount=None, price_currency=None, price_is_free=None,
            provider="IBM"):
    return PersistedCourse(
        course_id=cid, source="coursera", source_group="coursera", source_type="api",
        source_url=f"https://c.test/{cid}", name=f"Course {cid}", raw_description="d",
        content_hash=f"h_{cid}", status=status, duplicate_of=duplicate_of,
        taught_skills=skills, provider=provider, rating=rating, review_count=review_count,
        price_amount=price_amount, price_currency=price_currency, price_is_free=price_is_free,
    )


def _map(store, pairs):
    store.upsert_course_map([
        {"skill_key": k, "esco_uri": u, "method": "exact", "similarity": None,
         "esco_version": "test-1"}
        for k, u in pairs
    ])
    store.connect().commit()   # course_market reads on a SEPARATE connection


def test_by_esco_returns_only_active_canonical_courses_with_quality(store):
    store.upsert_batch([
        _course("good", skills=["accounting"], rating=4.5, review_count=120,
                price_amount=0.0, price_is_free=True),
        _course("rej", skills=["accounting"], status="rejected"),
        _course("dup", skills=["accounting"], duplicate_of="good"),
    ])
    store.connect().commit()
    _map(store, [("accounting", "uri:ACC")])

    out = courses_for_skills(["uri:ACC"], [], config=_cfg(store))
    got = out["by_esco"]["uri:ACC"]
    assert [c.course_id for c in got] == ["good"], "a rejected or duplicate course leaked"

    c = got[0]
    assert isinstance(c, CourseCandidate)
    assert c.title == "Course good" and c.url == "https://c.test/good"
    assert c.rating == 4.5 and c.review_count == 120
    assert c.price is not None and c.price.is_free is True and c.price.amount == 0.0


def test_unmapped_skill_falls_back_to_exact_taught_skill_key(store):
    store.upsert_batch([_course("np", skills=["numpy"], provider="Google")])
    store.connect().commit()
    # no course_esco_map row for numpy -> retrieval must use the exact key path

    out = courses_for_skills([], ["numpy"], config=_cfg(store))
    assert [c.course_id for c in out["by_key"]["numpy"]] == ["np"]
    # a key nobody teaches is present but empty (asked != found)
    out2 = courses_for_skills([], ["numpy", "rust"], config=_cfg(store))
    assert out2["by_key"]["rust"] == []


def test_a_course_teaching_two_requested_codes_appears_under_both(store):
    store.upsert_batch([_course("multi", skills=["sql", "python"])])
    store.connect().commit()
    _map(store, [("sql", "uri:SQL"), ("python", "uri:PY")])

    out = courses_for_skills(["uri:SQL", "uri:PY"], [], config=_cfg(store))
    assert [c.course_id for c in out["by_esco"]["uri:SQL"]] == ["multi"]
    assert [c.course_id for c in out["by_esco"]["uri:PY"]] == ["multi"]


def test_missing_price_yields_none_not_a_fabricated_object(store):
    store.upsert_batch([_course("bare", skills=["excel"])])   # all price cols null
    store.connect().commit()
    _map(store, [("excel", "uri:XL")])

    out = courses_for_skills(["uri:XL"], [], config=_cfg(store))
    c = out["by_esco"]["uri:XL"][0]
    assert c.price is None and c.rating is None
