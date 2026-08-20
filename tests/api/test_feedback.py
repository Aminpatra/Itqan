"""Likes, dislikes, and the thing that makes them mean something.

`test_a_disliked_job_disappears_from_the_list` is the test that carries this
file. Storing a verdict and still showing the same card after a reload is what
`BACKEND.md` §1.5 warns against in as many words — it reads as the product
ignoring the person — so the endpoint existing is not the feature. The exclusion
is.

`test_withdrawing_a_dislike_brings_the_card_back` is its necessary other half.
Filtering is invisible: the card is gone and so is the control that removed it,
so if a mis-tap were permanent the person would have no way back at all.
"""

from __future__ import annotations

import pytest

CONFIRMED = {"fullName": "Maryam Al Balushi", "birthDate": None,
             "graduationDate": "2025-06", "phone": None, "skills": ["SQL"],
             "preferences": {"coursePricing": "free", "workArrangement": "remote",
                             "knowsRole": "yes", "preferredRole": "Data Analyst",
                             "openToOtherRoles": "yes"},
             "documentId": None}


def _feedback(client, **kw):
    body = {"subject": "job", "verdict": "dislike", "itemId": "jp_991", **kw}
    return client.post("/api/preferences/feedback", json=body)


def _finished_run(client):
    """Drive a real run to `done`, so /api/jobs and /api/courses have rows.

    The same shape `test_assistant.py` uses: the worker is a thread, so the two
    phases are polled exactly as the UI polls them rather than assumed.
    """
    doc = client.post("/api/documents", files={"file": ("cv.txt", b"cv", "text/plain")},
                      data={"kind": "cv"}).json()["id"]
    job = client.post("/api/analysis", json={"documentIds": [doc]}).json()["jobId"]
    for _ in range(200):
        if client.get(f"/api/analysis/{job}").json()["stage"] in ("awaiting_confirmation",
                                                                 "failed"):
            break
    client.post("/api/profile", json=CONFIRMED)
    for _ in range(200):
        if client.get(f"/api/analysis/{job}").json()["stage"] in ("done", "failed"):
            break
    return job


# ---------------------------------------------------------------------------
# recording
# ---------------------------------------------------------------------------
def test_a_new_account_has_no_opinions(signed_in):
    res = signed_in.get("/api/preferences/feedback")
    assert res.status_code == 200
    assert res.json() == {"jobs": {}, "courses": {}}


def test_a_verdict_is_readable_back(signed_in):
    assert _feedback(signed_in, verdict="like").status_code == 200
    assert signed_in.get("/api/preferences/feedback").json()["jobs"]["jp_991"] == "like"


def test_the_latest_verdict_wins(signed_in):
    _feedback(signed_in, verdict="like")
    _feedback(signed_in, verdict="dislike")
    assert signed_in.get("/api/preferences/feedback").json()["jobs"]["jp_991"] == "dislike"


def test_the_history_survives_underneath(signed_in, store):
    """Append-only: "liked it, then changed my mind after reading the location"
    is a different signal from "disliked it", and an UPDATE would erase it."""
    _feedback(signed_in, verdict="like")
    _feedback(signed_in, verdict="dislike")

    rows = store._all("SELECT verdict FROM app_feedback ORDER BY created_at")
    assert [r["verdict"] for r in rows] == ["like", "dislike"]


@pytest.mark.parametrize("bad", [
    {"subject": "posting"}, {"verdict": "meh"}, {"itemId": "  "},
])
def test_a_meaningless_row_is_refused(signed_in, bad):
    assert _feedback(signed_in, **bad).status_code == 400


def test_an_unknown_reason_keeps_the_verdict_and_drops_the_reason(signed_in, store):
    """The client sends this fire-and-forget and swallows failures, so a 400
    would discard the thumb along with the reason — losing the part worth having
    to reject the part that was optional."""
    assert _feedback(signed_in, reason="theVibes").status_code == 200

    row = store._one("SELECT verdict, reason FROM app_feedback")
    assert row["verdict"] == "dislike" and row["reason"] is None


def test_a_note_is_kept_only_alongside_other(signed_in, store):
    _feedback(signed_in, reason="wrongLocation", note="too far")
    assert store._one("SELECT note FROM app_feedback")["note"] is None

    _feedback(signed_in, reason="other", note="too far to commute")
    row = store._one("SELECT note FROM app_feedback ORDER BY created_at DESC LIMIT 1")
    assert row["note"] == "too far to commute"


def test_feedback_needs_a_session(client):
    assert client.post("/api/preferences/feedback",
                       json={"subject": "job", "verdict": "like",
                             "itemId": "x"}).status_code == 401


def test_one_account_cannot_see_anothers(signed_in, client, store):
    _feedback(signed_in, verdict="like")

    client.post("/api/logout")
    client.post("/api/auth/signup", data={"email": "other@itqan.test",
                                          "password": "Str0ng!pass", "name": "Other"})
    store.mark_email_verified(store.user_by_email("other@itqan.test")["user_id"])
    assert client.get("/api/preferences/feedback").json() == {"jobs": {}, "courses": {}}


# ---------------------------------------------------------------------------
# the exclusion — the reason any of this exists
# ---------------------------------------------------------------------------
def test_a_disliked_job_disappears_from_the_list(signed_in):
    """THE test. A verdict that changes nothing reappears as the same card after
    a reload, which reads as the product ignoring the person."""
    _finished_run(signed_in)
    before = signed_in.get("/api/jobs").json()
    assert before, "no jobs to filter — the fixture run produced nothing"

    _feedback(signed_in, itemId=before[0]["id"], verdict="dislike")

    after = signed_in.get("/api/jobs").json()
    assert before[0]["id"] not in [j["id"] for j in after]
    assert len(after) == len(before) - 1


def test_withdrawing_a_dislike_brings_the_card_back(signed_in):
    """Filtering is invisible: the card is gone and so is the control that
    removed it. If a mis-tap were permanent there would be no way back."""
    _finished_run(signed_in)
    target = signed_in.get("/api/jobs").json()[0]["id"]

    _feedback(signed_in, itemId=target, verdict="dislike")
    assert target not in [j["id"] for j in signed_in.get("/api/jobs").json()]

    _feedback(signed_in, itemId=target, verdict="like")
    assert target in [j["id"] for j in signed_in.get("/api/jobs").json()]


def test_a_like_does_not_remove_anything(signed_in):
    _finished_run(signed_in)
    before = signed_in.get("/api/jobs").json()
    _feedback(signed_in, itemId=before[0]["id"], verdict="like")
    assert len(signed_in.get("/api/jobs").json()) == len(before)


def test_a_disliked_course_disappears_too(signed_in):
    _finished_run(signed_in)
    before = signed_in.get("/api/courses").json()
    if not before:
        pytest.skip("the fixture run recommended no courses")

    _feedback(signed_in, subject="course", itemId=before[0]["id"], verdict="dislike")
    assert before[0]["id"] not in [c["id"] for c in signed_in.get("/api/courses").json()]


def test_one_persons_dislike_does_not_hide_anothers_card(signed_in, client, store):
    """The filter is per account. Sharing it would let one user quietly edit
    everyone else's results."""
    _finished_run(signed_in)
    target = signed_in.get("/api/jobs").json()[0]["id"]
    _feedback(signed_in, itemId=target, verdict="dislike")

    client.post("/api/logout")
    client.post("/api/auth/signup", data={"email": "other2@itqan.test",
                                          "password": "Str0ng!pass", "name": "Other"})
    store.mark_email_verified(store.user_by_email("other2@itqan.test")["user_id"])
    _finished_run(client)

    assert target in [j["id"] for j in client.get("/api/jobs").json()]


# ---------------------------------------------------------------------------
# knowsRole
# ---------------------------------------------------------------------------
def test_knows_role_survives_the_round_trip(signed_in):
    """It rides in the preferences payload, which `stored_profile` spreads back
    whole. Pinned because a whitelist added there later would drop it silently."""
    signed_in.post("/api/profile", json={**CONFIRMED,
                                         "preferences": {**CONFIRMED["preferences"],
                                                         "knowsRole": "no"}})
    prefs = signed_in.get("/api/profile").json()["preferences"]
    assert prefs["knowsRole"] == "no"


def test_not_asked_is_distinguishable_from_no(signed_in):
    """`null` means the question was never put to them — every account that
    onboarded before it existed. Reading that as 'no' would have Hud offer to
    explore roles to someone who never said they were unsure."""
    signed_in.post("/api/profile", json={**CONFIRMED,
                                         "preferences": {**CONFIRMED["preferences"],
                                                         "knowsRole": None}})
    assert signed_in.get("/api/profile").json()["preferences"]["knowsRole"] is None


def test_a_disliked_job_leaves_the_dashboard_too(signed_in):
    """The miss that caused this bug. `/api/jobs` and `/api/courses` were
    filtered and the dashboard was not — so a disliked posting vanished from the
    Jobs screen and carried on sitting on the front page, which is the screen
    people read first.

    Asserted on `topMatches`, the surface that was wrong, rather than on the one
    that was already right.
    """
    _finished_run(signed_in)
    top = signed_in.get("/api/dashboard").json()["topMatches"]
    if not top:
        pytest.skip("the fixture run matched no jobs")

    _feedback(signed_in, itemId=top[0]["id"], verdict="dislike")

    after = signed_in.get("/api/dashboard").json()["topMatches"]
    assert top[0]["id"] not in [j["id"] for j in after]


def test_the_dashboard_still_offers_two_matches_when_it_can(signed_in):
    """Filtered BEFORE the limit, not after. Taking the top two and then dropping
    a disliked one leaves a single card where two were asked for, which reads as
    "we ran out of matches" rather than "you hid one"."""
    _finished_run(signed_in)
    all_jobs = signed_in.get("/api/jobs").json()
    if len(all_jobs) < 3:
        pytest.skip("needs at least three matches to tell the two cases apart")

    _feedback(signed_in, itemId=all_jobs[0]["id"], verdict="dislike")
    assert len(signed_in.get("/api/dashboard").json()["topMatches"]) == 2
