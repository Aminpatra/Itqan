"""`GET` and `PUT /api/profile` — the half of the profile API that was missing.

The app has called both since it was rewritten. Neither existed. `getProfile`
does `.catch(() => null)` and a null profile is a legitimate answer ("nothing
confirmed yet"), so the failure was completely silent: **the profile screen
showed its empty state for every user, however complete their profile**, with
nothing in any log to say why.

The test that matters most here is
`test_editing_a_profile_does_not_start_a_new_run`. `PUT` writes the same row
`POST` does, and the difference between them is the whole design: `POST` ends
onboarding and starts the matching; `PUT` is someone fixing a typo in their phone
number. If `PUT` ever spawned a run, every edit would cost a full re-match and
change the user's dashboard underneath them.
"""

from __future__ import annotations

import time
from typing import Any


def _await_stage(client, job_id: str, *, want: set[str], timeout: float = 5.0) -> dict:
    """The worker is a thread, so poll it exactly as the UI does."""
    deadline = time.time() + timeout
    body: dict = {}
    while time.time() < deadline:
        body = client.get(f"/api/analysis/{job_id}").json()
        if body["stage"] in want:
            return body
        time.sleep(0.05)
    raise AssertionError(f"stage never reached {want}: {body}")

CONFIRMED: dict[str, Any] = {
    "fullName": "Maryam Al Balushi",
    "birthDate": None,
    "graduationDate": "2025-06",
    "phone": "+968 9123 4567",
    "skills": ["SQL", "Python"],
    "preferences": {"coursePricing": "free", "workArrangement": "remote",
                    "preferredRole": "Data Analyst", "openToOtherRoles": "yes"},
    "documentId": None,
}


def test_nothing_confirmed_yet_is_a_404_not_an_error(signed_in):
    """404 is the contract. The app renders it as an empty state, the same way
    /api/dashboard already signals "no run yet"."""
    assert signed_in.get("/api/profile").status_code == 404


def test_a_confirmed_profile_comes_back(signed_in):
    signed_in.post("/api/profile", json=CONFIRMED)

    body = signed_in.get("/api/profile").json()
    assert body["fullName"] == "Maryam Al Balushi"
    assert body["graduationDate"] == "2025-06"
    assert body["skills"] == ["SQL", "Python"]
    assert body["preferences"]["preferredRole"] == "Data Analyst"


def test_phone_survives_the_round_trip(signed_in):
    """It already persisted — `POST` stores the whole payload as jsonb — but
    there was no way to read it back, so the field was write-only."""
    signed_in.post("/api/profile", json=CONFIRMED)
    assert signed_in.get("/api/profile").json()["phone"] == "+968 9123 4567"


def test_a_profile_without_a_phone_reports_null_not_an_empty_string(signed_in):
    """`phone` is genuinely optional and the screen does not count it as missing.
    Null and "" would render differently, so the absence has to survive."""
    signed_in.post("/api/profile", json={**CONFIRMED, "phone": None})
    assert signed_in.get("/api/profile").json()["phone"] is None


def test_email_comes_from_the_account_not_the_extraction(signed_in):
    """The one field on that screen the pipeline must not be able to rewrite.

    A CV listing a university address would otherwise silently replace the
    address the person signs in with — and the profile screen is exactly where
    they would believe it.
    """
    signed_in.post("/api/profile", json={**CONFIRMED, "email": "scraped@from-the-cv.test"})

    assert signed_in.get("/api/profile").json()["email"] == "maryam@itqan.test"


def test_documents_are_listed_and_scoped_to_the_owner(signed_in, client, store):
    signed_in.post("/api/documents", files={"file": ("cv.txt", b"a real cv", "text/plain")},
                   data={"kind": "cv"})
    signed_in.post("/api/profile", json=CONFIRMED)

    docs = signed_in.get("/api/profile").json()["documents"]
    assert [d["fileName"] for d in docs] == ["cv.txt"]
    assert docs[0]["kind"] == "cv" and docs[0]["id"]

    # A second account must not see the first one's files.
    client.post("/api/logout")
    client.post("/api/auth/signup", data={"email": "other@itqan.test",
                                          "password": "Str0ng!pass", "name": "Other"})
    # Verified before it confirms anything: a fresh signup is unverified, and
    # `POST /api/profile` refuses an unverified account with 403 — which would
    # make this pass for the wrong reason, reading an empty list off a profile
    # that was never written rather than off one correctly scoped to its owner.
    store.mark_email_verified(store.user_by_email("other@itqan.test")["user_id"])
    client.post("/api/profile", json=CONFIRMED)
    assert client.get("/api/profile").json()["documents"] == []


def test_signed_out_is_401_not_404(signed_in, client):
    """A missing profile and a missing session are different answers, and the app
    routes to different screens for them."""
    signed_in.post("/api/profile", json=CONFIRMED)
    client.post("/api/logout")
    assert client.get("/api/profile").status_code == 401


# ---------------------------------------------------------------------------
# PUT — an edit, never a re-match
# ---------------------------------------------------------------------------
def test_an_edit_is_stored(signed_in):
    signed_in.post("/api/profile", json=CONFIRMED)
    res = signed_in.put("/api/profile", json={**CONFIRMED, "phone": "+968 9999 0000"})

    assert res.status_code == 200 and res.json() == {"ok": True}
    assert signed_in.get("/api/profile").json()["phone"] == "+968 9999 0000"


def test_editing_a_profile_does_not_start_a_new_run(signed_in, runner, store):
    """THE test in this file.

    `POST` is the end of onboarding and starts phase two. `PUT` is a correction.
    Correcting a birth date is not a reason to re-run the matching, and if it
    were, the dashboard would change underneath someone who only fixed a typo.
    """
    signed_in.post("/api/documents", files={"file": ("cv.txt", b"cv", "text/plain")},
                   data={"kind": "cv"})
    signed_in.post("/api/profile", json=CONFIRMED)
    calls_before = list(runner.calls)

    res = signed_in.put("/api/profile", json={**CONFIRMED, "fullName": "Maryam A. Al Balushi"})

    assert res.status_code == 200
    assert "jobId" not in res.json(), "a PUT must not hand back a job to poll"
    assert runner.calls == calls_before, "a profile edit re-ran the pipeline"


def test_an_edit_keeps_the_run_the_profile_belonged_to(signed_in, store):
    """The stored profile is bound to the run that produced it. An edit must not
    orphan that link, or `suggestedRole` and the results shown beside the profile
    stop being the ones it actually came from."""
    doc = signed_in.post("/api/documents",
                         files={"file": ("cv.txt", b"cv", "text/plain")},
                         data={"kind": "cv"}).json()["id"]
    job = signed_in.post("/api/analysis", json={"documentIds": [doc]}).json()["jobId"]
    # The run pauses for a person; confirming before it gets there would bind the
    # profile to nothing, which is the state this test exists to rule out.
    _await_stage(signed_in, job, want={"awaiting_confirmation", "failed"})
    signed_in.post("/api/profile", json=CONFIRMED)

    user_id = store.user_by_email("maryam@itqan.test")["user_id"]
    bound = store.profile(user_id)["run_id"]
    assert bound is not None, "confirming should bind the profile to its run"

    signed_in.put("/api/profile", json={**CONFIRMED, "phone": "+968 1"})

    assert store.profile(user_id)["run_id"] == bound, (
        "an edit re-bound the profile to a different run, or dropped the link")


def test_suggested_role_is_absent_until_a_run_has_produced_one(signed_in):
    """Null before any matching, and the screen says "nothing suggested yet".
    A title invented from the user's own typed preference would be presenting
    their input back to them as an agent's finding."""
    signed_in.post("/api/profile", json=CONFIRMED)
    assert signed_in.get("/api/profile").json()["suggestedRole"] is None


def test_avatar_url_is_null_until_one_is_uploaded(signed_in):
    signed_in.post("/api/profile", json=CONFIRMED)
    assert signed_in.get("/api/profile").json()["avatarUrl"] is None
