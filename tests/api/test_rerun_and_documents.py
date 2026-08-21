"""One re-run credit whichever door, and deleting an uploaded document.

`test_the_profile_path_spends_the_same_credit_as_the_chat` is the test that
carries this file. The credit used to be claimed inside `POST
/api/assistant/rerun`, so it bound the chat and nothing else — while the
Documents screen called `POST /api/analysis` through `beginReupload()` and got a
full re-read, re-confirm and re-match for free, without limit. One surface asked
permission and charged; the other did the same work repeatedly for nothing.

`test_deleting_a_document_removes_the_file_too` is the other one. A row deleted
while the bytes stay on disk is the privacy half done, and the half nobody
notices is the one still holding somebody's CV.
"""

from __future__ import annotations

from pathlib import Path

CONFIRMED = {"fullName": "Maryam Al Balushi", "skills": [], "preferences": {}}


def _upload(client, name="cv.txt", kind="cv"):
    return client.post("/api/documents", files={"file": (name, b"a cv", "text/plain")},
                       data={"kind": kind}).json()


def _finish_a_run(client):
    doc = _upload(client)["id"]
    job = client.post("/api/analysis", json={"documentIds": [doc]}).json()["jobId"]
    for _ in range(200):
        if client.get(f"/api/analysis/{job}").json()["stage"] in ("awaiting_confirmation",
                                                                 "failed"):
            break
    client.post("/api/profile", json=CONFIRMED)
    for _ in range(200):
        if client.get(f"/api/analysis/{job}").json()["stage"] in ("done", "failed"):
            break
    return doc


# ---------------------------------------------------------------------------
# one credit, whichever door
# ---------------------------------------------------------------------------
def test_the_first_analysis_is_free(signed_in):
    """No completed run means this is onboarding, not a re-run. Charging for it
    would mean a new account spent its weekly credit to see any results at all."""
    doc = _upload(signed_in)["id"]
    assert signed_in.post("/api/analysis", json={"documentIds": [doc]}).status_code == 200


def test_a_second_analysis_costs_the_weekly_credit(signed_in):
    doc = _finish_a_run(signed_in)

    again = signed_in.post("/api/analysis", json={"documentIds": [doc]})
    assert again.status_code == 200, "the first re-run should be allowed"

    third = signed_in.post("/api/analysis", json={"documentIds": [doc]})
    assert third.status_code == 429
    assert third.json()["error"] == "rerun_limit_reached"


def test_the_profile_path_spends_the_same_credit_as_the_chat(signed_in):
    """THE test. Spend it in the chat, then try from the profile and be refused.

    Before this, the Documents screen re-read, re-confirmed and re-matched for
    free and without limit, while the chat charged for exactly the same work.
    """
    doc = _finish_a_run(signed_in)

    spent = signed_in.post("/api/assistant/rerun", json={"confirm": True, "mode": "match"})
    assert spent.status_code == 200, spent.text

    blocked = signed_in.post("/api/analysis", json={"documentIds": [doc]})
    assert blocked.status_code == 429, "the profile path ignored the spent credit"


def test_and_the_other_way_round(signed_in):
    doc = _finish_a_run(signed_in)
    assert signed_in.post("/api/analysis", json={"documentIds": [doc]}).status_code == 200

    refused = signed_in.post("/api/assistant/rerun", json={"confirm": True, "mode": "match"})
    assert refused.status_code == 429


def test_the_assistants_own_rerun_is_not_charged_twice(signed_in, store):
    """`full` mode claims before spawning and does NOT go through /api/analysis.
    Pinned because "these two paths do not overlap" is exactly the kind of claim
    that quietly stops being true."""
    from api.assistant import week_start

    _finish_a_run(signed_in)
    signed_in.post("/api/assistant/rerun", json={"confirm": True, "mode": "full"})

    user = store.user_by_email("maryam@itqan.test")
    used = store.quota_used(user["user_id"], kind="rerun",
                            period_start=week_start(signed_in.app.state.config))
    assert used == 1, f"one re-run should cost one credit, spent {used}"


# ---------------------------------------------------------------------------
# deleting a document
# ---------------------------------------------------------------------------
def test_deleting_a_document_removes_the_file_too(signed_in, store):
    """A row deleted while the bytes stay is the privacy half done."""
    doc = _upload(signed_in)
    user = store.user_by_email("maryam@itqan.test")
    stored = Path(store.document(doc["id"], user["user_id"])["stored_path"])
    assert stored.exists(), "the fixture never wrote a file"

    assert signed_in.delete(f"/api/documents/{doc['id']}").status_code == 204

    assert store.document(doc["id"], user["user_id"]) is None
    assert not stored.exists(), "the row went and the file stayed"


def test_a_deleted_document_leaves_the_list(signed_in):
    """Read through /api/profile, which is the list the screen actually renders.

    A confirmed profile is needed first, because that route answers 404 until
    there is one — the empty-state contract, not a failure.
    """
    _finish_a_run(signed_in)
    doc = _upload(signed_in, name="extra.txt", kind="transcript")
    assert doc["id"] in [d["id"] for d in signed_in.get("/api/profile").json()["documents"]]

    signed_in.delete(f"/api/documents/{doc['id']}")
    assert doc["id"] not in [d["id"] for d in signed_in.get("/api/profile").json()["documents"]]


def test_another_accounts_document_is_404_not_403(signed_in, client, store):
    """Confirming that a document exists is itself a disclosure, so a foreign id
    and a missing one must be indistinguishable."""
    doc = _upload(signed_in)

    client.post("/api/logout")
    client.post("/api/auth/signup", data={"email": "other@itqan.test",
                                          "password": "Str0ng!pass", "name": "Other"})
    store.mark_email_verified(store.user_by_email("other@itqan.test")["user_id"])

    assert client.delete(f"/api/documents/{doc['id']}").status_code == 404
    # And it is still there for its owner.
    user = store.user_by_email("maryam@itqan.test")
    assert store.document(doc["id"], user["user_id"]) is not None


def test_deleting_a_document_does_not_invalidate_a_finished_run(signed_in):
    """Results are already stored as an envelope. Removing the source changes
    nothing about what was concluded — only what a later re-run would read."""
    doc = _finish_a_run(signed_in)
    before = signed_in.get("/api/dashboard").json()

    signed_in.delete(f"/api/documents/{doc}")

    assert signed_in.get("/api/dashboard").json()["readiness"] == before["readiness"]


def test_deleting_needs_a_session(client):
    assert client.delete("/api/documents/doc_whatever").status_code == 401


# ---------------------------------------------------------------------------
# adding a transcript to a profile that already holds a CV
# ---------------------------------------------------------------------------
def _wait_for_read(client, runner, job):
    """Wait until Agent A has been handed its files, and say so if it never was.

    Sleeps rather than spinning: a tight polling loop starves the worker thread on
    a single core, so the test would reach its assertions while the run was still
    going and tear the database connection out from under it.
    """
    import time

    deadline = time.time() + 10.0
    while time.time() < deadline:
        stage = client.get(f"/api/analysis/{job}").json()["stage"]
        if stage in ("awaiting_confirmation", "done", "failed"):
            assert stage != "failed", "the run failed before Agent A was reached"
            return runner.read
        time.sleep(0.02)
    raise AssertionError("the run never reached the confirm step")


def test_a_transcript_can_be_added_when_the_cv_is_already_on_file(signed_in, runner):
    """THE test. Someone who onboarded with a CV, and later wants their transcript
    read, submits ONE id and it is not a CV.

    That used to be a 400 `cv_required` — telling a person whose CV we are holding
    that a CV is required. A run is built from what is ON FILE, which is how the
    assistant's full re-run has always worked.
    """
    _finish_a_run(signed_in)
    transcript = _upload(signed_in, name="transcript.pdf", kind="transcript")["id"]

    res = signed_in.post("/api/analysis", json={"documentIds": [transcript]})
    assert res.status_code == 200, res.text

    read = _wait_for_read(signed_in, runner, res.json()["jobId"])
    # Asserted on the FILES, not the status code: a 200 would pass just as well
    # with the transcript dropped on the floor.
    assert len(read["cv"]) == 1, "the CV already on file was not read"
    assert len(read["transcript"]) == 1, "the transcript that prompted this was not read"


def test_the_run_records_the_cv_it_actually_read(signed_in, store):
    """The fallback CV was never named in the request, so recording only the
    submitted ids would leave the run claiming it read documents it did not."""
    cv = _finish_a_run(signed_in)
    transcript = _upload(signed_in, name="t.pdf", kind="transcript")["id"]

    job = signed_in.post("/api/analysis", json={"documentIds": [transcript]}).json()["jobId"]

    user = store.user_by_email("maryam@itqan.test")
    recorded = store.run(job_id=job, user_id=user["user_id"])["document_ids"]
    assert cv in recorded and transcript in recorded


def test_an_account_with_no_cv_at_all_is_still_refused(signed_in):
    """The guard is narrowed, not removed. Agent A genuinely cannot run without
    a CV, and saying so here beats failing three stages in."""
    transcript = _upload(signed_in, name="transcript.pdf", kind="transcript")["id"]

    res = signed_in.post("/api/analysis", json={"documentIds": [transcript]})
    assert res.status_code == 400
    assert res.json()["error"] == "cv_required"


def test_two_cvs_do_not_blend_into_one_profile(signed_in, runner, store):
    """Only the NEWEST is read.

    Showing what is already on file means replacing a CV submits the old one
    alongside the new one. Reading both would merge two resumes into a profile
    describing nobody — and it fails silently: no error, just details belonging to
    a person who does not exist.
    """
    old = _upload(signed_in, name="old-cv.pdf")["id"]
    new = _upload(signed_in, name="new-cv.pdf")["id"]

    job = signed_in.post("/api/analysis",
                         json={"documentIds": [old, new]}).json()["jobId"]
    read = _wait_for_read(signed_in, runner, job)

    assert len(read["cv"]) == 1, f"two resumes were read together: {read['cv']}"
    user = store.user_by_email("maryam@itqan.test")
    assert read["cv"][0] == store.document(new, user["user_id"])["stored_path"], \
        "the older CV won; `documents()` is not ordering newest-first"


def test_adding_a_transcript_spends_the_weekly_credit(signed_in, runner):
    """Decided 2026-08-21: it runs the whole pipeline, so it costs what any other
    re-run costs. Free-on-new-document would make any upload an unlimited re-run,
    which is the hole `test_the_profile_path_spends_the_same_credit_as_the_chat`
    exists to keep closed."""
    _finish_a_run(signed_in)
    transcript = _upload(signed_in, name="t.pdf", kind="transcript")["id"]

    spent = signed_in.post("/api/analysis", json={"documentIds": [transcript]})
    assert spent.status_code == 200
    # Let that run reach the pause before the fixture tears the database
    # connection out from under its worker thread.
    _wait_for_read(signed_in, runner, spent.json()["jobId"])

    again = signed_in.post("/api/analysis", json={"documentIds": [transcript]})
    assert again.status_code == 429
    assert again.json()["error"] == "rerun_limit_reached"
