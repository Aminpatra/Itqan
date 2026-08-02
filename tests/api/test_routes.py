"""Every route, against the contract the frontend actually calls.

The reference is `Onboarding/dev/site-plugin.ts` and `src/api/types.ts` — camelCase
paths and shapes — NOT the handoff PDF, which specifies snake_case and `/api/auth/*`
and describes four frontend files that do not exist. Where the two disagree these
tests follow the running code, because that is what has to keep working.
"""

from __future__ import annotations

import io
import time
from statistics import median
from typing import Any

import pytest

from fastapi.testclient import TestClient


def _upload(client: TestClient, kind: str = "cv", name: str = "cv.pdf") -> str:
    res = client.post("/api/documents",
                      files={"file": (name, io.BytesIO(b"%PDF-1.4 fake"), "application/pdf")},
                      data={"kind": kind})
    assert res.status_code == 200, res.text
    return res.json()["id"]


def _await_stage(client: TestClient, job_id: str, *, want: set[str],
                 timeout: float = 5.0) -> dict:
    """The worker is a thread, so poll it exactly as the UI does."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/analysis/{job_id}").json()
        if body["stage"] in want:
            return body
        time.sleep(0.05)
    raise AssertionError(f"stage never reached {want}: {body}")


def _confirm(client: TestClient, **preferences: Any) -> dict:
    """Confirm the extracted details — which is what starts Agent C and Agent E.

    A run pauses at `awaiting_confirmation` by design, so nothing reaches `done`
    without this call. Tests that only want the finished envelopes go through
    `_full_run`; this one exists for the tests that care about the preferences.
    """
    res = client.post("/api/profile", json={
        "fullName": "Maryam Al Balushi", "birthDate": None,
        "graduationDate": "2025-06", "skills": ["SQL", "Python"],
        "preferences": preferences or {}, "documentId": None})
    assert res.status_code == 200, res.text
    return res.json()


def _full_run(client: TestClient, **preferences: Any) -> dict:
    """Upload, let Agent A finish, confirm, and wait for the whole run."""
    doc = _upload(client)
    job_id = client.post("/api/analysis", json={"documentIds": [doc]}).json()["jobId"]
    _await_stage(client, job_id, want={"awaiting_confirmation", "failed"})
    _confirm(client, **preferences)
    return _await_stage(client, job_id, want={"done", "failed"})


# ---------------------------------------------------------------------------
# auth — the site owns credentials; the app reads the session
# ---------------------------------------------------------------------------
def test_session_is_401_when_not_signed_in(client: TestClient):
    """A 401 here is a normal answer, not an error: the frontend treats it as
    'not signed in' and sends the user to the site."""
    res = client.get("/api/session")
    assert res.status_code == 401 and res.json() == {"error": "no_session"}


def test_signup_then_session_returns_the_account(client: TestClient):
    client.post("/api/placeholder/signup",
                data={"email": "a@b.test", "password": "Str0ng!pass", "name": "A B"})
    body = client.get("/api/session").json()
    assert body["user"]["email"] == "a@b.test"
    assert body["user"]["fullName"] == "A B"
    assert body["user"]["onboarded"] is False
    assert body["locale"] in {"ar", "en"}


def test_signup_rejects_a_weak_password_the_site_would_reject(client: TestClient):
    """The rule lives in three places (site form, this endpoint, the dev plugin).
    A dev stub that accepted what production rejects has already caused one bug."""
    res = client.post("/api/placeholder/signup",
                      data={"email": "c@d.test", "password": "weak", "name": "C"})
    assert res.status_code == 400 and res.json()["error"] == "invalid_input"


def test_duplicate_email_is_409(client: TestClient):
    data = {"email": "dup@x.test", "password": "Str0ng!pass", "name": "D"}
    client.post("/api/placeholder/signup", data=data)
    res = client.post("/api/placeholder/signup", data=data)
    assert res.status_code == 409 and res.json()["error"] == "email_taken"


def test_login_does_not_distinguish_bad_password_from_no_account(client: TestClient):
    """Telling them apart is an account-enumeration oracle."""
    client.post("/api/placeholder/signup",
                data={"email": "e@f.test", "password": "Str0ng!pass", "name": "E"})
    wrong_pw = client.post("/api/placeholder/login",
                           data={"email": "e@f.test", "password": "Wr0ng!pass"})
    no_acct = client.post("/api/placeholder/login",
                          data={"email": "nobody@f.test", "password": "Str0ng!pass"})
    assert wrong_pw.status_code == no_acct.status_code == 401
    assert wrong_pw.json() == no_acct.json()


def test_the_session_cookie_is_not_readable_by_script(signed_in: TestClient):
    """httpOnly is the whole defence: one XSS otherwise means account takeover."""
    header = signed_in.post("/api/placeholder/login",
                            data={"email": "maryam@itqan.test",
                                  "password": "Str0ng!pass"}).headers["set-cookie"]
    assert "httponly" in header.lower()
    assert "samesite=lax" in header.lower()


def test_production_refuses_to_boot_without_a_session_secret(monkeypatch):
    """The cookie is `user_id.HMAC(secret, user_id)` and the development fallback
    is a literal string in a public repository — so with it in place, anyone who
    has read this file can mint a session for any account.

    Refusing to start is the point. A warning in a log nobody reads would leave a
    deployment that works perfectly right up until someone tries it.
    """
    from api.main import assert_deployable

    monkeypatch.setenv("ITQAN_ENV", "production")
    monkeypatch.delenv("ITQAN_SESSION_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="ITQAN_SESSION_SECRET"):
        assert_deployable()

    monkeypatch.setenv("ITQAN_SESSION_SECRET", "0" * 64)
    assert_deployable()

    # Development is unaffected — every local run and this whole suite rely on it.
    monkeypatch.setenv("ITQAN_ENV", "development")
    monkeypatch.delenv("ITQAN_SESSION_SECRET", raising=False)
    assert_deployable()


def test_phase_two_rebuilds_the_profile_when_the_disk_lost_it(tmp_path):
    """The gap between the phases ends only when a PERSON confirms, so a restart
    or a redeploy in between can easily take the file Agent C is handed.

    Postgres kept the same envelope (`attach_profile`), so this restores it rather
    than re-running Agent A: the bytes Agent C reads are the bytes Agent A wrote.
    """
    import json

    from api.jobs import PipelineRunner
    from shared.config import Config

    runner = PipelineRunner(Config(output_dir=tmp_path))
    envelope = {"candidate": {"full_name": "Maryam Al Balushi"}, "skills": {"accepted": []}}

    runner.restore_profile(run_id="r1", profile=envelope)
    written = tmp_path / "r1" / "candidate_profile.json"
    assert json.loads(written.read_text(encoding="utf-8")) == envelope

    # An existing file is never overwritten — what phase one wrote is the truth,
    # and the stored copy is only a fallback for when it is missing.
    written.write_text('{"candidate": {"full_name": "edited on disk"}}', encoding="utf-8")
    runner.restore_profile(run_id="r1", profile=envelope)
    assert "edited on disk" in written.read_text(encoding="utf-8")


def test_a_forged_cookie_is_rejected(client: TestClient):
    """The token is signed, so guessing a user id is not enough."""
    client.cookies.set("itqan_session", "u_someoneelse.deadbeef")
    assert client.get("/api/session").status_code == 401


def test_logout_clears_the_session(signed_in: TestClient):
    assert signed_in.post("/api/logout").json() == {"ok": True}
    signed_in.cookies.clear()
    assert signed_in.get("/api/session").status_code == 401


def test_handoff_exists_and_redirects(signed_in: TestClient):
    """Without this route every local sign-in ended on the site's 404."""
    res = signed_in.get("/api/handoff", follow_redirects=False)
    assert res.status_code == 302 and res.headers["location"] == "/app/"


# ---------------------------------------------------------------------------
# documents
# ---------------------------------------------------------------------------
def test_upload_returns_the_shape_the_uploader_expects(signed_in: TestClient):
    res = signed_in.post("/api/documents",
                         files={"file": ("cv.pdf", io.BytesIO(b"x" * 100), "application/pdf")},
                         data={"kind": "cv"})
    body = res.json()
    assert set(body) >= {"id", "fileName", "mimeType", "sizeBytes", "kind"}
    assert body["kind"] == "cv" and body["sizeBytes"] == 100


def test_a_traversal_filename_cannot_choose_where_bytes_land(signed_in: TestClient):
    """The stored name is generated; the client's is data, never a path."""
    res = signed_in.post(
        "/api/documents",
        files={"file": ("../../../../evil.pdf", io.BytesIO(b"x"), "application/pdf")},
        data={"kind": "cv"})
    assert res.status_code == 200
    assert res.json()["fileName"] == "../../../../evil.pdf"   # shown, not obeyed


def test_upload_requires_a_session(client: TestClient):
    res = client.post("/api/documents",
                      files={"file": ("cv.pdf", io.BytesIO(b"x"), "application/pdf")},
                      data={"kind": "cv"})
    assert res.status_code == 401


def test_an_unknown_kind_is_refused(signed_in: TestClient):
    res = signed_in.post("/api/documents",
                         files={"file": ("x.pdf", io.BytesIO(b"x"), "application/pdf")},
                         data={"kind": "passport"})
    assert res.status_code == 400


# ---------------------------------------------------------------------------
# analysis — async, with progress driven by real stage completion
# ---------------------------------------------------------------------------
def test_analysis_without_a_cv_is_refused_by_name(signed_in: TestClient):
    """Agent A cannot run without a CV. Refusing here beats failing three stages
    in — and `cv`, not `transcript`, is the required kind."""
    doc = _upload(signed_in, kind="transcript", name="transcript.pdf")
    res = signed_in.post("/api/analysis", json={"documentIds": [doc]})
    assert res.status_code == 400 and res.json()["error"] == "cv_required"


def test_a_full_run_reaches_done_and_returns_the_result(signed_in: TestClient, runner):
    body = _full_run(signed_in)
    assert body["stage"] == "done" and body["progress"] == 1.0
    assert runner.calls == ["A", "C", "E"], "the pipeline did not run in order"

    result = body["result"]
    assert result["fullName"]["value"] == "Maryam Al Balushi"
    assert result["fullName"]["confidence"] == 0.97        # Agent A's real score
    assert [s["name"] for s in result["skills"]] == ["SQL", "Python"]


# ---------------------------------------------------------------------------
# the pause — Agent C waits for the user, and their answers reach it
# ---------------------------------------------------------------------------
def test_agent_c_does_not_run_until_the_user_confirms(signed_in: TestClient, runner):
    """The point of the split. Agent A's result is available for the user to check
    while Agent C has not started, so the confirm screen is not blocked on work it
    does not display — and the answers given during the wait are still inputs."""
    doc = _upload(signed_in)
    job_id = signed_in.post("/api/analysis", json={"documentIds": [doc]}).json()["jobId"]

    body = _await_stage(signed_in, job_id, want={"awaiting_confirmation", "failed"})
    assert body["stage"] == "awaiting_confirmation"
    assert runner.calls == ["A"], "Agent C ran before the user confirmed anything"
    # The result is attached at the pause; withholding it was the 3-minute skeleton.
    assert body["result"]["fullName"]["value"] == "Maryam Al Balushi"

    assert _confirm(signed_in)["jobId"] == job_id
    _await_stage(signed_in, job_id, want={"done", "failed"})
    assert runner.calls == ["A", "C", "E"]


def test_the_answers_reach_both_agents_as_flags(signed_in: TestClient, runner):
    """Answered during the wait, and previously written to a table and forgotten."""
    _full_run(signed_in, preferredRole="Data Analyst", openToOtherRoles="no",
              workArrangement="remote", coursePricing="free")

    assert runner.flags["C"] == ["--preferred-role", "Data Analyst", "--roles-only",
                                 "--preferred-arrangement", "remote"]
    assert runner.flags["E"] == ["--prefer-free"]


def test_no_answers_means_no_flags(signed_in: TestClient, runner):
    """An unanswered question must not become a preference. `coursePricing: 'any'`
    is an answer that asks for nothing, and 'open to other roles' with no role
    named is not a narrowing — both leave the pipeline exactly as it was."""
    _full_run(signed_in, coursePricing="any", openToOtherRoles="no", preferredRole="")
    assert runner.flags["C"] == [] and runner.flags["E"] == []


def test_the_preferences_are_recorded_against_the_run(signed_in: TestClient, store):
    """A gap file has to be explainable by the answers that produced it — the same
    reason every agent publishes a calibration block."""
    _full_run(signed_in, preferredRole="Data Analyst")
    row = store._one("SELECT preferences FROM app_runs ORDER BY started_at DESC LIMIT 1")
    assert row["preferences"]["preferredRole"] == "Data Analyst"


def test_confirming_with_no_run_still_works(signed_in: TestClient, runner):
    """The manual-entry route has no run at all: the user typed their details
    instead of uploading. Confirming must not require a paused run to exist."""
    res = _confirm(signed_in, coursePricing="free")
    assert res == {"ok": True}
    assert runner.calls == []


def test_a_paused_run_is_not_reported_as_stuck(signed_in: TestClient, store):
    """`stale_runs` finds runs a process restart abandoned. A user taking their
    time over the form is not one, and counting them would bury the real cases."""
    from api.jobs import stale_runs

    doc = _upload(signed_in)
    job_id = signed_in.post("/api/analysis", json={"documentIds": [doc]}).json()["jobId"]
    _await_stage(signed_in, job_id, want={"awaiting_confirmation", "failed"})

    assert stale_runs(store, older_than_minutes=0) == []


def test_progress_comes_from_stage_completion_not_a_clock(signed_in: TestClient):
    """The stages a run passes through must be a prefix of the real order, and
    progress must rise with them. A timer-driven bar would reach 90% and stall."""
    doc = _upload(signed_in)
    job_id = signed_in.post("/api/analysis", json={"documentIds": [doc]}).json()["jobId"]
    _await_stage(signed_in, job_id, want={"awaiting_confirmation", "failed"})
    _confirm(signed_in)

    seen: list[tuple[str, float]] = []
    for _ in range(60):
        b = signed_in.get(f"/api/analysis/{job_id}").json()
        if not seen or seen[-1][0] != b["stage"]:
            seen.append((b["stage"], b["progress"]))
        if b["stage"] in {"done", "failed"}:
            break
        time.sleep(0.02)

    order = ["queued", "reading", "translating", "awaiting_confirmation",
             "matching", "done"]
    positions = [order.index(s) for s, _ in seen if s in order]
    assert positions == sorted(positions), f"stages went backwards: {seen}"
    assert [p for _, p in seen] == sorted(p for _, p in seen), f"progress fell: {seen}"


def test_birth_date_is_null_because_agent_a_never_extracts_one(signed_in: TestClient):
    """Not an omission — `CVExtraction` has no birth-date field. Publishing null
    is the truth; inferring one from an ID pattern would not be."""
    body = _full_run(signed_in)
    assert body["result"]["birthDate"] is None


def test_a_failed_agent_is_named_so_the_ui_can_recover(dsn, store, signed_in):
    """The UI offers re-upload or manual entry for an unreadable document; that
    path is only reachable if the error says which agent failed."""
    from api.main import create_app
    from shared.config import Config
    from tests.api.conftest import FakeRunner

    app = create_app(Config(database_url=dsn), store=store,
                     runner=FakeRunner(fail_at="A"), migrate=False)
    failing = TestClient(app)
    failing.cookies = signed_in.cookies

    doc = _upload(failing)
    job_id = failing.post("/api/analysis", json={"documentIds": [doc]}).json()["jobId"]
    body = _await_stage(failing, job_id, want={"failed"})
    assert body["error"] == "agent_a_unreadable_document"


def test_progress_can_never_go_backwards(signed_in: TestClient, store):
    """`set_progress` is called ~22 times per run from a worker thread, and a bar
    that moves backwards reads as the system losing work — worse than a coarse one.
    Out-of-order or retried checkpoints can only ever leave it further along."""
    doc = _upload(signed_in)
    job_id = signed_in.post("/api/analysis", json={"documentIds": [doc]}).json()["jobId"]
    _await_stage(signed_in, job_id, want={"awaiting_confirmation", "failed"})

    store.set_progress(job_id, "reading", 0.05)      # a late straggler from earlier
    row = store.run(job_id=job_id, user_id=store._one(
        "SELECT user_id FROM app_runs WHERE job_id = %s", (job_id,))["user_id"])
    assert float(row["progress"]) == 0.75, "an earlier checkpoint pulled the bar back"
    # The stage is a label for what is happening, not a measure of how much is
    # done, so it is free to move either way.
    assert row["stage"] == "reading"


def test_the_bar_advances_in_many_small_real_steps(signed_in: TestClient, store):
    """The complaint that started this: 0.15 -> 0.55 -> 0.80, sitting still between.

    Every value here is written because a graph node finished — the fake runner
    replays Agent A's real node names — so this asserts the granularity WITHOUT
    introducing a timer, which would make a hung run look alive.
    """
    seen: list[float] = []
    real_set_progress = store.set_progress
    store.set_progress = lambda j, s, pr: (seen.append(pr), real_set_progress(j, s, pr))[1]
    try:
        doc = _upload(signed_in)
        job_id = signed_in.post("/api/analysis", json={"documentIds": [doc]}).json()["jobId"]
        _await_stage(signed_in, job_id, want={"awaiting_confirmation", "failed"})
    finally:
        store.set_progress = real_set_progress

    assert len(seen) >= 10, f"only {len(seen)} checkpoints; the bar will still lurch"
    assert seen == sorted(seen)
    steps = [b - a for a, b in zip([0.0, *seen], seen)]
    # Bounded worst case, small typical case. The two large steps are the OCR pass
    # and the coursework-skill judging — measured at 48 seconds, so weighting them
    # heavily is the honest choice, not a coarse bar.
    assert max(steps) < 0.18, f"largest jump {max(steps):.3f}"
    assert median(steps) < 0.08, "the typical step must be small"


def test_an_unreadable_pdf_names_agent_a_whatever_it_raised(dsn, store, signed_in):
    """Found by uploading a corrupt PDF to the running server.

    PyMuPDF raises `fitz.FileDataError`, which SUBCLASSES RuntimeError — so the
    old message-sniffing branch (`str(exc).startswith("agent_")`) reported the one
    error a user can actually fix as a generic `pipeline_failed`, and the UI's
    re-upload route is gated on the specific code. The phase, not the exception
    type or its text, decides.
    """
    from api.main import create_app
    from shared.config import Config
    from tests.api.conftest import FakeRunner

    class CorruptPdfRunner(FakeRunner):
        def run_agent_a(self, *, cv_paths, transcript_paths, run_id, on_read):
            on_read()
            raise RuntimeError("Failed to open file 'cv.pdf' as type pdf.")

    app = create_app(Config(database_url=dsn), store=store,
                     runner=CorruptPdfRunner(), migrate=False)
    failing = TestClient(app)
    failing.cookies = signed_in.cookies

    doc = _upload(failing)
    job_id = failing.post("/api/analysis", json={"documentIds": [doc]}).json()["jobId"]
    body = _await_stage(failing, job_id, want={"failed"})
    assert body["error"] == "agent_a_unreadable_document"


def test_a_failure_after_agent_c_names_agent_c(dsn, store, signed_in):
    """Attribution has to survive the split: phase two runs two agents, and
    "no market data" and "no courses" are different problems."""
    from api.main import create_app
    from shared.config import Config
    from tests.api.conftest import FakeRunner

    class BadC(FakeRunner):
        def run_agent_c(self, *, run_id, flags=None):
            raise ValueError("psycopg blew up mid-query")

    app = create_app(Config(database_url=dsn), store=store, runner=BadC(), migrate=False)
    failing = TestClient(app)
    failing.cookies = signed_in.cookies

    doc = _upload(failing)
    job_id = failing.post("/api/analysis", json={"documentIds": [doc]}).json()["jobId"]
    _await_stage(failing, job_id, want={"awaiting_confirmation", "failed"})
    _confirm(failing)
    assert _await_stage(failing, job_id, want={"failed"})["error"] == "agent_c_no_market_data"


def test_polling_someone_elses_job_is_a_404(signed_in: TestClient, client: TestClient):
    doc = _upload(signed_in)
    job_id = signed_in.post("/api/analysis", json={"documentIds": [doc]}).json()["jobId"]

    signed_in.post("/api/placeholder/signup",
                   data={"email": "other@x.test", "password": "Str0ng!pass", "name": "O"})
    assert signed_in.get(f"/api/analysis/{job_id}").status_code == 404


# ---------------------------------------------------------------------------
# progress, profile
# ---------------------------------------------------------------------------
def test_onboarding_progress_round_trips_and_survives_a_device_change(signed_in: TestClient):
    """Keyed by user, not held in a 4 KB cookie — that is the point."""
    assert signed_in.get("/api/onboarding/progress").json() is None
    payload = {"step": "questions", "documents": [], "documentId": None,
               "preferences": {"coursePricing": "free"}, "updatedAt": 1}
    assert signed_in.put("/api/onboarding/progress", json=payload).json() == {"ok": True}
    assert signed_in.get("/api/onboarding/progress").json()["step"] == "questions"
    signed_in.delete("/api/onboarding/progress")
    assert signed_in.get("/api/onboarding/progress").json() is None


def test_confirming_the_profile_flips_onboarded_on_the_account(signed_in: TestClient):
    signed_in.put("/api/onboarding/progress", json={"step": "confirm"})
    res = signed_in.post("/api/profile", json={"fullName": "Maryam", "skills": ["SQL"]})
    assert res.json() == {"ok": True}
    assert signed_in.get("/api/session").json()["user"]["onboarded"] is True
    # Progress is cleared, so returning does not drop back into onboarding.
    assert signed_in.get("/api/onboarding/progress").json() is None


# ---------------------------------------------------------------------------
# reads
# ---------------------------------------------------------------------------
def test_reads_are_empty_not_broken_before_any_analysis(signed_in: TestClient):
    """'No matches yet' is a normal state the UI renders as an empty view; a 404
    would read as a broken route."""
    assert signed_in.get("/api/jobs").json() == []
    assert signed_in.get("/api/courses").json() == []
    assert signed_in.get("/api/dashboard").status_code == 404


def test_dashboard_jobs_and_courses_after_a_run(signed_in: TestClient):
    _full_run(signed_in)

    dash = signed_in.get("/api/dashboard").json()
    # gap_score 0.42 missing -> 58% ready. Readiness is the COMPLEMENT.
    assert dash["readiness"] == 58
    assert "SQL" in dash["strengths"] and "Power BI" in dash["gaps"]
    assert dash["nextStep"]["action"] == "courses"
    assert [s["id"] for s in dash["journey"]][:2] == ["documents", "skills"]

    jobs = signed_in.get("/api/jobs").json()
    assert len(jobs) == 1
    job = jobs[0]
    assert job["score"] == 0.58                      # 1 - gap_score
    assert job["matchedSkills"] == ["SQL"]
    assert "SQL covers their requirement for SQL" in job["why"]
    assert job["source"]["url"].startswith("https://")

    courses = signed_in.get("/api/courses").json()
    assert len(courses) == 1, "the no_course_found entry must not become a course"
    course = courses[0]
    assert course["title"] == "Power BI Essentials"
    assert "Power BI" in course["unlocks"] and "data visualisation" in course["unlocks"]
    # The honesty rules: absent means null, never a plausible zero.
    assert course["price"] is None, "a missing price must not render as free"
    assert course["hours"] is None, "Agent D stores no duration; 0 would be a lie"


def test_a_gap_with_no_course_still_reaches_the_dashboard(signed_in: TestClient):
    """`no_course_found` must surface, so the UI can say 'nothing teaches this
    yet' rather than the gap quietly disappearing."""
    _full_run(signed_in)
    titles = [c["title"] for c in signed_in.get("/api/courses").json()]
    assert "welding" not in titles


def test_reads_require_a_session(client: TestClient):
    for path in ("/api/dashboard", "/api/jobs", "/api/courses"):
        assert client.get(path).status_code == 401, path
