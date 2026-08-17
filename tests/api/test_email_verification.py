"""Email verification at signup: what the gate refuses, and what the code costs.

Two tests carry this file.

`test_the_sixth_attempt_fails_even_with_the_right_code` is the security property
the design rests on. Six digits is a million combinations, and neither the length
nor the sha256 the code is stored under is what makes that safe — an attacker
holding the table enumerates every preimage instantly. The attempt limit is the
whole defence, so it is the thing to pin.

`test_onboarding_refuses_an_unverified_account` is the one that makes the feature
mean anything. The site redirects and the app's route guards redirect, but both
are navigation; a request made with `curl` sees neither. If the routes do not
refuse, verification is a suggestion.
"""

from __future__ import annotations

import hashlib
import time

import pytest

EMAIL = "maryam@itqan.test"


@pytest.fixture(autouse=True)
def relay(monkeypatch):
    """Capture what would have been sent instead of opening a socket.

    Patched at `send`, so the threading and failure handling around it stay under
    test — the same seam `test_password_reset.py` uses.
    """
    sent: list[dict] = []
    monkeypatch.setattr("api.email.send", lambda **kw: sent.append(kw))
    return sent


def _wait_for_mail(sent: list[dict], count: int = 1) -> None:
    deadline = time.time() + 3.0
    while time.time() < deadline and len(sent) < count:
        time.sleep(0.02)


def _code_from(sent: list[dict]) -> str:
    """Read the code out of the delivered message, the way a person would."""
    _wait_for_mail(sent, len(sent) or 1)
    assert sent, "no email was sent"
    for line in sent[-1]["body"].splitlines():
        token = line.strip()
        if token.isdigit() and len(token) == 6:
            return token
    raise AssertionError(f"no 6-digit code in the message: {sent[-1]['body']!r}")


def _verify(client, code):
    return client.post("/api/auth/verify-email", data={"code": code})


# ---------------------------------------------------------------------------
# the happy path
# ---------------------------------------------------------------------------
def test_signing_up_sends_a_code(unverified, relay):
    _wait_for_mail(relay)
    assert len(relay) == 1
    assert relay[0]["to"] == EMAIL
    assert len(_code_from(relay)) == 6


def test_the_code_verifies_the_account(unverified, relay):
    assert _verify(unverified, _code_from(relay)).status_code == 200
    assert unverified.get("/api/session").json()["user"]["emailVerified"] is True


def test_an_account_starts_unverified(unverified):
    assert unverified.get("/api/session").json()["user"]["emailVerified"] is False


def test_a_second_submit_of_a_spent_code_still_reports_success(unverified, relay):
    """The tab was left open, or the button was double-clicked. The state they
    wanted is the state that holds, so this is 200 — not an error about a code
    that has already done its job."""
    code = _code_from(relay)
    assert _verify(unverified, code).status_code == 200
    again = _verify(unverified, code)
    assert again.status_code == 200 and again.json()["alreadyVerified"] is True


# ---------------------------------------------------------------------------
# the attempt limit — the actual control
# ---------------------------------------------------------------------------
def test_a_wrong_code_is_422_and_counts_down(unverified, relay):
    _code_from(relay)
    res = _verify(unverified, "000000")
    assert res.status_code == 422
    # Saying how many are left costs nothing — it is their own account and their
    # own code — and withholding it makes the fifth failure look like the first,
    # right up until the code silently dies.
    assert res.json()["attemptsRemaining"] == 4


def test_the_sixth_attempt_fails_even_with_the_right_code(unverified, relay):
    """THE test. Five wrong answers kill the code, so guessing is capped at 5 in
    1,000,000 — which is what makes a six-digit secret defensible at all."""
    code = _code_from(relay)
    wrong = "000000" if code != "000000" else "111111"

    for _ in range(5):
        assert _verify(unverified, wrong).status_code in (422, 410)

    dead = _verify(unverified, code)
    assert dead.status_code == 410, "a spent code still accepted the right answer"
    assert unverified.get("/api/session").json()["user"]["emailVerified"] is False


def test_running_out_of_attempts_reads_as_over_not_as_wrong(unverified, relay):
    """The last failure answers 410, not 422. 422 invites a sixth try that cannot
    succeed; 410 is the panel that offers a new code."""
    _code_from(relay)
    for _ in range(4):
        _verify(unverified, "000000")
    assert _verify(unverified, "000000").status_code == 410


def test_attempts_are_counted_under_concurrency(unverified, relay, store):
    """A limit that is checked and then incremented is not a limit.

    Ten simultaneous wrong guesses against a cap of five must leave `attempts` at
    exactly five: the guarded UPDATE refuses the rest rather than letting them
    all read the same pre-increment value and proceed.
    """
    from concurrent.futures import ThreadPoolExecutor

    _code_from(relay)
    with ThreadPoolExecutor(max_workers=10) as pool:
        list(pool.map(lambda _: _verify(unverified, "000000"), range(10)))

    row = store.user_by_email(EMAIL)
    attempts = store._one(
        "SELECT attempts FROM app_email_verifications WHERE user_id = %s",
        (row["user_id"],))
    assert attempts["attempts"] == 5, f"attempt cap leaked under load: {attempts}"


# ---------------------------------------------------------------------------
# expiry and resend
# ---------------------------------------------------------------------------
def test_an_expired_code_is_410(unverified, relay, store):
    code = _code_from(relay)
    row = store.user_by_email(EMAIL)
    store._exec("UPDATE app_email_verifications SET expires_at = now() - interval '1 minute' "
                " WHERE user_id = %s", (row["user_id"],))
    assert _verify(unverified, code).status_code == 410


def test_a_resend_invalidates_the_previous_code(unverified, relay):
    """One outstanding code per account. The message that just arrived is the one
    that works, and its predecessor stops working the moment it is replaced —
    otherwise every resend adds a live credential."""
    first = _code_from(relay)
    unverified.post("/api/auth/resend-verification")
    _wait_for_mail(relay, 2)
    second = _code_from(relay)

    assert first != second
    assert _verify(unverified, first).status_code == 422, "an old code still worked"
    assert _verify(unverified, second).status_code == 200


def test_a_resend_is_rate_limited_but_still_answers_200(unverified, relay):
    """Throttled and sent look identical, so hammering the button reveals
    nothing and shows the same reassurance either way."""
    for _ in range(10):
        assert unverified.post("/api/auth/resend-verification").status_code == 200
    _wait_for_mail(relay, 6)
    # 1 from signup + the per-user hourly limit of 5.
    assert len(relay) <= 6, f"the resend limit did not hold: {len(relay)} messages"


def test_a_verified_account_resending_sends_nothing(unverified, relay):
    _verify(unverified, _code_from(relay))
    before = len(relay)
    assert unverified.post("/api/auth/resend-verification").status_code == 200
    time.sleep(0.2)
    assert len(relay) == before


# ---------------------------------------------------------------------------
# the countdown the page shows
# ---------------------------------------------------------------------------
def _status(client):
    return client.get("/api/auth/verification").json()


def test_a_fresh_code_reports_close_to_ten_minutes(unverified, relay):
    _code_from(relay)
    state = _status(unverified)
    assert state["verified"] is False
    assert 570 <= state["secondsRemaining"] <= 600, state
    assert state["attemptsRemaining"] == 5


def test_the_countdown_comes_from_the_row_and_not_from_a_constant(unverified, relay, store):
    """THE test. A timer started when the page loads restarts at ten minutes
    after every reload, while the code was issued at signup — so it would show
    minutes remaining on a code the server has already killed.

    Ageing the row must move this number. If it does not, the endpoint is
    returning a constant and the page is lying with extra steps.
    """
    _code_from(relay)
    row = store.user_by_email(EMAIL)
    store._exec("UPDATE app_email_verifications "
                "   SET expires_at = expires_at - interval '7 minutes' "
                " WHERE user_id = %s", (row["user_id"],))

    assert 120 <= _status(unverified)["secondsRemaining"] <= 180


def test_an_expired_code_reports_zero_and_never_a_negative(unverified, relay, store):
    """Past the expiry the raw subtraction goes negative, and a negative count
    reaches the page as a timer reading `-1:-13`."""
    _code_from(relay)
    row = store.user_by_email(EMAIL)
    store._exec("UPDATE app_email_verifications "
                "   SET expires_at = now() - interval '5 minutes' "
                " WHERE user_id = %s", (row["user_id"],))

    assert _status(unverified)["secondsRemaining"] == 0


def test_a_spent_code_reports_zero(unverified, relay):
    _verify(unverified, _code_from(relay))
    state = _status(unverified)
    assert state["verified"] is True and state["secondsRemaining"] == 0


def test_a_resend_restarts_the_countdown(unverified, relay, store):
    _code_from(relay)
    row = store.user_by_email(EMAIL)
    store._exec("UPDATE app_email_verifications "
                "   SET expires_at = expires_at - interval '8 minutes' "
                " WHERE user_id = %s", (row["user_id"],))
    assert _status(unverified)["secondsRemaining"] < 180

    unverified.post("/api/auth/resend-verification")
    assert _status(unverified)["secondsRemaining"] > 500, "a new code kept the old window"


def test_attempts_remaining_survives_a_reload(unverified, relay):
    """The browser forgets the wrong-code message; the server does not."""
    _code_from(relay)
    _verify(unverified, "000000")
    assert _status(unverified)["attemptsRemaining"] == 4


def test_the_countdown_needs_a_session(client):
    assert client.get("/api/auth/verification").status_code == 401


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method,path,kwargs", [
    ("post", "/api/documents", {"files": {"file": ("cv.txt", b"hello", "text/plain")},
                                "data": {"kind": "cv"}}),
    ("post", "/api/analysis", {"json": {"documentIds": ["doc_1"]}}),
    ("put", "/api/onboarding/progress", {"json": {"step": 1}}),
    ("post", "/api/profile", {"json": {"fullName": "Maryam"}}),
    ("put", "/api/profile", {"json": {"fullName": "Maryam"}}),
])
def test_onboarding_refuses_an_unverified_account(unverified, method, path, kwargs):
    """Server-side, on the routes themselves. The redirects elsewhere are
    navigation; this is the control, and it is what a `curl` sees."""
    res = getattr(unverified, method)(path, **kwargs)
    assert res.status_code == 403, f"{method.upper()} {path} let an unverified account through"
    assert res.json()["error"] == "email_unverified"
    # 403 and not 401: they ARE signed in, and 401 would bounce the app back to a
    # login page that changes nothing — a loop.


def test_the_same_routes_work_once_verified(signed_in):
    res = signed_in.put("/api/onboarding/progress", json={"step": 1})
    assert res.status_code == 200


def test_the_refusal_says_where_to_go(unverified):
    res = unverified.put("/api/onboarding/progress", json={"step": 1})
    assert res.json()["verifyUrl"].endswith("/verify-email/")


def test_reads_are_not_gated(unverified):
    """An unverified account has no completed run, so these already answer 404.
    Gating them too would add a failure mode without removing a capability."""
    assert unverified.get("/api/dashboard").status_code in (404, 200)


def test_handoff_sends_an_unverified_user_to_the_code_page(unverified):
    res = unverified.get("/api/handoff", follow_redirects=False)
    assert res.status_code == 302
    assert res.headers["location"].endswith("/verify-email/")


def test_handoff_sends_a_verified_user_to_the_app(signed_in):
    res = signed_in.get("/api/handoff", follow_redirects=False)
    assert res.headers["location"] == "/app/"


# ---------------------------------------------------------------------------
# the code must not linger anywhere it could be read
# ---------------------------------------------------------------------------
def test_the_raw_code_is_never_stored(unverified, relay, store):
    code = _code_from(relay)
    row = store.user_by_email(EMAIL)
    stored = store._one(
        "SELECT code_hash FROM app_email_verifications WHERE user_id = %s",
        (row["user_id"],))
    assert stored["code_hash"] == hashlib.sha256(code.encode()).hexdigest()
    assert code not in stored["code_hash"]


def test_a_dead_relay_still_lets_signup_succeed_and_logs_it(client, monkeypatch, caplog):
    """Signup must not fail because the relay is down — the account exists and a
    resend can follow. The operator is the one told, through the log."""
    def _boom(**_kw):
        raise OSError("relay unreachable")

    monkeypatch.setattr("api.email.send", _boom)
    with caplog.at_level("ERROR"):
        res = client.post("/api/placeholder/signup",
                          data={"email": "someone@itqan.test", "password": "Str0ng!pass",
                                "name": "Someone"})
        assert res.status_code == 200
        time.sleep(0.3)

    text = caplog.text
    assert "verification" in text, "a verification failure was not identified as one"
    assert "password-reset" not in text, "mislabelled as a password reset"
    assert "someone@itqan.test" not in text, "the full address reached the log"
