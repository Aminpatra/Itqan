"""Password recovery: what it must never reveal, and what a reset must actually do.

Two tests carry this file.

`test_a_known_and_an_unknown_address_are_indistinguishable` is the security
property the whole endpoint is shaped around. If a missing address answers
differently, the form becomes a way to find out who is registered — and the cost
of getting it right is that a broken relay is invisible, which is why the boot
check in `test_routes.py` is fatal.

`test_a_reset_evicts_the_old_session` is the one that makes the feature worth
having. Session tokens used to be permanent, so resetting a password left a
stolen cookie working — the feature failing at precisely the case it exists for.
"""

from __future__ import annotations

import hashlib

import pytest

EMAIL = "maryam@itqan.test"
GOOD = "N3w!passw0rd"



def _forgot(client, email=EMAIL):
    return client.post("/api/auth/forgot-password", data={"email": email})


def _reset_mails(sent: list[dict]) -> list[dict]:
    """Only the reset messages.

    Two kinds of mail now leave this system, and `signed_in` sends the other one:
    creating an account mails a verification code. So `sent[-1]` stopped meaning
    "the reset email" — and worse, it stopped meaning it INTERMITTENTLY, because
    the send is on a background thread and its position in the list is a race.
    Selecting by content is what makes these assertions about the reset flow
    again rather than about thread scheduling.
    """
    return [m for m in sent if "token=" in m.get("body", "")]


def _reset_mail(sent: list[dict]) -> dict:
    mails = _reset_mails(sent)
    assert mails, "no password-reset email was sent"
    return mails[-1]


def _token_from(sent: list[dict]) -> str:
    """Pull the token out of the delivered link, the way a person would."""
    body = _reset_mail(sent)["body"]
    return body.split("token=")[1].split()[0].strip()


def _wait_for_mail(sent: list[dict], count: int = 1) -> None:
    """Wait for `count` RESET messages. The send is on a background thread.

    Counting reset mails rather than all mail, for the same reason `_reset_mail`
    selects by content: `signed_in` now also sends a verification code, so a bare
    `len(sent) >= 2` could be satisfied by one verification plus one reset — and
    the caller would read the FIRST token twice, believing it had waited for the
    second. That failure would have appeared as a flake in a test about tokens.
    """
    import time
    deadline = time.time() + 3.0
    while time.time() < deadline and len(_reset_mails(sent)) < count:
        time.sleep(0.02)


# ---------------------------------------------------------------------------
# what must not be revealed
# ---------------------------------------------------------------------------
def test_a_known_and_an_unknown_address_are_indistinguishable(signed_in, client, relay):
    """THE security property.

    Same status, same body. A difference here — including a difference in
    response TIME, which is why the SMTP conversation is on a background thread —
    turns this form into an account-enumeration oracle.
    """
    known = _forgot(client, EMAIL)
    unknown = _forgot(client, "nobody@nowhere.test")

    assert known.status_code == unknown.status_code == 200
    assert known.json() == unknown.json() == {"ok": True}


def test_no_mail_is_sent_to_an_address_with_no_account(client, relay):
    _forgot(client, "nobody@nowhere.test")

    import time
    time.sleep(0.2)
    assert relay == []


def test_being_rate_limited_looks_exactly_like_success(signed_in, client, relay):
    """Revealing the limit would itself be a signal — and would tell an attacker
    when to back off rather than making them fail invisibly."""
    for _ in range(3):
        assert _forgot(client).status_code == 200
    _wait_for_mail(relay, 3)

    fourth = _forgot(client)

    assert fourth.status_code == 200 and fourth.json() == {"ok": True}
    import time
    time.sleep(0.2)
    assert len(_reset_mails(relay)) == 3, "a 4th email was sent past the hourly limit"


def test_the_raw_token_is_never_stored(signed_in, client, store, relay):
    """A database leak must not hand over working reset links."""
    _forgot(client)
    _wait_for_mail(relay)
    token = _token_from(relay)

    with store.connect().cursor() as cur:
        cur.execute("SELECT token_hash FROM app_password_resets")
        stored = [r["token_hash"] for r in cur.fetchall()]

    assert token not in stored
    assert hashlib.sha256(token.encode()).hexdigest() in stored


def test_the_throttle_table_does_not_record_who_tried(signed_in, client, store, relay):
    """It would otherwise become the list the identical responses exist to
    protect, handed over by the logging instead."""
    _forgot(client)

    with store.connect().cursor() as cur:
        cur.execute("SELECT subject FROM app_reset_throttle")
        subjects = [r["subject"] for r in cur.fetchall()]

    assert subjects, "nothing was throttled"
    assert EMAIL not in subjects
    assert hashlib.sha256(EMAIL.encode()).hexdigest() in subjects


# ---------------------------------------------------------------------------
# the reset itself
# ---------------------------------------------------------------------------
def test_the_link_lets_someone_set_a_new_password_and_log_in(signed_in, client, relay):
    _forgot(client)
    _wait_for_mail(relay)
    token = _token_from(relay)

    res = client.post("/api/auth/reset-password", data={"token": token, "password": GOOD})
    assert res.status_code == 200

    client.post("/api/logout")
    assert client.post("/api/auth/login",
                       data={"email": EMAIL, "password": GOOD}).status_code == 200


def test_the_old_password_stops_working(signed_in, client, relay):
    _forgot(client)
    _wait_for_mail(relay)
    client.post("/api/auth/reset-password",
                data={"token": _token_from(relay), "password": GOOD})

    client.post("/api/logout")
    assert client.post("/api/auth/login",
                       data={"email": EMAIL, "password": "Str0ng!pass"}).status_code == 401


def test_a_token_works_exactly_once(signed_in, client, relay):
    _forgot(client)
    _wait_for_mail(relay)
    token = _token_from(relay)

    assert client.post("/api/auth/reset-password",
                       data={"token": token, "password": GOOD}).status_code == 200
    second = client.post("/api/auth/reset-password",
                         data={"token": token, "password": "An0ther!pass"})

    assert second.status_code == 410


def test_using_one_token_kills_the_others(signed_in, client, relay):
    """Two "forgot password" clicks must not leave a spare key to an account its
    owner believes they have just secured."""
    _forgot(client)
    _wait_for_mail(relay, 1)
    first = _token_from(relay)
    _forgot(client)
    _wait_for_mail(relay, 2)
    second = _token_from(relay)
    assert first != second

    assert client.post("/api/auth/reset-password",
                       data={"token": second, "password": GOOD}).status_code == 200

    assert client.post("/api/auth/reset-password",
                       data={"token": first, "password": "An0ther!pass"}).status_code == 410


@pytest.mark.parametrize("token", ["", "not-a-token", "x" * 43])
def test_a_bad_token_is_410(client, token):
    res = client.post("/api/auth/reset-password", data={"token": token, "password": GOOD})

    assert res.status_code == 410


def test_an_expired_token_is_410(signed_in, client, store, relay):
    _forgot(client)
    _wait_for_mail(relay)
    token = _token_from(relay)
    with store.connect().cursor() as cur:
        cur.execute("UPDATE app_password_resets SET expires_at = now() - interval '1 minute'")

    res = client.post("/api/auth/reset-password", data={"token": token, "password": GOOD})

    assert res.status_code == 410


def test_a_weak_password_is_422_not_410(signed_in, client, relay):
    """The status-code distinction the front end forces.

    It renders 400 AND 410 as "this link has expired". Answering a short password
    with either would send someone to fetch a new link that fails identically —
    a loop they cannot escape. 422 lands on the generic error, which is true.
    """
    _forgot(client)
    _wait_for_mail(relay)

    res = client.post("/api/auth/reset-password",
                      data={"token": _token_from(relay), "password": "short"})

    assert res.status_code == 422


def test_a_rejected_password_does_not_burn_the_token(signed_in, client, relay):
    """Otherwise a typo costs a second email and a second wait."""
    _forgot(client)
    _wait_for_mail(relay)
    token = _token_from(relay)

    client.post("/api/auth/reset-password", data={"token": token, "password": "short"})

    assert client.post("/api/auth/reset-password",
                       data={"token": token, "password": GOOD}).status_code == 200


def test_a_reset_does_not_sign_the_user_in(client, signed_in, relay):
    """The front end navigates to the login page, and using the new password is
    what proves it is the one that works."""
    _forgot(client)
    _wait_for_mail(relay)
    client.post("/api/logout")

    client.post("/api/auth/reset-password",
                data={"token": _token_from(relay), "password": GOOD})

    assert client.get("/api/session").status_code == 401


# ---------------------------------------------------------------------------
# session eviction — the reason this feature is worth having
# ---------------------------------------------------------------------------
def test_a_reset_evicts_the_old_session(signed_in, client, relay):
    """THE test that makes recovery mean something.

    Session tokens were `user_id + HMAC(secret, user_id)`: deterministic and
    permanent. A cookie captured once worked forever, and changing the password
    did nothing to it — so the feature failed at the exact case it exists for.
    """
    assert signed_in.get("/api/session").status_code == 200
    stolen = dict(signed_in.cookies)

    _forgot(client)
    _wait_for_mail(relay)
    client.post("/api/auth/reset-password",
                data={"token": _token_from(relay), "password": GOOD})

    # The very same cookie, replayed.
    replayed = client.get("/api/session", cookies=stolen)
    assert replayed.status_code == 401, "a session captured before the reset still works"


def test_a_fresh_login_after_the_reset_works(signed_in, client, relay):
    """The other half: eviction must not lock the real owner out too."""
    _forgot(client)
    _wait_for_mail(relay)
    client.post("/api/auth/reset-password",
                data={"token": _token_from(relay), "password": GOOD})
    client.post("/api/logout")

    client.post("/api/auth/login", data={"email": EMAIL, "password": GOOD})

    assert client.get("/api/session").status_code == 200


# ---------------------------------------------------------------------------
# the relay
# ---------------------------------------------------------------------------
def test_the_link_matches_the_contract_and_carries_the_locale(signed_in, client, relay):
    _forgot(client)
    _wait_for_mail(relay)

    body = _reset_mail(relay)["body"]
    assert "/forgot-password/?token=" in body
    assert "/ar/" in body or "/en/" in body


# ---------------------------------------------------------------------------
# language — English by default, and the language they were actually reading
# ---------------------------------------------------------------------------
def test_a_new_account_defaults_to_english(client, store, relay):
    """The bug behind an Arabic reset email: signup passed NO locale, so every
    account — including one created entirely on the English site — was stored
    'ar' from the column default."""
    client.post("/api/auth/signup", data={"email": "fresh@itqan.test",
                                          "password": "Str0ng!pass", "name": "Fresh"})

    assert store.user_by_email("fresh@itqan.test")["locale"] == "en"


@pytest.mark.parametrize("cookie, expected", [("ar", "ar"), ("en", "en"),
                                              ("fr", "en"), ("", "en")])
def test_signup_stores_the_language_being_browsed(client, store, cookie, expected):
    """The site's language toggle already writes `itqan_locale`; nothing new was
    needed to know this, it was simply never read."""
    if cookie:
        client.cookies.set("itqan_locale", cookie)
    client.post("/api/auth/signup", data={"email": f"u{cookie or 'none'}@itqan.test",
                                          "password": "Str0ng!pass", "name": "U"})

    assert store.user_by_email(f"u{cookie or 'none'}@itqan.test")["locale"] == expected


def test_the_email_is_english_for_an_english_account(client, store, relay):
    """Asserted on the DECODED body: a non-ASCII message arrives base64-encoded,
    so a naive substring check would pass on anything at all."""
    client.cookies.set("itqan_locale", "en")
    client.post("/api/auth/signup", data={"email": "en@itqan.test",
                                          "password": "Str0ng!pass", "name": "E"})
    client.post("/api/auth/forgot-password", data={"email": "en@itqan.test"})
    _wait_for_mail(relay)

    assert "Reset your Itqan password" == _reset_mail(relay)["subject"]
    assert "Open this link to choose a new one" in _reset_mail(relay)["body"]
    assert "/en/forgot-password/" in _reset_mail(relay)["body"]


def test_an_arabic_account_still_gets_arabic(client, store, relay):
    """The account's stated preference wins. Silently switching someone who chose
    Arabic into English would be the same class of mistake in the other
    direction."""
    client.cookies.set("itqan_locale", "ar")
    client.post("/api/auth/signup", data={"email": "ar@itqan.test",
                                          "password": "Str0ng!pass", "name": "A"})
    client.post("/api/auth/forgot-password", data={"email": "ar@itqan.test"})
    _wait_for_mail(relay)

    assert "إتقان" in _reset_mail(relay)["subject"]
    assert "/ar/forgot-password/" in _reset_mail(relay)["body"]


# ---------------------------------------------------------------------------
# expiry
# ---------------------------------------------------------------------------
def test_the_link_lasts_ten_minutes(signed_in, client, store, relay):
    """The number in the email comes from the same constant as the database
    expiry, so the sentence cannot drift from the behaviour."""
    from shared.config import Config
    assert Config().reset_token_minutes == 10

    _forgot(client)
    _wait_for_mail(relay)
    assert "10 minutes" in _reset_mail(relay)["body"]

    with store.connect().cursor() as cur:
        cur.execute("SELECT expires_at - created_at AS life FROM app_password_resets")
        life = cur.fetchone()["life"]
    assert 9 * 60 <= life.total_seconds() <= 11 * 60


def test_a_token_is_dead_at_eleven_minutes(signed_in, client, store, relay):
    _forgot(client)
    _wait_for_mail(relay)
    token = _token_from(relay)
    with store.connect().cursor() as cur:
        cur.execute("UPDATE app_password_resets "
                    "SET created_at = now() - interval '11 minutes', "
                    "    expires_at = now() - interval '1 minute'")

    assert client.post("/api/auth/reset-password",
                       data={"token": token, "password": GOOD}).status_code == 410


def test_a_dead_relay_still_answers_200_and_logs_it(signed_in, client, monkeypatch, caplog):
    """The one failure the user cannot be told about, so the operator must be."""
    import api.email as email_module

    def explode(**_kw):
        raise OSError("connection refused")

    monkeypatch.setattr("api.email.send", explode)
    before = email_module.SEND_FAILURES

    with caplog.at_level("ERROR"):
        res = _forgot(client)
        import time
        time.sleep(0.3)

    assert res.status_code == 200
    assert email_module.SEND_FAILURES == before + 1
    assert "failed" in caplog.text


def test_the_log_does_not_contain_the_token_or_the_full_address(
        signed_in, client, monkeypatch, caplog):
    """A log is not the place for a credential, nor a mailing list."""
    monkeypatch.setattr("api.email.send",
                        lambda **_kw: (_ for _ in ()).throw(OSError("nope")))

    with caplog.at_level("ERROR"):
        _forgot(client)
        import time
        time.sleep(0.3)

    assert EMAIL not in caplog.text
    assert "token=" not in caplog.text
