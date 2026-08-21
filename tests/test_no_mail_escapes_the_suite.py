"""The test suite must be incapable of sending email. Pinned here, not assumed.

On 2026-08-20 a single run of `tests/api` delivered roughly 200 messages to a
live Brevo account. Nothing was misconfigured and no test was wrong: `signed_in`
signs a user up, signup mails a verification code, and the fixture is
function-scoped, so ~180 tests each sent a real message to `maryam@itqan.test`.
A reserved TLD can never resolve, so every one was a hard bounce charged against
the sender that real users' verification codes depend on.

Three separate assumptions had to hold for that, and all three were false:

* that a developer machine points at a sink relay -- it points at `.env`, which
  holds the production credentials, because `shared/config` calls `load_dotenv()`
  unconditionally;
* that the reserved-TLD guard covered it -- it was gated on `ITQAN_ENV=production`
  and tests run as development;
* that the `relay` fixture covered it -- it was autouse but MODULE-scoped, so it
  protected two files out of twelve.

**This file lives at the top level on purpose.** Everything under `tests/api/`
skips when no test database is configured, so the guards proven there are silent
on an offline run -- which is most runs. These four assertions cost nothing and
run every single time.
"""

from __future__ import annotations

import smtplib

import pytest

from shared.config import Config


def test_the_smtp_constructor_is_taken_away():
    """The backstop from `tests/conftest.py`. Blanking the environment only makes
    a send fail quietly; this makes it fail at the line that tried."""
    with pytest.raises(AssertionError, match="tried to open an SMTP socket"):
        smtplib.SMTP("smtp-relay.brevo.com", 587, timeout=5)


def test_the_ban_reaches_the_class_send_actually_uses():
    """THE one that matters, and the reason the patch is on `SMTP.__init__`
    rather than on the name `smtplib.SMTP`.

    `api.email.send` does not call `smtplib.SMTP` -- it calls `_RecordingSMTP`, a
    subclass captured at import time. Replacing the name would leave that subclass
    bound to the real class and the socket would still open, which is precisely
    the shape of gap that let this happen in the first place.
    """
    from api import email as email_module

    with pytest.raises(AssertionError, match="tried to open an SMTP socket"):
        email_module._RecordingSMTP("smtp-relay.brevo.com", 587, timeout=5)


def test_the_suite_holds_no_relay_credentials():
    """`.env` on this machine really does hold the production relay. This asserts
    that `tests/conftest.py` wins over it -- `load_dotenv` does not override an
    existing variable, and that fact is load-bearing rather than incidental."""
    config = Config()
    assert config.smtp_host == ""
    assert config.smtp_user == ""
    assert config.smtp_password == ""


def test_sending_is_switched_off_for_this_process():
    """Belt to the socket ban's braces: even given a relay, nothing would go out."""
    from api import email as email_module

    assert email_module.mail_enabled() is False
