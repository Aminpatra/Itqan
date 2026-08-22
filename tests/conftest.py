import os
import sys
from pathlib import Path

# Tests import `shared` and `agents` as top-level packages.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# A user agent for the suite, set unconditionally.
#
# `Config.user_agent` defaults to a string containing "contact-not-configured",
# and `require_identified_user_agent()` refuses to make live requests with it —
# a deliberate guard, because a scraper that does not identify itself leaves a
# site operator no option but to block it.
#
# The tests never make a real request (every client is a fake), but the guard
# fires before the fake is reached, so six of them need a configured value. On a
# developer machine `.env` supplies one and they pass; on a fresh clone or in CI
# there is no `.env` and they fail. That is a test defect, not a CI defect: a
# suite whose result depends on an untracked file on one person's laptop is not
# telling you what you think it is. It went unnoticed exactly that long.
#
# Set here rather than in a fixture because several test modules construct a
# `Config()` at import time, which happens during collection — before any fixture
# runs. And set unconditionally rather than with `setdefault`, so the suite
# behaves identically everywhere; a developer's real contact address making the
# tests pass differently from CI is the bug being fixed.
#
# `load_dotenv` does not override existing environment variables, so this also
# wins over a local `.env`.
os.environ["ITQAN_USER_AGENT"] = "ItqanTestBot/0.0 (+tests@itqan.invalid)"

# ---------------------------------------------------------------------------
# NO TEST MAY SEND EMAIL. This is enforced twice, because once was not enough.
#
# On 2026-08-20 a full API run delivered roughly 200 messages to a live Brevo
# account. Not a stray call: `signed_in` in tests/api/conftest.py signs a user up,
# signup sends a verification code, and the fixture is function-scoped -- so every
# one of the ~180 tests outside test_email_verification.py and test_password_reset.py
# sent a real message to maryam@itqan.test. A reserved TLD can never resolve, so
# each was a guaranteed hard bounce charged against the sender that real users'
# verification codes depend on.
#
# It was possible because protection was OPT-IN and eleven of twelve files did not
# opt in: the `relay` fixture is autouse but MODULE-scoped, and the reserved-TLD
# guard in api/email.py only fired when ITQAN_ENV was production. Tests run as
# development, against the same .env that holds the production relay credentials.
#
# 1. Blank the relay settings, for the same reason and by the same mechanism as
#    the user agent above: `load_dotenv` does not override an existing variable,
#    so this beats .env on every machine including this one.
# Nobody is exempt from a quota during a test run.
#
# `ITQAN_UNLIMITED_EMAILS` lists the developer accounts that skip the assistant's
# limits, and a developer's own `.env` supplies it. Left alone, a test asserting
# "a new account gets 30 messages" would pass or fail depending on whose laptop
# it ran on -- the same defect this file already fixes for the user agent, and
# the reason that comment says a suite depending on an untracked file is not
# telling you what you think it is.
#
# Any test that WANTS an exemption sets it explicitly with monkeypatch.
os.environ["ITQAN_UNLIMITED_EMAILS"] = ""

for _smtp in ("ITQAN_SMTP_HOST", "ITQAN_SMTP_USER", "ITQAN_SMTP_PASSWORD",
              "ITQAN_SMTP_FROM"):
    os.environ[_smtp] = ""

# 2. Take away the socket itself.
#
# Blanking the environment only makes a send fail QUIETLY, which is how this went
# unnoticed for a day. This makes it fail loudly, at the exact line that tried.
#
# Patched on `smtplib.SMTP.__init__` rather than on the SMTP name, and that is the
# whole reason it works: api.email.send does not call `smtplib.SMTP` -- it calls
# `_RecordingSMTP`, a subclass captured at import time. Replacing the NAME would
# leave that subclass bound to the real class and the socket would still open.
# Patching the inherited constructor catches _RecordingSMTP, SMTP_SSL, and any
# subclass written later, which is what makes this a guard rather than a whack-a-mole.
#
# Applied at import, not as a fixture, so it is already in force during collection
# and there is no ordering question to get wrong. A test that needs to stand in for
# the constructor patches over it locally, as tests/api/test_email_delivery.py does.
import smtplib  # noqa: E402


def _forbid_smtp_socket(self, *args, **kwargs):
    raise AssertionError(
        "a test tried to open an SMTP socket. Nothing in this suite may send "
        "email: the addresses here are reserved TLDs that can never resolve, so "
        "every message is a hard bounce charged against the sender real users "
        "depend on. Use the `relay` fixture in tests/api/conftest.py, which "
        "captures what would have been sent."
    )


smtplib.SMTP.__init__ = _forbid_smtp_socket
