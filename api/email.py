"""Outbound email: the password-reset link, and the signup verification code.

stdlib `smtplib` and `email.message`, so no new dependency for something this
small.

**The operator is the only one who can be told about a failure.** The
forgot-password endpoint answers 200 whether or not the address has an account —
that is what stops the form becoming a way to enumerate who is registered — which
means a relay that has quietly stopped working looks exactly like a relay that is
working. Nobody complains, because everybody is told "check your email".

**ACCEPTANCE IS NOT DELIVERY, and this module used to imply otherwise.** It
counted exceptions and called `SEND_FAILURES` the number worth alerting on.
Measured 2026-08-18, during a real outage in which no message reached anyone for
hours: `SEND_FAILURES` was **0**, and had been 0 since the feature shipped. The
relay accepted every message with a 250 and delivered none, so nothing raised and
nothing was counted. The docstring above had predicted exactly that failure and
then instrumented the half of it that does not happen.

A 250 means the relay has QUEUED the message. What these counters measure is the
hand-off — attempted, accepted, refused, failed — and nothing beyond it. The only
authority on whether a person received anything is the provider's own log, or a
delivery webhook (see the note at the end of this module).

**MAIL IS OFF UNLESS THIS IS PRODUCTION.** See `mail_enabled` below. Measured
2026-08-20: one run of the API test suite delivered ~200 messages to reserved-TLD
addresses through the live Brevo account, because `shared/config` calls
`load_dotenv()` unconditionally and therefore a test run reads the SAME relay
credentials the live site does. Nothing here was misconfigured -- there was simply
no switch that said "this box does not send", and every guard that existed had
been written on the assumption that local runs point somewhere harmless. They do
not. A hard bounce is charged against the sender that real users' codes depend
on, so the default had to become silence.

**The token exists in this module and nowhere else it could linger.** It is
interpolated into the message and never logged, never returned, and never stored
— `AppStore` only ever sees its hash. The same rule governs the verification
code, and applies harder: six digits is a credential anyone reading a log could
simply type in.
"""

from __future__ import annotations

import logging
import smtplib
import threading
from email.message import EmailMessage
from typing import Optional
from urllib.parse import quote

from shared.config import Config

log = logging.getLogger("itqan.email")

# The hand-off, counted. Read by `/api/health`.
#
# `SEND_FAILURES` alone was not enough to see an outage — see the docstring. The
# useful signal is the pair: attempts that are not becoming acceptances, or a
# `LAST_SEND_AT` that has stopped moving while people are signing up.
SENDS_ATTEMPTED = 0
SENDS_ACCEPTED = 0
SENDS_REFUSED = 0
# Messages this process deliberately did not send because sending is switched off
# outside production. Kept separate from `refused` (the address cannot receive)
# and from `failed` (the relay would not take it): all three mean "nothing went
# out", and collapsing them would destroy the only distinction an operator cares
# about -- whether the silence is chosen or broken.
SENDS_SUPPRESSED = 0
SEND_FAILURES = 0
LAST_SEND_AT: Optional[str] = None
_COUNTER_LOCK = threading.Lock()
# Kept as an alias: the old name is referenced by existing tests and reads
# naturally where only failures matter.
_FAILURE_LOCK = _COUNTER_LOCK


def counters() -> dict:
    """A snapshot for the health endpoint.

    Deliberately includes `lastSendAt`. A rate is not enough to notice that mail
    has stopped — zero attempts and zero failures is also what a quiet Tuesday
    looks like — but a hand-off timestamp that has not moved while signups are
    arriving is unambiguous.
    """
    with _COUNTER_LOCK:
        return {"attempted": SENDS_ATTEMPTED, "accepted": SENDS_ACCEPTED,
                "refused": SENDS_REFUSED, "suppressed": SENDS_SUPPRESSED,
                "failed": SEND_FAILURES, "lastSendAt": LAST_SEND_AT}


# Reserved TLDs. The DNS root guarantees these never resolve (RFC 2606 / 6761),
# so every message addressed to one is a HARD BOUNCE by construction.
#
# This is not hypothetical tidiness. On 2026-08-18 six probe accounts were
# created on production at `@itqan.test` while testing; the ~7 resulting bounces
# went against a sender that otherwise handles a handful of real messages a day,
# and mail stopped being delivered shortly afterwards. A bounce is a charge
# against sender reputation that every later message pays — including the one a
# real person is waiting for.
_UNDELIVERABLE_TLDS = (".test", ".invalid", ".example", ".localhost")
# Reserved SECOND-level names, matched exactly rather than by suffix: a suffix
# test would also refuse `myexample.com`, which is somebody's real domain.
_UNDELIVERABLE_DOMAINS = frozenset({"example.com", "example.net", "example.org"})


def is_undeliverable(address: str) -> bool:
    """True for an address that provably cannot receive mail.

    Reserved TLDs only, and the narrowness is the point: an MX lookup would be
    cleverer and would fail closed on a real domain having a bad afternoon,
    silently denying a real person their code. These four can never resolve, on
    any day, by standard.
    """
    _, at, domain = (address or "").strip().lower().rpartition("@")
    # `rpartition` puts the WHOLE string in the last slot when the separator is
    # absent, so testing `domain` alone reads "nodomain" as a domain. The
    # separator is what says an address was even well-formed.
    if not at or not domain:
        return True
    return domain in _UNDELIVERABLE_DOMAINS or domain.endswith(_UNDELIVERABLE_TLDS)


class _RecordingSMTP(smtplib.SMTP):
    """An SMTP client that keeps the relay's reply to DATA.

    `send_message` computes that reply and throws it away, which is why nothing
    in the log could say what the relay called the message. Brevo returns a queue
    id in its 250, and that id is the difference between a support ticket that is
    an argument and one that is a lookup.
    """

    last_reply: tuple = (0, b"")

    def data(self, msg):                                    # type: ignore[override]
        code, resp = super().data(msg)
        self.last_reply = (code, resp)
        return code, resp


class EmailNotConfigured(RuntimeError):
    """No relay configured. Raised at startup, never per request."""


def is_configured(config: Optional[Config] = None) -> bool:
    return bool((config or Config()).smtp_host)


class Undeliverable(RuntimeError):
    """The address cannot receive mail, so nothing was sent."""


def _in_production() -> bool:
    """The same switch `api.main.in_production` reads.

    Duplicated rather than imported: `api.main` imports THIS module, so importing
    back would be a cycle. Four lines against an import graph that has to stay
    one-directional is the right trade, and the env var is the contract either
    way.
    """
    import os

    return os.getenv("ITQAN_ENV", "development").lower() == "production"


def mail_enabled() -> bool:
    """Whether this process may put a message on the wire at all.

    **Off unless this is production, or someone has explicitly said otherwise.**

    The reason is measured, not precautionary. `shared/config.py` calls
    `load_dotenv()` unconditionally, so a developer machine and the test suite
    read the SAME relay credentials the live site uses -- and on 2026-08-20 that
    meant a single test run delivered ~200 messages to reserved-TLD addresses
    through the production Brevo account, every one a hard bounce charged against
    the sender that real users' verification codes depend on. Nothing in the code
    was wrong on its own; there was simply no switch that said "this box does not
    send".

    Derived from `ITQAN_ENV` rather than requiring a new variable to be set in
    production, DELIBERATELY: a switch that defaulted off everywhere would mean
    the next API deploy quietly stopped mailing real users unless the VPS `.env`
    were edited first, and this project has already had one outage from a deploy
    landing ahead of its precondition. Nothing on the box has to change.

    `ITQAN_SMTP_ENABLED=1` forces it on for the rare case of testing a real relay
    deliberately -- and it must be typed by a person, on purpose, for that one run.
    """
    import os

    if _in_production():
        return True
    return os.getenv("ITQAN_SMTP_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")


def _suppress(*, to: str, subject: str, body: str, purpose: str) -> None:
    """Record a message that was not sent, and show it to whoever is running this.

    **This is the one place the verification code is written to a log, and the
    exception is deliberate rather than an oversight.** The rule everywhere else
    in this module is that the code never appears in a log line, because six
    digits is a working credential for its ten minutes. Here nothing is being
    delivered, so the log is the ONLY way to walk a signup on a developer machine
    -- and the alternative is reading it out of the database by hand, which is the
    kind of friction that ends with someone switching real sending back on.

    Production can never reach this branch: `mail_enabled()` is true there by
    construction. So the rule stays absolute exactly where it protects anybody.
    """
    global SENDS_SUPPRESSED
    with _COUNTER_LOCK:
        SENDS_SUPPRESSED += 1
    lines = [
        "mail is OFF outside production, so this %s message was NOT sent.",
        "  to:      %s",
        "  subject: %s",
        "%s",
        "  (set ITQAN_SMTP_ENABLED=1 to send for real -- see mail_enabled)",
    ]
    log.info("\n".join(lines), purpose, to, subject, body)


def send(*, to: str, subject: str, body: str, config: Optional[Config] = None,
         purpose: str = "email") -> None:
    """Send one plain-text message. Raises on failure; callers decide.

    Returns nothing on success, but LOGS the hand-off — see the module docstring
    for why a silent success is the thing that hid an outage.
    """
    global SENDS_ATTEMPTED, SENDS_ACCEPTED, SENDS_REFUSED, LAST_SEND_AT
    import time
    from datetime import datetime, timezone

    config = config or Config()

    # Checked FIRST, ahead of both the relay check and the address check, and
    # both orderings carry a reason.
    #
    # Ahead of `smtp_host`, so a machine with no relay configured shows the
    # message instead of raising -- which is the normal state of a dev box now
    # that the test suite blanks those variables.
    #
    # Ahead of `is_undeliverable`, so a local signup at `@itqan.test` -- which is
    # exactly what a developer types -- still prints its code. Refusing it for a
    # bounce that cannot happen when nothing is sent would make the local flow
    # untestable, and an untestable local flow is what gets real sending switched
    # back on.
    if not mail_enabled():
        _suppress(to=to, subject=subject, body=body, purpose=purpose)
        return

    if not config.smtp_host:
        raise EmailNotConfigured("ITQAN_SMTP_HOST is not set")

    # Refused BEFORE the socket opens, so a guaranteed bounce never reaches the
    # relay at all.
    #
    # This used to be gated on `_in_production()`, on the assumption that local
    # runs point at a sink happy to accept `@itqan.test`. They do not: dev, tests
    # and production all read the same `.env`, so the gate held the door open for
    # the ~200 test-suite bounces of 2026-08-20. A reserved TLD cannot resolve on
    # any host, in any environment, on any day -- the environment was never
    # relevant to the fact, only to my assumption about the relay.
    if is_undeliverable(to):
        with _COUNTER_LOCK:
            SENDS_REFUSED += 1
        log.error("REFUSED to send %s email to %s: that domain can never receive "
                  "mail, and a hard bounce is charged against this sender's "
                  "reputation for every later message", purpose, _redact(to))
        raise Undeliverable(f"{_redact(to)} cannot receive mail")

    message = EmailMessage()
    message["From"] = config.smtp_from or config.smtp_user
    message["To"] = to
    message["Subject"] = subject
    message.set_content(body)

    with _COUNTER_LOCK:
        SENDS_ATTEMPTED += 1

    started = time.monotonic()
    with _RecordingSMTP(config.smtp_host, config.smtp_port,
                        timeout=config.smtp_timeout_s) as server:
        if config.smtp_starttls:
            server.starttls()
        if config.smtp_user:
            server.login(config.smtp_user, config.smtp_password)
        server.send_message(message)
        code, reply = server.last_reply

    elapsed_ms = int((time.monotonic() - started) * 1000)
    with _COUNTER_LOCK:
        SENDS_ACCEPTED += 1
        LAST_SEND_AT = datetime.now(timezone.utc).isoformat()

    # The relay's own words for this message. `queued as <id>` is what turns
    # "we sent it" into something the provider can look up — and the redacted
    # address is what keeps this log from becoming a mailing list.
    detail = reply.decode("utf-8", "replace").strip() if isinstance(reply, bytes) else str(reply)
    log.info("%s email handed to relay for %s in %dms: %s %s",
             purpose, _redact(to), elapsed_ms, code, detail)


def send_in_background(*, to: str, subject: str, body: str,
                       config: Optional[Config] = None,
                       purpose: str = "email") -> threading.Thread:
    """Hand the send to a thread and return immediately.

    Two reasons, both load-bearing:

    * a relay that takes four seconds must not hold the HTTP request open;
    * **the response time must not depend on whether the account exists.** With
      the send off the request path, the two branches of forgot-password differ
      by one indexed SELECT and one INSERT — microseconds, inside network noise.
      With it on the path they would differ by an entire SMTP conversation, which
      is trivially measurable and would leak exactly what the identical response
      body exists to hide.

    `purpose` names the flow in the failure log and nothing else. It exists
    because this function was written for one caller and hardcoded
    "password-reset" into its error line — so the first verification failure
    would have been reported, in production, as a broken password reset, sending
    whoever read it to look at the wrong flow.
    """
    config = config or Config()

    def _run() -> None:
        global SEND_FAILURES
        try:
            send(to=to, subject=subject, body=body, config=config, purpose=purpose)
        except Undeliverable:
            # Already counted as `refused` and already logged. NOT a send
            # failure: `failed` is meant to say "the relay would not take it",
            # and folding our own deliberate refusals into it would make a burst
            # of test signups read as a relay outage — blunting the one number
            # an operator is supposed to trust.
            return
        except Exception as exc:                # noqa: BLE001 - never reaches a user
            with _FAILURE_LOCK:
                SEND_FAILURES += 1
            # The address, not the token, and not the body. An operator needs to
            # know delivery is broken; the log is not the place for a credential.
            log.error("%s email to %s failed: %s: %s",
                      purpose, _redact(to), type(exc).__name__, exc)

    thread = threading.Thread(target=_run, daemon=True, name="itqan-email")
    thread.start()
    return thread


def _redact(address: str) -> str:
    """`m***@example.com` — enough to correlate a complaint with a log line,
    not enough to turn the log into a mailing list."""
    name, _, domain = (address or "").partition("@")
    if not domain:
        return "***"
    return f"{name[:1]}***@{domain}"


# ---------------------------------------------------------------------------
# The message itself.
#
# Bilingual, chosen by the account's own locale — the same rule every other
# user-facing string in this system follows. Plain text on purpose: an HTML mail
# would need a second template, and this message is four sentences and a link.
# ---------------------------------------------------------------------------
_SUBJECT = {
    "en": "Reset your Itqan password",
    "ar": "إعادة تعيين كلمة المرور في إتقان",
}

_BODY = {
    "en": (
        "Someone asked to reset the password for your Itqan account.\n\n"
        "Open this link to choose a new one:\n{link}\n\n"
        "The link works once and expires in {minutes} minutes.\n\n"
        "If this was not you, you can ignore this message — nothing has changed, "
        "and your password still works.\n"
    ),
    "ar": (
        "طلب أحدهم إعادة تعيين كلمة المرور لحسابك في إتقان.\n\n"
        "افتح هذا الرابط لاختيار كلمة مرور جديدة:\n{link}\n\n"
        "الرابط يعمل مرة واحدة وتنتهي صلاحيته خلال {minutes} دقيقة.\n\n"
        "إذا لم تكن أنت، يمكنك تجاهل هذه الرسالة — لم يتغير شيء، وكلمة مرورك "
        "الحالية ما زالت تعمل.\n"
    ),
}


_VERIFY_SUBJECT = {
    "en": "Your Itqan verification code",
    "ar": "رمز التحقق الخاص بك في إتقان",
}

# The code sits on its own line, unpunctuated. A trailing full stop is the kind
# of thing that gets selected along with the digits and pasted into the field.
_VERIFY_BODY = {
    "en": (
        "Welcome to Itqan.\n\n"
        "Your verification code is:\n\n"
        "{code}\n\n"
        "Enter it on the page you were taken to after signing up. The code "
        "expires in {minutes} minutes, and you can ask for a new one at any "
        "time.\n\n"
        "If you did not create an Itqan account, you can ignore this message — "
        "nothing has been set up, and the address will not be used again.\n"
    ),
    "ar": (
        "مرحبًا بك في إتقان.\n\n"
        "رمز التحقق الخاص بك هو:\n\n"
        "{code}\n\n"
        "أدخله في الصفحة التي وصلت إليها بعد إنشاء الحساب. تنتهي صلاحية الرمز "
        "خلال {minutes} دقيقة، ويمكنك طلب رمز جديد في أي وقت.\n\n"
        "إذا لم تنشئ حسابًا في إتقان، يمكنك تجاهل هذه الرسالة — لم يتم إعداد أي "
        "شيء، ولن يُستخدم هذا العنوان مرة أخرى.\n"
    ),
}


def verification_message(*, code: str, locale: str, minutes: int) -> tuple[str, str]:
    """Subject and body for the signup verification code.

    The code is formatted into the body here and nowhere else — it is never
    logged, never returned to a caller, and never stored: `AppStore` sees only
    its sha256.
    """
    lang = locale if locale in ("ar", "en") else "en"
    return _VERIFY_SUBJECT[lang], _VERIFY_BODY[lang].format(code=code, minutes=minutes)


def reset_link(*, site_url: str, locale: str, token: str) -> str:
    """The URL from BACKEND.md: `https://<site>/<ar|en>/forgot-password/?token=…`

    The token is percent-encoded even though `token_urlsafe` produces nothing
    that needs it — because the day someone changes how tokens are minted, this
    should not become the reason a reset link silently breaks.
    """
    lang = locale if locale in ("ar", "en") else "en"
    return f"{site_url.rstrip('/')}/{lang}/forgot-password/?token={quote(token)}"


def reset_message(*, link: str, locale: str, minutes: int) -> tuple[str, str]:
    lang = locale if locale in ("ar", "en") else "en"
    return _SUBJECT[lang], _BODY[lang].format(link=link, minutes=minutes)


# ---------------------------------------------------------------------------
# What is still missing, named so it is a decision rather than an oversight.
#
# Nothing in this file can tell whether a message was DELIVERED. The counters
# above stop at the relay's 250, and 2026-08-18 is the proof that a 250 and a
# delivery are different events: every message was accepted and none arrived.
#
# The answer is the provider's delivery webhook — Brevo will POST delivered,
# bounced, blocked and complained events. That needs a public endpoint,
# signature verification and somewhere to put the events, which is a real piece
# of work and deserves designing rather than being bolted on during an incident.
# Until it exists, the provider's own dashboard is the only authority on whether
# anyone received anything, and this module should not be read as claiming
# otherwise.
# ---------------------------------------------------------------------------
