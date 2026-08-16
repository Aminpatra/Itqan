"""Outbound email. Currently one job: the password-reset link.

stdlib `smtplib` and `email.message`, so no new dependency for something this
small.

**The operator is the only one who can be told about a failure.** The
forgot-password endpoint answers 200 whether or not the address has an account —
that is what stops the form becoming a way to enumerate who is registered — which
means a relay that has quietly stopped working looks exactly like a relay that is
working. Nobody complains, because everybody is told "check your email". So every
failure here is logged loudly and counted, and `SEND_FAILURES` is the number
worth alerting on.

**The token exists in this module and nowhere else it could linger.** It is
interpolated into the message and never logged, never returned, and never stored
— `AppStore` only ever sees its hash.
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

# Bumped on every failed send. Read by the health endpoint and worth an alert:
# see the module docstring for why silence is not evidence of success here.
SEND_FAILURES = 0
_FAILURE_LOCK = threading.Lock()


class EmailNotConfigured(RuntimeError):
    """No relay configured. Raised at startup, never per request."""


def is_configured(config: Optional[Config] = None) -> bool:
    return bool((config or Config()).smtp_host)


def send(*, to: str, subject: str, body: str, config: Optional[Config] = None) -> None:
    """Send one plain-text message. Raises on failure; callers decide."""
    config = config or Config()
    if not config.smtp_host:
        raise EmailNotConfigured("ITQAN_SMTP_HOST is not set")

    message = EmailMessage()
    message["From"] = config.smtp_from or config.smtp_user
    message["To"] = to
    message["Subject"] = subject
    message.set_content(body)

    with smtplib.SMTP(config.smtp_host, config.smtp_port,
                      timeout=config.smtp_timeout_s) as server:
        if config.smtp_starttls:
            server.starttls()
        if config.smtp_user:
            server.login(config.smtp_user, config.smtp_password)
        server.send_message(message)


def send_in_background(*, to: str, subject: str, body: str,
                       config: Optional[Config] = None) -> threading.Thread:
    """Hand the send to a thread and return immediately.

    Two reasons, both load-bearing:

    * a relay that takes four seconds must not hold the HTTP request open;
    * **the response time must not depend on whether the account exists.** With
      the send off the request path, the two branches of forgot-password differ
      by one indexed SELECT and one INSERT — microseconds, inside network noise.
      With it on the path they would differ by an entire SMTP conversation, which
      is trivially measurable and would leak exactly what the identical response
      body exists to hide.
    """
    config = config or Config()

    def _run() -> None:
        global SEND_FAILURES
        try:
            send(to=to, subject=subject, body=body, config=config)
        except Exception as exc:                # noqa: BLE001 - never reaches a user
            with _FAILURE_LOCK:
                SEND_FAILURES += 1
            # The address, not the token, and not the body. An operator needs to
            # know delivery is broken; the log is not the place for a credential.
            log.error("password-reset email to %s failed: %s: %s",
                      _redact(to), type(exc).__name__, exc)

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


def reset_link(*, site_url: str, locale: str, token: str) -> str:
    """The URL from BACKEND.md: `https://<site>/<ar|en>/forgot-password/?token=…`

    The token is percent-encoded even though `token_urlsafe` produces nothing
    that needs it — because the day someone changes how tokens are minted, this
    should not become the reason a reset link silently breaks.
    """
    lang = locale if locale in ("ar", "en") else "ar"
    return f"{site_url.rstrip('/')}/{lang}/forgot-password/?token={quote(token)}"


def reset_message(*, link: str, locale: str, minutes: int) -> tuple[str, str]:
    lang = locale if locale in ("ar", "en") else "ar"
    return _SUBJECT[lang], _BODY[lang].format(link=link, minutes=minutes)
