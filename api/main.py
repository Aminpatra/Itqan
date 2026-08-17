"""The FastAPI app joining the pipeline to the web app.

Routes, status codes and response shapes match `Onboarding/dev/site-plugin.ts`
exactly — that stub is the contract the UI is actually developed against, and
`src/api/http.ts` is what calls it. Where the handoff PDF disagrees (snake_case,
`/api/auth/*`, synchronous analysis, `{en, ar}` objects) the running code wins;
the PDF describes a later target and four of the files it names do not exist yet.

An orchestrator, not an agent: it lives beside `agents/` and drives the agents
through their public CLIs, exactly as `agents/pipeline.py` does. It never reaches
into an agent's internals, and it reads pipeline tables only through the published
surfaces in `shared/`, so the eligibility predicates stay single-sourced.
"""

from __future__ import annotations

import os
import re
import secrets
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, File, Form, Request, Response, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse

from shared.config import Config

from . import assistant as assistant_module
from . import avatars
from . import email as email_module
from . import jobs as jobs_module
from . import mapping
from .db import AppStore, apply_migrations, hash_password, verify_password


def _locale_of(request: Request) -> str:
    """The language this request was made in, or English.

    Reads the `itqan_locale` cookie the site's language toggle sets. English is
    the fallback rather than Arabic (2026-08-16): a visitor who has never touched
    the toggle has expressed no preference, and defaulting them into a language
    they may not read is the worse of the two guesses.
    """
    value = (request.cookies.get(LOCALE_COOKIE) or "").strip().lower()
    return value if value in ("ar", "en") else "en"


def _sha256(value: str) -> str:
    """What gets stored instead of the thing itself.

    Used for reset tokens and for throttle keys. A fast hash is correct for both:
    a token is 256 bits of entropy rather than a guessable secret, and an email
    address is hashed to keep the throttle table from becoming a list of who
    tried — not to withstand an offline attack on the address space.
    """
    import hashlib
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

SESSION_COOKIE = "itqan_session"          # names fixed by dev/site-plugin.ts
LOCALE_COOKIE = "itqan_locale"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024       # the UI states 10 MB

# The UI's DocumentKind. `cv` is required — Agent A requires --cv and treats
# --transcript as optional, which is why REQUIRED_KIND was corrected to 'cv'.
DOCUMENT_KINDS = {"cv", "transcript", "certificate", "certification",
                  "recommendation", "other"}
REQUIRED_KIND = "cv"

# Same rule as itqan-website/src/scripts/form.ts and the dev plugin. Three places
# must agree; if this changes, change all three or dev accepts what prod rejects.
_PW_RULES = (
    (lambda p: len(p) >= 8, "at least 8 characters"),
    (lambda p: re.search(r"[a-z]", p), "a lowercase letter"),
    (lambda p: re.search(r"[A-Z]", p), "an uppercase letter"),
    (lambda p: re.search(r"\d", p), "a digit"),
    (lambda p: re.search(r"[^A-Za-z0-9]", p), "a symbol"),
)


def password_problems(password: str) -> list[str]:
    return [why for ok, why in _PW_RULES if not ok(password)]


def _new_code(digits: int) -> str:
    """A zero-padded numeric code from the system CSPRNG.

    `secrets.randbelow`, not `random`: the module that seeds itself from the
    clock is fine for shuffling and is a credential generator that can be
    predicted. Zero-padded because a code is a fixed-width string to the person
    typing it — dropping a leading zero would silently make one code in ten
    shorter than the field expects.
    """
    return str(secrets.randbelow(10 ** digits)).zfill(digits)


def verify_path(locale: str) -> str:
    """Where an unverified person is sent. Same origin, so a relative path.

    Site-owned, like the forgot-password page: the marketing site holds the auth
    screens and the app holds the product.
    """
    lang = locale if locale in ("ar", "en") else "en"
    return f"/{lang}/verify-email/"


def create_app(config: Optional[Config] = None, *,
               store: Optional[AppStore] = None,
               runner: Optional[jobs_module.PipelineRunner] = None,
               assistant_llm: Any = None,
               migrate: bool = True) -> FastAPI:
    """Dependencies are injected, like every agent's `Deps` — so the tests drive
    the real routes with a fake pipeline and never touch OpenAI or OCR."""
    config = config or Config()
    # Before anything else, and before a single request can be served.
    assert_deployable()
    app = FastAPI(title="Itqan API", docs_url="/api/docs", openapi_url="/api/openapi.json")

    if migrate:
        apply_migrations(config.require_database_url())
    app.state.config = config
    app.state.store = store or AppStore.from_config(config)
    app.state.runner = runner or jobs_module.PipelineRunner(config)
    app.state.upload_dir = Path(config.output_dir) / "uploads"
    app.state.upload_dir.mkdir(parents=True, exist_ok=True)
    # Agent S's model, built once. Injected in tests; built here otherwise, and
    # LAZILY — importing langchain and constructing a client at boot would make
    # an API that never chats pay for one, and `require_api_key` would refuse to
    # start a deployment that only serves the dashboard.
    #
    # Absent, Agent S answers deterministically rather than erroring. That is a
    # real degradation and it is COUNTED: `answer_source` on every row says
    # 'template', so a key that silently stops working is visible in one query
    # instead of looking like a model that got terser.
    app.state.assistant_llm = assistant_llm
    if assistant_llm is None and os.getenv("OPENAI_API_KEY"):
        from agents.agent_s_assistant.schemas import AssistantReply
        from shared.llm import build_llm, structured
        app.state.assistant_llm = structured(build_llm(config), AssistantReply)

    # ---- session plumbing -------------------------------------------------
    def current_user(request: Request) -> Optional[dict[str, Any]]:
        token = request.cookies.get(SESSION_COOKIE)
        if not token:
            return None
        user_id, epoch = _user_from_token(token)
        if not user_id:
            return None
        user = request.app.state.store.user_by_id(user_id)
        if user is None:
            return None
        # The signature proves the cookie is ours; the epoch proves it has not
        # been revoked since. A password reset bumps the row, and every cookie
        # minted before it stops resolving here — which is what makes recovering
        # a compromised account actually evict whoever was in it.
        if int(user.get("session_epoch") or 0) != epoch:
            return None
        return user

    def require_user(request: Request) -> dict[str, Any]:
        user = current_user(request)
        if user is None:
            # The frontend treats 401 on these as "not signed in", not an error.
            raise _Unauthorized()
        return user

    def require_verified_user(request: Request) -> dict[str, Any]:
        """Signed in AND has proved the address. The gate, in one place.

        **This is what actually enforces verification.** The site redirects an
        unverified person to the code page and the app's route guards send them
        there too, but both are navigation: a request made with `curl`, or by a
        stale build deployed before this one, never sees either. Only a route
        that refuses can be relied on, which is the same reason `require_user`
        exists rather than trusting the app not to render a signed-out screen.

        Applied to the routes that ADVANCE onboarding, not to the reads. An
        unverified account has no completed run, so `/api/dashboard` and friends
        already answer 404 — gating them too would add a second failure mode
        without taking away a single capability.
        """
        user = require_user(request)
        if not user.get("email_verified_at"):
            raise _Unverified()
        return user

    class _Unauthorized(Exception):
        pass

    class _Unverified(Exception):
        pass

    @app.exception_handler(_Unauthorized)
    async def _unauth(_request: Request, _exc: _Unauthorized) -> JSONResponse:
        return JSONResponse({"error": "no_session"}, status_code=401)

    @app.exception_handler(_Unverified)
    async def _unverified(request: Request, _exc: _Unverified) -> JSONResponse:
        # 403, not 401: they ARE signed in, and answering 401 would make the app
        # bounce them to the login page they just came from — a loop, and a
        # confusing one, since logging in again changes nothing.
        # `verifyUrl` travels with it so the client does not have to reconstruct
        # a route the site owns.
        return JSONResponse({"error": "email_unverified",
                             "verifyUrl": verify_path(_locale_of(request))},
                            status_code=403)

    def public_user(row: dict[str, Any]) -> dict[str, Any]:
        return {"id": row["user_id"], "fullName": row["full_name"],
                "email": row["email"], "onboarded": bool(row["onboarded"]),
                "emailVerified": row.get("email_verified_at") is not None}

    def set_session(response: Response, user_id: str, locale: str,
                    epoch: int = 0) -> None:
        # httpOnly so no script can read it; SameSite=Lax because the site and the
        # app are same-site. Secure is omitted on http://localhost, where the
        # browser would otherwise drop the cookie and every local login would
        # silently fail to establish a session.
        # Keyed off ITQAN_ENV, not off a substring of the database URL. The old
        # heuristic inferred "am I local?" from the dev password appearing in the
        # DSN — correct by luck, and wrong the moment the DSN changes (which
        # deploying it does). Secure is omitted locally because the browser drops
        # a Secure cookie on http://localhost and every local login would then
        # silently fail to establish a session.
        secure = in_production()
        response.set_cookie(SESSION_COOKIE, _token_for(user_id, epoch), httponly=True,
                            samesite="lax", secure=secure, path="/")
        response.set_cookie(LOCALE_COOKIE, locale, httponly=False,
                            samesite="lax", secure=secure, path="/")

    # ---- auth: the site owns credentials, the app only reads the session ----
    #
    # BOTH paths are live, deliberately. `/api/placeholder/*` shipped the word
    # "placeholder" in a production URL and is renamed to `/api/auth/*`, but the
    # marketing site is static Astro: the HTML on the box posts to whatever path
    # it was BUILT with, and it deploys from a different repo by a different job.
    # Whichever of the two deploys lands second, there is a window where live
    # HTML posts to a path the live API might not have. The alias makes that
    # window harmless. Remove it once both sides are deployed and the old path
    # shows no traffic.
    @app.post("/api/auth/signup")
    @app.post("/api/placeholder/signup")   # alias — see the note below
    async def signup(request: Request, response: Response, email: str = Form(...),
                     password: str = Form(...), name: str = Form("")) -> Any:
        store: AppStore = app.state.store
        if store.user_by_email(email):
            return JSONResponse({"error": "email_taken"}, status_code=409)
        if password_problems(password):
            return JSONResponse({"error": "invalid_input"}, status_code=400)
        # The language they were actually reading when they signed up. This route
        # used to pass no locale at all, so every account — including one created
        # entirely on the English site — was stored 'ar' from the column default,
        # and every agent-authored string and email came back in Arabic. The
        # cookie is already written by the site's language toggle; nothing new is
        # needed to know this.
        user = store.create_user(email=email, full_name=name, password=password,
                                 locale=_locale_of(request))
        set_session(response, user["user_id"], user["locale"])
        # Signed in, but NOT verified: the session is what lets them ask for a
        # new code without this endpoint ever having to accept a bare email
        # address from an anonymous caller — which is the enumeration and
        # mail-bombing surface forgot-password had to be built around.
        _send_verification(user_id=user["user_id"], email=user["email"],
                           locale=user["locale"])
        return {"ok": True, "verifyUrl": verify_path(user["locale"])}

    def _send_verification(*, user_id: str, email: str, locale: str) -> None:
        """Mint a code, store its hash, mail the code. Used by signup and resend.

        The code exists as a string in this function and in the message it is
        formatted into. It is not returned, not logged and not stored — the store
        receives `sha256(code)`, and a caller wanting to know what was sent has to
        read the person's inbox, which is the point.
        """
        store: AppStore = app.state.store
        config: Config = app.state.config
        code = _new_code(config.verification_code_digits)
        store.issue_verification(user_id=user_id, code_hash=_sha256(code),
                                 minutes=config.verification_code_minutes)
        subject, body = email_module.verification_message(
            code=code, locale=locale, minutes=config.verification_code_minutes)
        email_module.send_in_background(to=email, subject=subject, body=body,
                                        config=config, purpose="verification")

    @app.post("/api/auth/login")
    @app.post("/api/placeholder/login")    # alias — see the note below
    async def login(response: Response, email: str = Form(...),
                    password: str = Form(...)) -> Any:
        store: AppStore = app.state.store
        row = store.user_by_email(email)
        # One branch for "no such account" and "wrong password" on purpose: telling
        # them apart is an account-enumeration oracle.
        if not row or not verify_password(password, row["password_hash"]):
            return JSONResponse({"error": "invalid_credentials"}, status_code=401)
        set_session(response, row["user_id"], row["locale"],
                    int(row.get("session_epoch") or 0))
        return {"ok": True}

    # ---- email verification ------------------------------------------------
    @app.post("/api/auth/verify-email")
    async def verify_email(request: Request, code: str = Form("")) -> Any:
        """Spend one attempt against the outstanding code.

        `Form("")` rather than `Form(...)` for the same reason the reset route
        uses it: an empty field must reach THIS handler and get an answer it can
        act on, not FastAPI's own 422 describing a malformed request.

        **422 for a wrong code**, carrying the attempts left. Telling them costs
        nothing — it is their own account and their own code — and not telling
        them means the fifth failure looks identical to the first, right up until
        the code silently dies.

        **410 for a code that is over** — expired, already used, or out of
        attempts. All three end the same way, with a new code, and the page shows
        one panel for that outcome. It is also the only honest answer available:
        once the attempt limit is spent we deliberately stop comparing, so we do
        not know whether what they typed was right.
        """
        user = require_user(request)
        store: AppStore = app.state.store
        config: Config = app.state.config

        if user.get("email_verified_at"):
            # Already done — the tab was left open, or a second submit landed.
            # 200, because the state they wanted is the state that holds.
            return {"ok": True, "alreadyVerified": True}

        row = store.claim_verification_attempt(
            user["user_id"], max_attempts=config.verification_max_attempts)
        if row is None:
            return JSONResponse({"error": "code_expired"}, status_code=410)

        import hmac
        # Constant-time, though the realistic attack here is guessing rather than
        # timing. It costs one call, and the alternative is a comparison whose
        # duration depends on how many leading digits were right.
        if not hmac.compare_digest(row["code_hash"], _sha256(code.strip())):
            remaining = max(0, config.verification_max_attempts - int(row["attempts"]))
            if remaining == 0:
                # That was the last one. Say so with the code-is-over answer
                # rather than a wrong-code answer, so the page offers a new code
                # instead of inviting a sixth try that cannot succeed.
                return JSONResponse({"error": "code_expired"}, status_code=410)
            return JSONResponse({"error": "invalid_code", "attemptsRemaining": remaining},
                                status_code=422)

        store.mark_email_verified(user["user_id"])
        store.consume_verification(user["user_id"])
        return {"ok": True}

    @app.post("/api/auth/resend-verification")
    async def resend_verification(request: Request) -> Any:
        """A new code, replacing the previous one.

        Always 200, including when rate limited: a throttled resend and a sent
        one look identical, so hammering the button tells the person nothing they
        could use and shows them the same "it is on its way" either way.

        Requires a session, so unlike forgot-password there is no address to
        enumerate and no stranger's inbox to aim at. The limits below bound a
        user hammering their own mailbox, and — per IP — one attacker doing it
        from many accounts, which the per-user limit cannot see.
        """
        user = require_user(request)
        config: Config = app.state.config
        store: AppStore = app.state.store
        same_answer: Any = {"ok": True}

        if user.get("email_verified_at"):
            return same_answer

        client_ip = request.client.host if request.client else "unknown"
        if not store.claim_reset_slot(_sha256(user["user_id"]), kind="verify_user",
                                      limit=config.verification_resends_per_user_hour):
            return same_answer
        if not store.claim_reset_slot(_sha256(client_ip), kind="verify_ip",
                                      limit=config.verification_resends_per_ip_hour):
            return same_answer

        _send_verification(user_id=user["user_id"], email=user["email"],
                           locale=user.get("locale") or _locale_of(request))
        return same_answer

    # ---- password recovery -------------------------------------------------
    @app.post("/api/auth/forgot-password")
    async def forgot_password(request: Request, email: str = Form(...)) -> Any:
        """Always 200, always the same body, whatever happened.

        A different answer — or a different response TIME — for an address with
        no account turns this form into a way to enumerate who is registered. So
        every branch below ends here identically: unknown address, rate-limited,
        relay down. The only party told about a failure is the operator, through
        the log and `email_module.SEND_FAILURES`.

        On timing, stated rather than overclaimed: the SMTP conversation is on a
        background thread, so the two branches differ by one indexed SELECT and
        one INSERT. That is microseconds, inside the jitter between two packets.
        An artificial fixed delay was considered and rejected — it would slow
        every legitimate request to hide a difference already lost in the noise.
        """
        store: AppStore = app.state.store
        config: Config = app.state.config
        same_answer: Any = {"ok": True}

        address = (email or "").strip().lower()
        client_ip = request.client.host if request.client else "unknown"

        # Hashed, so this table never becomes a list of who tried to recover an
        # account — the very fact the identical responses exist to protect.
        if not store.claim_reset_slot(_sha256(address), kind="email",
                                      limit=config.reset_requests_per_email_hour):
            return same_answer
        if not store.claim_reset_slot(_sha256(client_ip), kind="ip",
                                      limit=config.reset_requests_per_ip_hour):
            return same_answer

        row = store.user_by_email(address)
        if row is None:
            return same_answer

        token = secrets.token_urlsafe(32)
        store.create_password_reset(user_id=row["user_id"], token_hash=_sha256(token),
                                    minutes=config.reset_token_minutes)
        # The account's own locale wins: it is the person's stated preference,
        # and the person requesting a reset is not necessarily the person who
        # will read the mail. The request's cookie is the fallback, for an
        # account created before signup recorded one.
        locale = row.get("locale") or _locale_of(request)
        link = email_module.reset_link(site_url=config.site_url,
                                       locale=locale, token=token)
        subject, body = email_module.reset_message(
            link=link, locale=locale, minutes=config.reset_token_minutes)
        email_module.send_in_background(to=address, subject=subject, body=body,
                                        config=config)
        return same_answer

    @app.post("/api/auth/reset-password")
    async def reset_password(token: str = Form(""), password: str = Form("")) -> Any:
        """Spend a token and set a new password.

        Both fields default to empty rather than being required, so that THIS
        handler owns every case a user can actually reach. Declared `Form(...)`,
        a missing or empty token is rejected by FastAPI with its own 422 — and a
        mangled link would then show the generic "could not do that just now"
        instead of "this link has expired", which is the one message that tells
        the person what to do next.

        **410 for anything wrong with the token** — unknown, expired, already
        spent — because the front end renders 400 and 410 as the same "this link
        has expired" panel and the three are indistinguishable to the user
        anyway. Telling them apart would confirm that a token had once been
        issued for an address.

        **422 for a password that fails the rules**, and NOT 400: 400 is spoken
        for by that expired panel, so it would tell someone their link had died
        when their password was merely too short — sending them to fetch a new
        link that fails in exactly the same way. 422 lands on the front end's
        generic "could not do that just now", which is at least true.
        """
        store: AppStore = app.state.store

        # A missing or empty token is a broken link, not a malformed request:
        # answer it with the panel that tells them to ask for a new one.
        if not token.strip():
            return JSONResponse({"error": "invalid_token"}, status_code=410)

        # The password is checked BEFORE the token is spent. Failing the rules
        # must not consume a single-use token and force a second email over a
        # typo.
        if password_problems(password):
            return JSONResponse({"error": "invalid_password"}, status_code=422)

        claimed = store.consume_password_reset(_sha256(token))
        if claimed is None:
            return JSONResponse({"error": "invalid_token"}, status_code=410)

        user_id = claimed["user_id"]
        store.set_password(user_id, hash_password(password))
        # Any OTHER outstanding token dies with this one: two "forgot password"
        # clicks must not leave a spare key to an account its owner believes they
        # have just secured.
        store.invalidate_password_resets(user_id)
        # And every session issued before now stops resolving, which is the whole
        # point of recovering an account somebody else may be sitting in.
        store.bump_session_epoch(user_id)

        # Deliberately NOT signed in: the front end navigates to the login page,
        # and making them use the new password proves it is the one that works.
        return {"ok": True}

    @app.get("/api/handoff")
    async def handoff(request: Request) -> Any:
        """The site's forms navigate here after a successful submit.

        Same origin: the cookie already reaches the app, so there is nothing to
        hand over — but it must EXIST, or every sign-in lands on a 404. The `?t=`
        token `/api/session` accepts is the cross-origin version of this, for when
        the site and app are on different domains.
        """
        user = current_user(request)
        if user is None:
            return RedirectResponse("/", status_code=302)
        # Registered but unproved: the code page, not the app. Caught here as
        # well as in the app's own guards because the site is static — its built
        # HTML sends a new signup wherever it was compiled to, and a build made
        # before this change points straight at /app/.
        if not user.get("email_verified_at"):
            return RedirectResponse(verify_path(user.get("locale") or _locale_of(request)),
                                    status_code=302)
        return RedirectResponse("/app/", status_code=302)

    @app.get("/api/session")
    async def session(request: Request) -> Any:
        user = current_user(request)
        if user is None:
            return JSONResponse({"error": "no_session"}, status_code=401)
        return {
            "token": request.cookies.get(SESSION_COOKIE, ""),
            "user": public_user(user),
            "locale": request.cookies.get(LOCALE_COOKIE) or user["locale"],
        }

    @app.post("/api/logout")
    async def logout(response: Response) -> Any:
        response.delete_cookie(SESSION_COOKIE, path="/")
        return {"ok": True}

    # ---- documents ---------------------------------------------------------
    @app.post("/api/documents")
    async def upload(request: Request, file: UploadFile = File(...),
                     kind: str = Form("other")) -> Any:
        user = require_verified_user(request)
        if kind not in DOCUMENT_KINDS:
            return JSONResponse({"error": "invalid_input"}, status_code=400)

        payload = await file.read()
        if len(payload) > MAX_UPLOAD_BYTES:
            return JSONResponse({"error": "file_too_large"}, status_code=400)
        if not payload:
            return JSONResponse({"error": "empty_file"}, status_code=400)

        # Never trust the client's filename for a path: a name like
        # "../../.env" would otherwise decide where bytes land. The stored name is
        # generated; the original is kept only as data to show the user.
        safe = f"{secrets.token_hex(8)}{Path(file.filename or '').suffix[:10]}"
        target = app.state.upload_dir / user["user_id"] / safe
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)

        row = app.state.store.add_document(
            user_id=user["user_id"], file_name=file.filename or "document",
            mime_type=file.content_type or "application/octet-stream",
            size_bytes=len(payload), kind=kind, stored_path=str(target))
        return mapping.uploaded_document(row)

    # ---- analysis: async job, real stage-driven progress -------------------
    @app.post("/api/analysis")
    async def start_analysis(request: Request) -> Any:
        user = require_verified_user(request)
        body = await request.json()
        # camelCase, as `http.ts` sends it.
        ids = body.get("documentIds") or []
        docs = app.state.store.documents(user["user_id"], list(ids))
        if not docs:
            return JSONResponse({"error": "no_documents"}, status_code=400)
        if not any(d["kind"] == REQUIRED_KIND for d in docs):
            # Agent A cannot run without a CV, so refuse here with a nameable
            # reason rather than letting the pipeline fail three stages in.
            return JSONResponse({"error": "cv_required"}, status_code=400)

        run_id = _new_run_id()
        job_id = app.state.store.create_run(
            user_id=user["user_id"], run_id=run_id, document_ids=[d["document_id"] for d in docs])
        jobs_module.spawn(
            app.state.store, app.state.runner, job_id=job_id, run_id=run_id,
            cv_paths=[d["stored_path"] for d in docs if d["kind"] == REQUIRED_KIND],
            transcript_paths=[d["stored_path"] for d in docs if d["kind"] == "transcript"])
        return {"jobId": job_id}

    @app.get("/api/analysis/{job_id}")
    async def poll_analysis(request: Request, job_id: str) -> Any:
        user = require_user(request)
        row = app.state.store.run(job_id=job_id, user_id=user["user_id"])
        if row is None:
            return JSONResponse({"error": "no_job"}, status_code=404)
        out: dict[str, Any] = {"jobId": job_id, "stage": row["stage"],
                               "progress": float(row["progress"])}
        if row["stage"] == "failed":
            out["error"] = row["error_code"] or jobs_module.ERROR_UNKNOWN
        elif row["profile"] and row["stage"] in (jobs_module.STAGE_AWAITING, "done"):
            # `awaiting_confirmation` as well as `done`, and this is the line that
            # unblocks the confirm screen. Agent A's extraction is what that screen
            # asks the user to check; withholding it until the course recommender
            # finished meant three minutes of skeleton over data already on disk.
            out["result"] = mapping.analysis_result(row["profile"])
        return out

    # ---- onboarding progress ----------------------------------------------
    @app.get("/api/onboarding/progress")
    async def get_progress(request: Request) -> Any:
        user = require_user(request)
        return app.state.store.get_progress(user["user_id"])

    @app.put("/api/onboarding/progress")
    async def put_progress(request: Request) -> Any:
        user = require_verified_user(request)
        app.state.store.put_progress(user["user_id"], await request.json())
        return {"ok": True}

    @app.delete("/api/onboarding/progress")
    async def delete_progress(request: Request) -> Any:
        user = require_user(request)
        app.state.store.clear_progress(user["user_id"])
        return {"ok": True}

    # ---- the confirmed profile: also the phase-two trigger -----------------
    @app.post("/api/profile")
    async def confirm_profile(request: Request) -> Any:
        """Confirming the details is what STARTS the matching.

        This route was a bookkeeping write; it is now the hinge of the flow. The
        payload already carried `preferences` — the four answers the user gave
        while Agent A was reading — and those answers used to arrive after Agent C
        had already run, which made them decorative. Starting phase two from here
        is what makes them inputs.
        """
        user = require_verified_user(request)
        payload = await request.json()
        awaiting = app.state.store.awaiting_run(user["user_id"])
        latest = app.state.store.latest_completed_run(user["user_id"])
        # The run this profile belongs to: the one about to be matched, or failing
        # that the last one that produced envelopes (the manual-entry route and a
        # user editing their details later both land here).
        run_id = (awaiting or latest or {}).get("run_id")
        app.state.store.save_profile(user["user_id"], payload, run_id)
        # On the ACCOUNT, so returning on another device does not restart the flow.
        app.state.store.mark_onboarded(user["user_id"])
        app.state.store.clear_progress(user["user_id"])

        if awaiting is None:
            # No paused run: manual entry with no documents, or a re-confirmation
            # after the run already finished. Exactly the old behaviour.
            return {"ok": True}

        preferences = payload.get("preferences") or {}
        app.state.store.save_run_preferences(awaiting["job_id"], preferences)
        jobs_module.spawn_phase_two(
            app.state.store, app.state.runner, job_id=awaiting["job_id"],
            run_id=awaiting["run_id"], preferences=preferences,
            # Agent A's envelope, so phase two can rebuild the file Agent C reads
            # if a restart lost it. The wait between the phases ends when a PERSON
            # acts, so it is unbounded.
            profile=awaiting.get("profile"))
        # The job id goes back so the UI can keep watching: the agents are still
        # working when the user lands on the dashboard, and a progress bar they can
        # see is the difference between "still going" and "broken".
        return {"ok": True, "jobId": awaiting["job_id"]}

    # ---- the profile screen ------------------------------------------------
    def _avatar_url(user: dict[str, Any]) -> Optional[str]:
        """Same-origin, and only when a file is actually recorded."""
        return f"/api/profile/avatar/{user['user_id']}" if user.get("avatar_path") else None

    @app.get("/api/profile")
    async def get_profile(request: Request) -> Any:
        """`StoredProfile`, or 404 when nothing has been confirmed.

        404 is the CONTRACT here, not a failure: the app renders it as an empty
        state ("nothing confirmed yet"), exactly as /api/dashboard already does.

        This route did not exist while the app was calling it, and the client
        does `.catch(() => null)` — so the profile screen showed its empty state
        for every user, however complete their profile, with nothing in any log.
        """
        user = require_user(request)
        store: AppStore = app.state.store
        confirmed = store.profile(user["user_id"])
        if confirmed is None:
            return JSONResponse({"error": "no_profile"}, status_code=404)
        completed = store.latest_completed_run(user["user_id"]) or {}
        return mapping.stored_profile(
            confirmed=confirmed,
            account=user,
            documents=[mapping.uploaded_document(d)
                       for d in store.all_documents(user["user_id"])],
            gap=completed.get("skill_gap"),
            avatar_url=_avatar_url(user),
        )

    @app.put("/api/profile")
    async def update_profile(request: Request) -> Any:
        """An edit from the profile screen. MUST NOT re-run the pipeline.

        It writes the same row `POST /api/profile` does, and the difference is
        the whole point: POST is the end of onboarding and starts phase two,
        while this is someone correcting a phone number. If this ever spawned a
        run, every edit would cost a full re-match and silently change the
        dashboard under the user — which is why it is a separate route and has
        its own test rather than a comment.
        """
        user = require_verified_user(request)
        payload = await request.json()
        store: AppStore = app.state.store
        # Keep the run this profile already belonged to; an edit does not re-bind
        # it to a different run, and it must not orphan it either.
        existing = store.profile(user["user_id"]) or {}
        store.save_profile(user["user_id"], payload, existing.get("run_id"))
        return {"ok": True}

    @app.post("/api/profile/avatar")
    async def upload_avatar(request: Request, file: UploadFile = File(...)) -> Any:
        """Return the URL; never accept one. The server owns storage, so a client
        can never point this at a path it chose."""
        user = require_verified_user(request)
        try:
            path = avatars.store_avatar(user["user_id"], await file.read(),
                                        config=app.state.config)
        except avatars.AvatarRejected as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        store: AppStore = app.state.store
        # The old file goes only once the new one is safely written.
        avatars.remove_avatar(user.get("avatar_path"))
        store.set_avatar_path(user["user_id"], path)
        return {"avatarUrl": f"/api/profile/avatar/{user['user_id']}"}

    @app.delete("/api/profile/avatar", status_code=204)
    async def delete_avatar(request: Request) -> Response:
        user = require_user(request)
        avatars.remove_avatar(user.get("avatar_path"))
        app.state.store.set_avatar_path(user["user_id"], None)
        return Response(status_code=204)

    @app.get("/api/profile/avatar/{user_id}")
    async def serve_avatar(user_id: str) -> Any:
        """Public by design: it is an <img src> on a page the owner is already
        looking at, and requiring the session cookie on an image request buys
        nothing while breaking caching. The filename carries random bytes, so the
        URL cannot be guessed from a user id alone."""
        row = app.state.store.user_by_id(user_id)
        path = (row or {}).get("avatar_path")
        if not path or not avatars.is_within_avatar_dir(path, config=app.state.config):
            return JSONResponse({"error": "not_found"}, status_code=404)
        try:
            data = Path(path).read_bytes()
        except OSError:
            # Recorded but gone from disk — a restore, or a hand-cleaned volume.
            return JSONResponse({"error": "not_found"}, status_code=404)
        return Response(data, media_type=avatars.content_type(path),
                        headers={"Cache-Control": "private, max-age=300"})

    # ---- reads: never run an agent here -----------------------------------
    def _envelopes(request: Request) -> Optional[dict[str, Any]]:
        user = require_user(request)
        return app.state.store.latest_completed_run(user["user_id"])

    @app.get("/api/dashboard")
    async def dashboard(request: Request) -> Any:
        row = _envelopes(request)
        if row is None:
            return JSONResponse({"error": "no_analysis"}, status_code=404)
        return mapping.dashboard(row["profile"] or {}, row["skill_gap"] or {},
                                 row["recommendations"] or {})

    @app.get("/api/jobs")
    async def jobs_route(request: Request) -> Any:
        row = _envelopes(request)
        # An empty list, not a 404: "no matches yet" is a normal state the UI
        # renders as an empty view, and a 404 would read as a broken route.
        return mapping.job_matches(row["skill_gap"] or {}) if row else []

    @app.get("/api/courses")
    async def courses_route(request: Request) -> Any:
        row = _envelopes(request)
        return mapping.courses(row["recommendations"] or {}) if row else []

    # ---- Agent S -----------------------------------------------------------
    # Registered last, and given `require_user` rather than reaching for it:
    # every route in there is scoped to the session's user, and passing the
    # dependency in keeps that visible at the mount point instead of buried.
    assistant_module.register(app, require_user=require_user,
                              jobs_module=jobs_module, mapping=mapping)

    @app.get("/api/health")
    async def health() -> Any:
        return {"ok": True}

    return app


# ---------------------------------------------------------------------------
# Session tokens.
#
# Opaque and signed with a server secret, so a cookie cannot be forged by
# guessing a user id. Deliberately NOT a JWT: nothing here needs stateless
# verification across services, and a JWT would invite putting claims in it that
# then go stale (the frontend's own note that `onboarded` must live on the row,
# not in a cookie, is exactly that failure).
# ---------------------------------------------------------------------------
_DEV_SECRET = "dev-only-not-for-production"


def in_production() -> bool:
    """One switch for every "is this a real deployment?" decision.

    Previously each site guessed for itself — the Secure cookie flag was inferred
    from whether the database URL began with the local dev password, which worked
    by accident and would break the moment the DSN changed.
    """
    import os

    return os.getenv("ITQAN_ENV", "development").lower() == "production"


def _secret() -> bytes:
    import os
    return (os.getenv("ITQAN_SESSION_SECRET") or _DEV_SECRET).encode()


def assert_deployable() -> None:
    """Refuse to serve with a session secret anyone can read.

    The fallback above is a literal string in a **public** repository, and the
    cookie is just `user_id.HMAC(secret, user_id)` — so with the default in place,
    anyone who has seen this file can mint a valid session for any account. That
    is total authentication bypass, and it is silent: everything works perfectly
    right up until someone tries it.

    A deployment that will not boot beats one that boots insecure, so this raises
    rather than warns. Development is untouched.
    """
    if not in_production():
        return
    import os

    if (os.getenv("ITQAN_SESSION_SECRET") or _DEV_SECRET) == _DEV_SECRET:
        raise RuntimeError(
            "ITQAN_SESSION_SECRET is unset in production. Session cookies are "
            "signed with it, and the development fallback is public in this "
            "repository — every account would be forgeable. Set it to a long "
            "random value (`python -c \"import secrets;print(secrets.token_hex(32))\"`) "
            "and redeploy."
        )

    if not os.getenv("ITQAN_SMTP_HOST"):
        raise RuntimeError(
            "ITQAN_SMTP_HOST is unset in production. Password recovery answers 200 "
            "whether or not an address has an account — that is what stops the form "
            "leaking who is registered — so with no relay it would accept every "
            "request and send nothing, and NOBODY would find out: the user is told "
            "to check their email either way. Refusing to start is the only loud "
            "failure available. Set ITQAN_SMTP_{HOST,PORT,USER,PASSWORD,FROM}."
        )


def _token_for(user_id: str, epoch: int = 0) -> str:
    """`user_id.epoch.mac`, signed over the first two parts.

    The epoch is what makes a session revocable. Before it, this was
    `user_id + HMAC(secret, user_id)` — deterministic and permanent, so a cookie
    captured once worked forever and a password reset did nothing to it. That is
    the single case password recovery exists for, and it was the case it failed.

    Changing the format invalidates every cookie minted under the old one, which
    signs everybody out exactly once. Accepted deliberately (2026-08-16).
    """
    import hmac
    payload = f"{user_id}.{epoch}"
    mac = hmac.new(_secret(), payload.encode(), "sha256").hexdigest()[:32]
    return f"{payload}.{mac}"


def _user_from_token(token: str) -> tuple[str, int]:
    """`(user_id, epoch)`, or `("", 0)` if the signature does not hold.

    The epoch is returned rather than checked here because verifying it needs the
    database, and this function deliberately does not have one. The caller
    compares it against the row — see `current_user`.
    """
    import hmac
    payload, _, mac = token.rpartition(".")
    user_id, _, raw_epoch = payload.rpartition(".")
    if not user_id or not raw_epoch.isdigit():
        return "", 0
    epoch = int(raw_epoch)
    if not hmac.compare_digest(mac, _token_for(user_id, epoch).rpartition(".")[2]):
        return "", 0
    return user_id, epoch


def _new_run_id() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(4)


def get_app() -> FastAPI:
    """The entry point:

        uvicorn api.main:get_app --factory --port 8000

    A factory rather than a module-level `app`, because building the app connects
    to Postgres and applies migrations — work that must not happen merely because
    something imported this module (the test suite injects its own store and
    passes `migrate=False`).

    There used to be an `app = None` here "for uvicorn". It was a trap:
    `uvicorn api.main:app` started, printed "Application startup complete", and
    then returned a 500 for every single request. A missing attribute fails
    immediately and says so, which is the better of the two.
    """
    return create_app()
