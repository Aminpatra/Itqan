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
from . import jobs as jobs_module
from . import mapping
from .db import AppStore, apply_migrations, verify_password

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
        return request.app.state.store.user_by_id(_user_from_token(token))

    def require_user(request: Request) -> dict[str, Any]:
        user = current_user(request)
        if user is None:
            # The frontend treats 401 on these as "not signed in", not an error.
            raise _Unauthorized()
        return user

    class _Unauthorized(Exception):
        pass

    @app.exception_handler(_Unauthorized)
    async def _unauth(_request: Request, _exc: _Unauthorized) -> JSONResponse:
        return JSONResponse({"error": "no_session"}, status_code=401)

    def public_user(row: dict[str, Any]) -> dict[str, Any]:
        return {"id": row["user_id"], "fullName": row["full_name"],
                "email": row["email"], "onboarded": bool(row["onboarded"])}

    def set_session(response: Response, user_id: str, locale: str) -> None:
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
        response.set_cookie(SESSION_COOKIE, _token_for(user_id), httponly=True,
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
    async def signup(response: Response, email: str = Form(...), password: str = Form(...),
                     name: str = Form("")) -> Any:
        store: AppStore = app.state.store
        if store.user_by_email(email):
            return JSONResponse({"error": "email_taken"}, status_code=409)
        if password_problems(password):
            return JSONResponse({"error": "invalid_input"}, status_code=400)
        user = store.create_user(email=email, full_name=name, password=password)
        set_session(response, user["user_id"], user["locale"])
        return {"ok": True}

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
        set_session(response, row["user_id"], row["locale"])
        return {"ok": True}

    @app.get("/api/handoff")
    async def handoff(request: Request) -> Any:
        """The site's forms navigate here after a successful submit.

        Same origin: the cookie already reaches the app, so there is nothing to
        hand over — but it must EXIST, or every sign-in lands on a 404. The `?t=`
        token `/api/session` accepts is the cross-origin version of this, for when
        the site and app are on different domains.
        """
        signed_in = bool(current_user(request))
        return RedirectResponse("/app/" if signed_in else "/", status_code=302)

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
        user = require_user(request)
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
        user = require_user(request)
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
        user = require_user(request)
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
        user = require_user(request)
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
        user = require_user(request)
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
        user = require_user(request)
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


def _token_for(user_id: str) -> str:
    import hmac
    mac = hmac.new(_secret(), user_id.encode(), "sha256").hexdigest()[:32]
    return f"{user_id}.{mac}"


def _user_from_token(token: str) -> str:
    import hmac
    user_id, _, mac = token.rpartition(".")
    if not user_id or not hmac.compare_digest(mac, _token_for(user_id).split(".")[-1]):
        return ""
    return user_id


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
