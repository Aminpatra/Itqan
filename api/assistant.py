"""Agent S's routes: where the limits and the scoping are actually enforced.

The spec this was built from asks the model to check quotas and refuse
out-of-scope questions. **It does not work that way here**, and the difference
is the whole design.

This module claims quota with a guarded UPDATE *before* the graph is built, and
builds the model's entire world from one user's rows. So:

* over quota, the model never runs — there is no instruction to disregard;
* the model cannot reach another user, because the fact sheet it is handed holds
  one user's results and no store method behind it takes a user id from the
  request. The id comes from the signed session cookie, every time;
* the model cannot spend a credit. It may set `proposedRerun`; only
  ``POST /api/assistant/rerun`` with an explicit confirmation consumes one.

That last split matters more at these quotas than it would at generous ones:
**one rerun per week** means a credit spent because a model was persuasive, or
because a sentence in the conversation said "confirm: true", is the user's
entire allowance gone.

The lesson underneath all of this is the repo's own, measured three times:
`work_arrangement` fabricated on 19 of 19 postings against a prompt forbidding
it; Agent E calling an unpriced course free once in 25 draws with the warning
present; Agent A paraphrasing quotes it was told to copy. Instructions are not
controls.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta
from typing import Any, Optional
from zoneinfo import ZoneInfo

from fastapi import Request
from fastapi.responses import JSONResponse

from agents.agent_s_assistant.facts import build_fact_sheet
from agents.agent_s_assistant.graph import build_assistant_graph
from agents.agent_s_assistant.nodes import Deps
from shared.config import Config

MAX_QUESTION_CHARS = 2_000


# ---------------------------------------------------------------------------
# Quota periods.
#
# Local, not UTC. A user in Muscat told their quota resets at midnight and
# finding it reset at 4am has been told something false, and taking UTC because
# it is the default is exactly how that happens.
#
# A period is identified by its start DATE, so a new day or week is simply a row
# that does not exist yet. Nothing resets anything; there is no scheduled job to
# fail silently at midnight.
# ---------------------------------------------------------------------------
def _now(config: Config) -> datetime:
    return datetime.now(ZoneInfo(config.assistant_tz))


def day_start(config: Config) -> date:
    return _now(config).date()


def week_start(config: Config) -> date:
    """Monday of the current local week."""
    today = _now(config).date()
    return today - timedelta(days=today.weekday())


def resets_at(config: Config, *, kind: str) -> str:
    """When the current period ends, as an ISO instant the UI can render.

    Derived from the period rather than stored, which is what makes "no reset
    job" true rather than merely intended.
    """
    now = _now(config)
    if kind == "message":
        nxt = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        nxt = (now + timedelta(days=7 - now.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0)
    return nxt.isoformat()


def _period(config: Config, kind: str) -> date:
    return day_start(config) if kind == "message" else week_start(config)


def _limit(config: Config, kind: str) -> int:
    return (config.assistant_daily_messages if kind == "message"
            else config.assistant_weekly_reruns)


def quota_state(store: Any, config: Config, user_id: str, kind: str) -> dict[str, Any]:
    limit = _limit(config, kind)
    used = store.quota_used(user_id, kind=kind, period_start=_period(config, kind))
    return {"used": used, "limit": limit, "remaining": max(0, limit - used),
            "resetsAt": resets_at(config, kind=kind)}


# ---------------------------------------------------------------------------
def gaps_from(gap: dict[str, Any]) -> list[Any]:
    """The missing skills, richest form available, already in Agent C's order.

    Prefers `missing_skill_details`, which carries `jobs_missing_in` — a measured
    count that makes an answer specific ("3 of the roles you matched asked for
    it") instead of a bare list. Falls back to `most_common_missing_skills` for
    an envelope written before Agent C published the detail, which is the same
    legacy shape Agent E still tolerates.

    `priority_score` is deliberately NOT carried through: it is an internal
    weight on no meaningful scale, and `_FORBIDDEN` rejects any answer naming it.
    """
    aggregate = gap.get("aggregate") or {}
    details = aggregate.get("missing_skill_details")
    if details:
        return [{"skill": d.get("skill"), "jobs_missing_in": d.get("jobs_missing_in"),
                 "low_confidence": d.get("low_confidence")} for d in details][:10]
    return list(aggregate.get("most_common_missing_skills") or [])[:10]


def _newest(docs: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    """The most recent document of one kind, or none.

    `all_documents` returns newest first, so this takes the head. One CV, not
    every CV a person has ever uploaded: re-reading three old resumes alongside
    the current one would blend them into a profile that describes nobody.
    """
    matches = [d for d in docs if d.get("kind") == kind]
    return matches[:1]



def graph_for(app: Any) -> Any:
    """Built per request so a config or key change is picked up, and so the LLM
    is absent in tests unless a test supplies one."""
    return build_assistant_graph(
        Deps(config=app.state.config,
             llm=getattr(app.state, "assistant_llm", None)))


def facts_for(store: Any, user_id: str) -> tuple[str, bool, Optional[str],
                                                 list[dict[str, Any]], list[dict[str, Any]]]:
    """(fact sheet, has_results, run_id, jobs, courses).

    Module level, and shared by BOTH surfaces — `/api/assistant/*` and
    `/api/chat/*` — so there is one place that decides what a model may see. A
    second copy is how the two would eventually disagree about scope, which is
    the one thing here that must never differ.

    The raw `jobs` and `courses` come back alongside the rendered sheet because
    the chat needs them: the sheet gives the model handles like `[J2]`, and
    resolving a handle means indexing these very lists. They are
    `api/mapping`'s shapes, identical to what `/api/jobs` and `/api/courses`
    serve.

    Every read is keyed on the session's `user_id`. No branch takes an id from a
    request, which is what makes cross-user isolation a property of this
    function rather than of a prompt.
    """
    from . import mapping

    row = store.latest_completed_run(user_id)
    profile_prefs = (store.profile(user_id) or {}).get("payload") or {}
    knows_role = ((profile_prefs.get("preferences") or {}).get("knowsRole")) or None

    if row is None:
        return (build_fact_sheet(readiness=None, jobs=[], courses=[], gaps=[],
                                 suggested_role=None, matched_at=None,
                                 knows_role=knows_role),
                False, None, [], [])

    gap = row["skill_gap"] or {}
    recs = row["recommendations"] or {}
    board = mapping.dashboard(row["profile"] or {}, gap, recs)
    finished = row.get("finished_at")
    jobs = mapping.job_matches(gap, limit=10)
    courses = mapping.courses(recs)[:10]

    return (
        build_fact_sheet(
            readiness=board.get("readiness"),
            jobs=jobs,
            courses=courses,
            gaps=gaps_from(gap),
            suggested_role=mapping.suggested_role(gap),
            knows_role=knows_role,
            matched_at=finished.isoformat() if hasattr(finished, "isoformat") else finished,
        ),
        True,
        row.get("run_id"),
        jobs,
        courses,
    )


def register(app: Any, *, require_user, jobs_module, mapping) -> None:
    """Mount Agent S's routes. Called from `create_app`."""

    def _store():
        return app.state.store

    def _config() -> Config:
        return app.state.config

    def _graph():
        return graph_for(app)

    # ---- the fact sheet: one user's results, assembled in code -------------
    def _facts(user_id: str) -> tuple[str, bool, Optional[str]]:
        """Delegates to the shared builder, dropping the raw lists this surface
        does not use. One implementation, so `/api/assistant/*` and `/api/chat/*`
        cannot come to disagree about what a model may see."""
        sheet, has_results, run_id, _jobs, _courses = facts_for(_store(), user_id)
        return sheet, has_results, run_id

    # ---- routes -----------------------------------------------------------
    @app.get("/api/assistant/usage")
    async def usage(request: Request) -> Any:
        user = require_user(request)
        config = _config()
        return {
            "messages": quota_state(_store(), config, user["user_id"], "message"),
            "reruns": quota_state(_store(), config, user["user_id"], "rerun"),
        }

    @app.get("/api/assistant/messages")
    async def history(request: Request) -> Any:
        user = require_user(request)
        turns = _store().assistant_history(
            user["user_id"], limit=_config().assistant_history_turns * 2)
        return [{"id": t["message_id"], "role": t["role"], "content": t["content"],
                 "createdAt": t["created_at"].isoformat()} for t in turns]

    @app.post("/api/assistant/messages")
    async def ask(request: Request) -> Any:
        user = require_user(request)
        store, config = _store(), _config()
        body = await request.json()
        question = (body.get("text") or "").strip()

        if not question:
            return JSONResponse({"error": "empty_message"}, status_code=400)
        if len(question) > MAX_QUESTION_CHARS:
            # Bounded before the quota is claimed: a message too long to send is
            # not a message the user should be charged for.
            return JSONResponse({"error": "message_too_long"}, status_code=400)

        period = day_start(config)
        used = store.claim_quota(user["user_id"], kind="message",
                                 limit=config.assistant_daily_messages,
                                 period_start=period)
        if used is None:
            # The spec's wording, and no part of the request is processed. The
            # model is not called; nothing is logged against the user.
            limit = config.assistant_daily_messages
            return JSONResponse(
                {"error": "message_limit_reached",
                 "message": (f"You've reached today's message limit ({limit}/{limit}). "
                             f"It resets at {resets_at(config, kind='message')}."),
                 "usage": quota_state(store, config, user["user_id"], "message")},
                status_code=429)

        try:
            fact_sheet, has_results, run_id = _facts(user["user_id"])
            history_rows = store.assistant_history(
                user["user_id"], limit=config.assistant_history_turns * 2)

            store.add_assistant_message(user_id=user["user_id"], role="user",
                                        content=question, run_id=run_id)

            result = _graph().invoke({
                "question": question,
                "fact_sheet": fact_sheet,
                "history": [{"role": r["role"], "content": r["content"]}
                            for r in history_rows],
                "has_results": has_results,
            })
        except Exception:
            # The work did not happen, so the message does not count. This is why
            # quota is claimed-then-refunded rather than checked-then-incremented:
            # the latter is the race this design exists to avoid.
            store.refund_quota(user["user_id"], kind="message", period_start=period)
            raise

        answer = result.get("answer") or ""

        # An OUTAGE is not a message. The model was absent or raised, so the
        # reply is a sentence we generated ourselves, and taking one of somebody's
        # ten daily messages for it would be charging them for our downtime.
        #
        # A verification REJECTION is charged, and the asymmetry is deliberate:
        # there the model ran and we paid for it, and refunding would let someone
        # chat indefinitely for free by steering it into rejections.
        if result.get("model_failed"):
            store.refund_quota(user["user_id"], kind="message", period_start=period)

        store.add_assistant_message(
            user_id=user["user_id"], role="assistant", content=answer,
            run_id=run_id, answer_source=result.get("answer_source") or "template")

        rerun_quota = quota_state(store, config, user["user_id"], "rerun")
        payload: dict[str, Any] = {
            "answer": answer,
            "usage": quota_state(store, config, user["user_id"], "message"),
        }
        # Only ever a suggestion, and only offered when there is actually a
        # credit to spend — proposing something the user cannot do is worse than
        # not proposing it.
        if result.get("proposed_rerun") and rerun_quota["remaining"] > 0 and has_results:
            payload["proposedRerun"] = {
                "reason": result.get("rerun_reason"),
                "credits": rerun_quota,
            }
        return payload

    @app.post("/api/assistant/rerun")
    async def rerun(request: Request) -> Any:
        """Spend one weekly credit to re-run. Two modes.

        The ONLY path that consumes a rerun credit, and it requires the user to
        have said so in this request. The model has no way to reach here: it
        returns text and a boolean, and neither is routed to this function.

        * **`match`** (default) — Agents C and E against the corpus as it stands,
          reusing the confirmed profile. No document is re-read, and it runs to
          `done` on its own.
        * **`full`** — Agent A as well, from the user's documents as they stand
          now, so a newly uploaded CV is picked up.

        **A full rerun STOPS at `awaiting_confirmation`, by design** (user
        decision, 2026-08-15). That pause is where a person corrects what the
        model extracted, and running past it would publish a fresh extraction
        nobody reviewed — silently overwriting any correction they made last
        time. So this returns `awaitingConfirmation: true` and the caller must
        say plainly that the run is waiting for them, not finished.
        """
        user = require_user(request)
        store, config = _store(), _config()
        body = await request.json()

        mode = (body.get("mode") or "match").strip().lower()
        if mode not in ("match", "full"):
            return JSONResponse({"error": "unknown_mode"}, status_code=400)

        if body.get("confirm") is not True:
            # Explicit, in this request. At one credit a week, an accidental
            # spend is the user's whole allowance.
            return JSONResponse(
                {"error": "confirmation_required",
                 "message": ("This will use your rerun credit for this week."
                             if mode == "match" else
                             "This will use your rerun credit for this week, re-read "
                             "your documents, and ask you to confirm the details again."),
                 "usage": quota_state(store, config, user["user_id"], "rerun")},
                status_code=400)

        # Everything that makes a run impossible is checked BEFORE the credit is
        # claimed. A user must not be able to spend their whole weekly allowance
        # discovering that there was nothing to run.
        latest = store.latest_completed_run(user["user_id"])
        docs: list[dict[str, Any]] = []
        if mode == "match":
            if latest is None:
                return JSONResponse({"error": "no_completed_run"}, status_code=409)
        else:
            # NOT blocked by an existing paused run, and that was a real bug
            # caught by running this against a live account rather than fixtures.
            #
            # `stale_runs` deliberately never expires `awaiting_confirmation` —
            # someone taking ten minutes over the form is not a crashed process —
            # so abandoned onboarding attempts accumulate and live forever. The
            # dev account had FOUR, none newer than a week. A guard on "any run
            # is paused" therefore meant: anyone who ever abandoned onboarding can
            # never re-run again.
            #
            # Nothing is lost by allowing it. `awaiting_run` returns the NEWEST,
            # which is the one the app shows and the one `POST /api/profile`
            # binds to, so the older ones are already invisible. And the double-
            # spend worry is covered by the credit itself: at one a week, a second
            # attempt is refused by the quota, not by this.
            docs = store.all_documents(user["user_id"])
            if not any(d["kind"] == "cv" for d in docs):
                # Agent A cannot run without a CV. Named here rather than failing
                # three stages in, which is the same courtesy /api/analysis does.
                return JSONResponse({"error": "cv_required"}, status_code=409)

        period = week_start(config)
        used = store.claim_quota(user["user_id"], kind="rerun",
                                 limit=config.assistant_weekly_reruns,
                                 period_start=period)
        if used is None:
            return JSONResponse(
                {"error": "rerun_limit_reached",
                 "message": (f"You have no rerun credits left. They reset at "
                             f"{resets_at(config, kind='rerun')}."),
                 "usage": quota_state(store, config, user["user_id"], "rerun")},
                status_code=429)

        try:
            if mode == "match":
                profile = store.profile(user["user_id"]) or {}
                preferences = (profile.get("payload") or {}).get("preferences") or {}
                job_id = store.create_run(user_id=user["user_id"],
                                          document_ids=[], run_id=latest["run_id"])
                jobs_module.spawn_phase_two(
                    store, app.state.runner, job_id=job_id, run_id=latest["run_id"],
                    preferences=preferences, profile=latest.get("profile"))
                awaiting = False
            else:
                # A NEW run id: Agent A writes into `output/<run_id>/`, and reusing
                # the old one would overwrite a completed run's artifacts with a
                # half-finished one before the user has confirmed anything.
                run_id = f"run_{uuid.uuid4().hex[:12]}"
                # The newest of each kind, so a CV uploaded since the last run is
                # what gets read — which is the main reason to want this at all.
                cvs = _newest(docs, "cv")
                transcripts = _newest(docs, "transcript")
                job_id = store.create_run(
                    user_id=user["user_id"], run_id=run_id,
                    document_ids=[d["document_id"] for d in cvs + transcripts])
                jobs_module.spawn(
                    store, app.state.runner, job_id=job_id, run_id=run_id,
                    cv_paths=[d["stored_path"] for d in cvs],
                    transcript_paths=[d["stored_path"] for d in transcripts])
                awaiting = True
        except Exception:
            store.refund_quota(user["user_id"], kind="rerun", period_start=period)
            raise

        return {"jobId": job_id, "mode": mode,
                # Load-bearing for the caller's wording: a full rerun is NOT
                # finished when this returns, it is waiting for a person.
                "awaitingConfirmation": awaiting,
                "usage": quota_state(store, config, user["user_id"], "rerun")}
