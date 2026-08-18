"""Hud's chat: the product surface over Agent S.

`api/assistant.py` is the same engine with a plainer shape — it stays because it
is deployed, tested and is what the CLI mirrors. Both register against one store,
one quota and one graph, so the things that must not drift (who may be read, what
a message costs, what the model is shown) cannot.

**The rule this module exists to keep is "attach, never describe."** A posting
belongs in `jobs` as a whole card carrying its own `why`, `source` and
`retrievedAt`; the same posting written into Hud's prose is a claim with its
evidence stripped off. That is not a style preference — the brand bars the mascot
from verdicts and scores, and the exception granted for this screen rests
entirely on actionable things arriving as cards. So the model never emits a card.
It names a handle from the fact sheet, and `resolve_refs` decides whether that
handle points at something it was actually shown.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from agents.agent_s_assistant.facts import resolve_refs
from shared.config import Config

MAX_QUESTION_CHARS = 2_000
MAX_ATTACHMENTS = 5
MAX_ATTACHMENT_BYTES = 10 * 1024 * 1024
TITLE_CHARS = 60

# Ends the out-of-messages turn, and ONLY that turn.
#
# Written as escapes rather than pasted, because it is two codepoints — U+261D
# INDEX POINTING UP plus U+1F3FB EMOJI MODIFIER FITZPATRICK TYPE-1-2 — and a lost
# modifier renders as a different, yellower emoji while looking identical in a
# diff. A test asserts the exact sequence.
#
# In the template we write, never asked of the model: a requirement that depends
# on a model remembering it is a requirement that holds most of the time.
OUT_OF_MESSAGES_MARK = "\u261d\U0001f3fb"


def _ms(value: Any) -> int:
    """`createdAt` as epoch milliseconds, which is what `ChatMessage` declares."""
    return int(value.timestamp() * 1000) if hasattr(value, "timestamp") else 0


def _loads(value: Any) -> Any:
    """psycopg gives jsonb back parsed; a text column or a fake store may not."""
    return json.loads(value) if isinstance(value, str) else value


def message_out(row: dict[str, Any]) -> dict[str, Any]:
    """One stored turn, in the shape `src/api/types.ts` declares.

    `cards` is read back as stored rather than re-resolved against today's
    results. Re-resolving would delete a card from history the day its posting
    expires — quietly rewriting what Hud said months later, with nobody able to
    tell. The snapshot is honest because every card carries its own
    `source.retrievedAt`, so an old turn reads as "as at that date" rather than
    claiming to be current.
    """
    cards = _loads(row.get("cards")) or {}
    suggestions = _loads(row.get("suggestions"))
    attachments = _loads(row.get("attachments"))

    out: dict[str, Any] = {
        "id": row["message_id"],
        # The UI's vocabulary, not the table's: the row says 'assistant' because
        # that is what it is, the screen says 'hud' because that is who speaks.
        "role": "hud" if row.get("role") == "assistant" else "user",
        "text": row.get("content") or "",
        "createdAt": _ms(row.get("created_at")),
    }
    if cards.get("jobs"):
        out["jobs"] = cards["jobs"]
    if cards.get("courses"):
        out["courses"] = cards["courses"]
    if suggestions:
        out["suggestions"] = suggestions
    if attachments:
        out["attachments"] = attachments
    return out


async def read_ask(request: Request) -> tuple[str, Optional[str], list[dict[str, Any]], int]:
    """The question, the thread, and any attachments — as METADATA.

    Both encodings, because the client sends JSON normally and multipart only
    when a file rides along.

    **The bytes are read to be measured and then dropped.** Nothing here stores a
    file and nothing here creates a document: `POST /api/documents` is the only
    way into the pipeline, and it exists precisely because it has a human
    confirmation screen in the middle of it. A transcript dropped into a
    conversation that silently became the document the matching ran on would
    bypass the product's first trust moment.
    """
    content_type = request.headers.get("content-type", "")
    attachments: list[dict[str, Any]] = []
    rejected = 0

    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        question = str(form.get("question") or "").strip()
        raw_thread = form.get("threadId") or None
        for item in form.getlist("files")[:MAX_ATTACHMENTS]:
            if not hasattr(item, "read"):
                continue
            payload = await item.read()
            size = len(payload)
            del payload                      # measured, never stored
            if size > MAX_ATTACHMENT_BYTES:
                rejected += 1
                continue
            attachments.append({
                "id": f"att_{len(attachments) + 1}",
                "fileName": getattr(item, "filename", None) or "file",
                "mimeType": getattr(item, "content_type", None) or "application/octet-stream",
                "sizeBytes": size,
            })
    else:
        body = await request.json()
        question = str(body.get("question") or "").strip()
        raw_thread = body.get("threadId") or None

    return question, (str(raw_thread) if raw_thread else None), attachments, rejected


def question_with_attachments(question: str, attachments: list[dict[str, Any]]) -> str:
    """Tell the model a file arrived, and nothing more.

    The NAME only — never the contents, which are never read. Hud needs to know
    something was attached so he can point the person at the upload screen rather
    than ignore them; he must not be able to answer questions about what is
    inside it, because nothing in this path has opened it.
    """
    if not attachments:
        return question
    names = ", ".join(a["fileName"] for a in attachments)
    note = (f"[The person attached a file: {names}. You cannot read attachments. "
            f"If they want it analysed, tell them to upload it on the documents "
            f"screen, where they can confirm what was read.]")
    return f"{question}\n\n{note}" if question else note


def register(app: Any, *, require_user, assistant) -> None:
    """Mount the chat routes.

    `assistant` is `api.assistant`, passed rather than imported so the two share
    one quota accountant and one fact-sheet builder instead of growing a second
    copy that could disagree about what a message costs.
    """

    def _store():
        return app.state.store

    def _config() -> Config:
        return app.state.config

    @app.get("/api/chat/threads")
    async def list_threads(request: Request) -> Any:
        user = require_user(request)
        rows = _store().threads(user["user_id"])
        # An empty list is a normal answer on a new account, not an error.
        return [{"id": r["thread_id"], "title": r["title"],
                 "updatedAt": _ms(r["updated_at"])} for r in rows]

    @app.get("/api/chat/threads/{thread_id}")
    async def read_thread(request: Request, thread_id: str) -> Any:
        user = require_user(request)
        row = _store().thread(thread_id, user["user_id"])
        if row is None:
            # 404 and not 403 — for someone else's thread as much as for a
            # missing one. Confirming that another account's thread EXISTS is
            # itself a disclosure, so the two must be indistinguishable.
            return JSONResponse({"error": "not_found"}, status_code=404)
        messages = _store().thread_messages(thread_id, user["user_id"])
        return {"id": row["thread_id"], "title": row["title"],
                "updatedAt": _ms(row["updated_at"]),
                "messages": [message_out(m) for m in messages]}

    @app.post("/api/chat/rate")
    async def rate(request: Request) -> Any:
        user = require_user(request)
        body = await request.json()
        verdict = body.get("verdict")
        if verdict not in ("up", "down"):
            return JSONResponse({"error": "invalid_verdict"}, status_code=400)
        _store().rate_message(message_id=str(body.get("messageId") or ""),
                              user_id=user["user_id"], verdict=verdict)
        # 204 whether or not a row matched. The client never waits on this and
        # never shows a failure: a rating that misses is not worth an error in
        # front of someone who was only being helpful.
        return Response(status_code=204)

    @app.post("/api/chat/ask")
    async def ask(request: Request) -> Any:
        user = require_user(request)
        store, config = _store(), _config()
        question, thread_id, attachments, rejected = await read_ask(request)

        if not question and not attachments:
            return JSONResponse({"error": "empty_message"}, status_code=400)
        if len(question) > MAX_QUESTION_CHARS:
            # Bounded before the quota is claimed: a message too long to send is
            # not a message anyone should be charged for.
            return JSONResponse({"error": "message_too_long"}, status_code=400)

        # An unknown or someone else's thread id starts a NEW thread rather than
        # erroring. The id came from a client that may be holding a stale route,
        # and losing the question to a 404 is the worse answer.
        thread = store.thread(thread_id, user["user_id"]) if thread_id else None
        if thread is None:
            thread = store.create_thread(user_id=user["user_id"],
                                         title=(question or "Attachment")[:TITLE_CHARS])
        thread_id = thread["thread_id"]

        period = assistant.day_start(config)
        used = store.claim_quota(user["user_id"], kind="message",
                                 limit=config.assistant_daily_messages,
                                 period_start=period)

        # Stored before the answer exists, so the question survives whatever
        # happens next — including the limit branch below, where the whole point
        # is that the person can still see what they asked.
        store.add_assistant_message(
            user_id=user["user_id"], role="user", content=question,
            thread_id=thread_id, attachments=attachments or None)

        if used is None:
            # OVER THE LIMIT — answered 200 with a turn from Hud, not 429.
            #
            # `state/chat.tsx` restores the thread on ANY error, so a 429 makes
            # the person's question vanish under a generic "something went
            # wrong" — the worst available reading of "you have used today's
            # messages". As a turn it says the true thing in his voice and the
            # scrollback still shows what they asked. It costs no message, and
            # `answer_source='limit'` keeps it countable rather than letting it
            # pass for a model answer.
            limit = config.assistant_daily_messages
            row = store.add_assistant_message(
                user_id=user["user_id"], role="assistant", thread_id=thread_id,
                answer_source="limit",
                content=(f"That's all {limit} messages for today \u2014 they come back at "
                         f"{assistant.resets_at(config, kind='message')}. Your results "
                         f"are still on your dashboard in the meantime. "
                         f"{OUT_OF_MESSAGES_MARK}"))
            store.touch_thread(thread_id)
            return {"threadId": thread_id, "message": message_out(row)}

        try:
            fact_sheet, has_results, run_id, jobs, courses = assistant.facts_for(
                store, user["user_id"])
            history = store.thread_messages(thread_id, user["user_id"])
            result = assistant.graph_for(app).invoke({
                "question": question_with_attachments(question, attachments),
                "fact_sheet": fact_sheet,
                # THIS thread's turns, not the account's whole history: two
                # conversations about different things must not bleed into each
                # other's context.
                "history": [{"role": h["role"], "content": h["content"]}
                            for h in history[-(config.assistant_history_turns * 2):]],
                "has_results": has_results,
            })
        except Exception:
            # The work did not happen, so the message does not count. Claimed
            # then refunded, rather than checked then incremented, because the
            # latter is the race the guarded UPDATE exists to avoid.
            store.refund_quota(user["user_id"], kind="message", period_start=period)
            raise

        if result.get("model_failed"):
            # Our downtime, not their message.
            store.refund_quota(user["user_id"], kind="message", period_start=period)

        # THE RESOLUTION STEP: handles in, real cards out, invented ones dropped.
        # These are the very dicts `/api/jobs` and `/api/courses` serve, so a
        # posting in a chat turn and the same posting on the Jobs screen cannot
        # drift apart.
        cards = {
            "jobs": resolve_refs(result.get("job_refs"), jobs, kind="J"),
            "courses": resolve_refs(result.get("course_refs"), courses, kind="C"),
        }

        row = store.add_assistant_message(
            user_id=user["user_id"], role="assistant",
            content=result.get("answer") or "", run_id=run_id, thread_id=thread_id,
            answer_source=result.get("answer_source") or "template",
            cards=cards if (cards["jobs"] or cards["courses"]) else None,
            suggestions=result.get("suggestions") or None)
        store.touch_thread(thread_id)

        out = message_out(row)
        rerun_quota = assistant.quota_state(store, config, user["user_id"], "rerun")
        # A SUGGESTION, and only when there is genuinely a credit to spend —
        # proposing something the user cannot do is worse than not proposing it.
        # Nothing here spends anything: only POST /api/assistant/rerun with an
        # explicit confirmation does, which is what keeps a persuasive sentence
        # from costing someone their week.
        if result.get("proposed_rerun") and rerun_quota["remaining"] > 0 and has_results:
            out["proposedRerun"] = {"reason": result.get("rerun_reason"),
                                    "credits": rerun_quota}
        if rejected:
            out["attachmentsRejected"] = rejected
        return {"threadId": thread_id, "message": out}
