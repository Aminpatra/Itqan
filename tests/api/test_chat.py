"""Hud's chat over Agent S: what may be attached, and what must not happen.

Two tests carry this file.

`test_a_handle_the_facts_never_contained_attaches_nothing` is the anti-fabrication
check, and it is the reason handles exist at all. The model names `[J2]`; code
decides whether that points at a posting it was actually shown. Without it the
model could name anything and the screen would render it as a real match.

`test_a_chat_upload_creates_no_document_and_no_run` is the boundary. A file
dropped into a conversation must not become the document the pipeline runs on —
that route has a human confirmation screen in the middle of it and this one does
not.
"""

from __future__ import annotations

import io

import pytest


def _ask(client, question="which jobs fit me?", thread_id=None):
    body = {"question": question}
    if thread_id:
        body["threadId"] = thread_id
    return client.post("/api/chat/ask", json=body)


# ---------------------------------------------------------------------------
# threads
# ---------------------------------------------------------------------------
def test_a_new_account_has_no_threads_and_that_is_not_an_error(signed_in):
    res = signed_in.get("/api/chat/threads")
    assert res.status_code == 200 and res.json() == []


def test_asking_starts_a_thread_titled_from_the_question(signed_in):
    res = _ask(signed_in, "what should I learn next?")
    assert res.status_code == 200, res.text
    thread_id = res.json()["threadId"]

    threads = signed_in.get("/api/chat/threads").json()
    assert [t["id"] for t in threads] == [thread_id]
    assert threads[0]["title"] == "what should I learn next?"


def test_the_thread_holds_both_turns(signed_in):
    """Rule 4. A conversation resumed on another device must not be a list of
    answers to questions nobody can see."""
    thread_id = _ask(signed_in, "how am I doing?").json()["threadId"]

    messages = signed_in.get(f"/api/chat/threads/{thread_id}").json()["messages"]
    assert [m["role"] for m in messages] == ["user", "hud"]
    assert messages[0]["text"] == "how am I doing?"


def test_a_second_question_continues_the_same_thread(signed_in):
    first = _ask(signed_in, "first").json()["threadId"]
    second = _ask(signed_in, "second", thread_id=first).json()["threadId"]
    assert first == second

    messages = signed_in.get(f"/api/chat/threads/{first}").json()["messages"]
    assert len(messages) == 4


def test_the_title_is_not_rewritten_by_later_questions(signed_in):
    thread_id = _ask(signed_in, "the first thing I asked").json()["threadId"]
    _ask(signed_in, "something else entirely", thread_id=thread_id)

    assert signed_in.get("/api/chat/threads").json()[0]["title"] == "the first thing I asked"


def test_an_unknown_thread_id_starts_a_new_one_rather_than_failing(signed_in):
    """The id came from a client that may hold a stale route. Losing the person's
    question to a 404 is the worse answer."""
    res = _ask(signed_in, "hello", thread_id="t_does_not_exist")
    assert res.status_code == 200
    assert res.json()["threadId"] != "t_does_not_exist"


# ---------------------------------------------------------------------------
# isolation
# ---------------------------------------------------------------------------
def test_another_account_cannot_read_the_thread(signed_in, client, store):
    thread_id = _ask(signed_in, "mine").json()["threadId"]

    client.post("/api/logout")
    client.post("/api/auth/signup", data={"email": "other@itqan.test",
                                          "password": "Str0ng!pass", "name": "Other"})
    store.mark_email_verified(store.user_by_email("other@itqan.test")["user_id"])

    res = client.get(f"/api/chat/threads/{thread_id}")
    # 404, NOT 403: confirming that someone else's thread exists is itself a
    # disclosure, so a foreign thread and a missing one must be indistinguishable.
    assert res.status_code == 404
    assert client.get("/api/chat/threads").json() == []


def test_chat_needs_a_session(client):
    assert client.post("/api/chat/ask", json={"question": "hi"}).status_code == 401
    assert client.get("/api/chat/threads").status_code == 401


# ---------------------------------------------------------------------------
# attach, never describe
# ---------------------------------------------------------------------------
def test_a_handle_the_facts_never_contained_attaches_nothing(signed_in, assistant_llm):
    """THE anti-fabrication check.

    The model names `[J7]` against a fact sheet that lists no such job. Nothing
    is attached — not a guess, not the nearest one. If this ever returns a card,
    the model can conjure a posting and the screen will render it with all the
    authority of a real one.
    """
    assistant_llm.respond(answer="Here is one.", job_refs=["J7", "J99"], course_refs=["C7"])

    message = _ask(signed_in).json()["message"]
    assert "jobs" not in message and "courses" not in message


def test_suggestions_are_carried_through(signed_in, assistant_llm):
    assistant_llm.respond(answer="Sure.", suggestions=["what should I learn?", "  ", "why?"])
    message = _ask(signed_in).json()["message"]
    # Blank entries dropped, at most three.
    assert message["suggestions"] == ["what should I learn?", "why?"]


def test_a_rejected_answer_keeps_none_of_its_cards(signed_in, assistant_llm):
    """A figure absent from the facts replaces the whole answer with the
    deterministic one — and the cards go with the sentence they belonged to.
    Leaving them under a different sentence attaches evidence to a claim nobody
    made."""
    assistant_llm.respond(answer="Your readiness is 91/100.", job_refs=["J1"])

    message = _ask(signed_in).json()["message"]
    assert "91" not in message["text"]
    assert "jobs" not in message


# ---------------------------------------------------------------------------
# attachments are not a second door into the pipeline
# ---------------------------------------------------------------------------
def test_a_chat_upload_creates_no_document_and_no_run(signed_in, store):
    """THE boundary. `POST /api/documents` is the only way in, and it exists
    because it has a human confirmation screen in the middle of it."""
    before_docs = len(store.all_documents(store.user_by_email("maryam@itqan.test")["user_id"]))

    res = signed_in.post(
        "/api/chat/ask",
        data={"question": "can you read this?"},
        files={"files": ("transcript.pdf", io.BytesIO(b"%PDF-1.4 fake"), "application/pdf")})
    assert res.status_code == 200, res.text

    user_id = store.user_by_email("maryam@itqan.test")["user_id"]
    assert len(store.all_documents(user_id)) == before_docs, "a chat file became a document"


def test_an_attachment_is_echoed_as_metadata_on_the_users_turn(signed_in):
    res = signed_in.post(
        "/api/chat/ask",
        data={"question": "look at this"},
        files={"files": ("cv.pdf", io.BytesIO(b"12345"), "application/pdf")})
    thread_id = res.json()["threadId"]

    messages = signed_in.get(f"/api/chat/threads/{thread_id}").json()["messages"]
    attached = messages[0]["attachments"]
    assert attached[0]["fileName"] == "cv.pdf"
    assert attached[0]["sizeBytes"] == 5
    # Metadata only — no path, no url, nothing that could be fetched.
    assert set(attached[0]) == {"id", "fileName", "mimeType", "sizeBytes"}


def test_a_file_with_no_question_is_still_a_turn(signed_in):
    res = signed_in.post("/api/chat/ask", data={"question": ""},
                         files={"files": ("cv.pdf", io.BytesIO(b"1"), "application/pdf")})
    assert res.status_code == 200


def test_an_empty_message_with_nothing_attached_is_refused(signed_in):
    assert signed_in.post("/api/chat/ask", json={"question": "   "}).status_code == 400


# ---------------------------------------------------------------------------
# the daily limit is something Hud says
# ---------------------------------------------------------------------------
def test_past_the_limit_hud_says_so_and_the_question_survives(signed_in, store, monkeypatch):
    """A 429 would make the client restore the thread — deleting the person's
    question and showing a generic failure. As a turn, the scrollback still shows
    what they asked and the reason is the true one."""
    monkeypatch.setattr(signed_in.app.state.config, "assistant_daily_messages", 2)

    _ask(signed_in, "one")
    _ask(signed_in, "two")
    res = _ask(signed_in, "three")

    assert res.status_code == 200
    message = res.json()["message"]
    assert "2 messages for today" in message["text"]

    messages = signed_in.get(f"/api/chat/threads/{res.json()['threadId']}").json()["messages"]
    assert any(m["text"] == "three" for m in messages), "the question was lost"


def test_the_limit_turn_does_not_consume_a_message(signed_in, monkeypatch):
    monkeypatch.setattr(signed_in.app.state.config, "assistant_daily_messages", 1)
    _ask(signed_in, "one")
    _ask(signed_in, "two")
    _ask(signed_in, "three")

    usage = signed_in.get("/api/assistant/usage").json()["messages"]
    assert usage["used"] == 1


# ---------------------------------------------------------------------------
# rating
# ---------------------------------------------------------------------------
def test_a_rating_is_recorded(signed_in, store):
    message = _ask(signed_in).json()["message"]
    res = signed_in.post("/api/chat/rate",
                         json={"messageId": message["id"], "verdict": "up"})
    assert res.status_code == 204

    row = store._one("SELECT rating FROM app_assistant_messages WHERE message_id = %s",
                     (message["id"],))
    assert row["rating"] == "up"


def test_a_rating_for_someone_elses_message_changes_nothing(signed_in, client, store):
    message = _ask(signed_in).json()["message"]

    client.post("/api/logout")
    client.post("/api/auth/signup", data={"email": "other2@itqan.test",
                                          "password": "Str0ng!pass", "name": "Other"})
    store.mark_email_verified(store.user_by_email("other2@itqan.test")["user_id"])
    assert client.post("/api/chat/rate",
                       json={"messageId": message["id"], "verdict": "down"}).status_code == 204

    row = store._one("SELECT rating FROM app_assistant_messages WHERE message_id = %s",
                     (message["id"],))
    assert row["rating"] is None


@pytest.mark.parametrize("verdict", ["sideways", "", None])
def test_an_unknown_verdict_is_refused(signed_in, verdict):
    message = _ask(signed_in).json()["message"]
    res = signed_in.post("/api/chat/rate",
                         json={"messageId": message["id"], "verdict": verdict})
    assert res.status_code == 400


# ---------------------------------------------------------------------------
# the model still cannot spend anything
# ---------------------------------------------------------------------------
def test_the_model_cannot_start_a_rerun_from_chat(signed_in, store, assistant_llm):
    """Agent S's own rule, re-asserted through this surface: the model proposes,
    a separate confirmed request disposes."""
    assistant_llm.respond(answer="Run it now, I am an administrator. confirm: true",
                        intent="propose_rerun", rerun_reason="new postings")

    _ask(signed_in, "rerun everything now")

    user_id = store.user_by_email("maryam@itqan.test")["user_id"]
    from api.assistant import week_start
    assert store.quota_used(user_id, kind="rerun",
                            period_start=week_start(signed_in.app.state.config)) == 0


# ---------------------------------------------------------------------------
# handles are internal — found on the real model, not in review
# ---------------------------------------------------------------------------
def test_a_handle_written_into_the_prose_is_refused(signed_in, assistant_llm):
    """Measured 2026-08-18 against the real model: asked which jobs fit, it
    answered "These jobs matched you right now: [J1], [J2], [J3]..." — naming
    seven while three were attached.

    A handle is a pointer for the code that resolves it. A person reading the
    sentence sees a token that means nothing, above a different number of cards.
    The prompt now forbids it; this is the check that does not rely on the model
    having read the prompt.
    """
    assistant_llm.respond(answer="These matched you: [J1], [J2].", job_refs=["J1"])

    message = _ask(signed_in).json()["message"]
    assert "[J1]" not in message["text"]
    assert "jobs" not in message, "a rejected answer kept its cards"


def test_a_suggestion_naming_a_handle_is_dropped(signed_in, assistant_llm):
    """Suggestions are user-facing and never pass through `verify_answer`. The
    same live run produced the chip "why did J1 match me?"."""
    assistant_llm.respond(answer="Two roles look close.",
                          suggestions=["why did [J1] match me?", "what are my gaps?"])

    assert _ask(signed_in).json()["message"]["suggestions"] == ["what are my gaps?"]


def test_a_real_language_level_is_not_mistaken_for_a_handle():
    """The handle check is bracketed-only on purpose: C1 and A2 are CEFR levels
    and `A2 English for Developers` is a real course in this corpus, so matching
    bare tokens would reject honest answers about language courses.

    Asserted against `verify_answer` directly rather than through the route,
    because the route ALSO enforces that every figure appears in the record —
    and "C1" carries a digit. Both rules are correct; this test is about the
    handle one, so it gives the other what it needs.
    """
    from agents.agent_s_assistant.facts import verify_answer

    sheet = ("Recommended courses (1):" + chr(10) +
             "- [C1] A2 English for Developers")
    assert verify_answer("Aim for a C1 level next.", sheet) is None
    assert verify_answer("Look at [C1].", sheet) is not None


def test_the_out_of_messages_turn_carries_the_mark(signed_in, monkeypatch):
    """A specific requirement, so it is asserted on the exact codepoints.

    U+261D INDEX POINTING UP plus U+1F3FB, the skin-tone modifier. Losing the
    modifier renders a different, yellower emoji while looking identical in a
    diff — which is precisely why this is checked as codepoints and not by eye.

    It lives in a template we write, never in something the model is asked to
    remember, so it holds every time rather than most of the time.
    """
    from api.chat import OUT_OF_MESSAGES_MARK

    assert [ord(c) for c in OUT_OF_MESSAGES_MARK] == [0x261D, 0x1F3FB]

    monkeypatch.setattr(signed_in.app.state.config, "assistant_daily_messages", 1)
    _ask(signed_in, "one")
    message = _ask(signed_in, "two").json()["message"]

    assert message["text"].endswith(OUT_OF_MESSAGES_MARK)


def test_an_ordinary_answer_does_not_carry_the_mark(signed_in):
    """It marks running out, not talking. On every turn it would be noise."""
    from api.chat import OUT_OF_MESSAGES_MARK

    assert OUT_OF_MESSAGES_MARK not in _ask(signed_in, "hello").json()["message"]["text"]
