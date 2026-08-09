"""Profile photos: upload, serve, remove — and what must be refused.

The validation tests are the point. BACKEND.md says type and size are checked in
the browser as "a courtesy, not a control", so everything here goes through the
API directly, the way anything that is not the UI would.

`test_a_disguised_file_is_refused_by_its_bytes` is the one that matters. An
upload endpoint that stores what the client called an image, and serves it back
from our own origin, is stored XSS — so the format is decided by the file's magic
bytes, never by its extension or its declared Content-Type.
"""

from __future__ import annotations

from pathlib import Path

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
JPG = b"\xff\xd8\xff\xe0" + b"\x00" * 64
GIF = b"GIF89a" + b"\x00" * 64
WEBP = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 64

CONFIRMED = {"fullName": "Maryam Al Balushi", "skills": [], "preferences": {}}


def _upload(client, data: bytes, name: str = "me.png", mime: str = "image/png"):
    return client.post("/api/profile/avatar", files={"file": (name, data, mime)})


# ---------------------------------------------------------------------------
# the happy path
# ---------------------------------------------------------------------------
def test_uploading_returns_a_url_the_server_chose(signed_in):
    """Return the URL, never accept one — the server owns storage, so a client
    can never point this at a path of its choosing."""
    res = _upload(signed_in, PNG)

    assert res.status_code == 200
    assert res.json()["avatarUrl"].startswith("/api/profile/avatar/")


def test_the_photo_is_served_back(signed_in):
    url = _upload(signed_in, PNG).json()["avatarUrl"]

    got = signed_in.get(url)
    assert got.status_code == 200
    assert got.headers["content-type"] == "image/png"
    assert got.content == PNG


def test_it_appears_on_the_profile(signed_in):
    signed_in.post("/api/profile", json=CONFIRMED)
    assert signed_in.get("/api/profile").json()["avatarUrl"] is None

    _upload(signed_in, JPG)
    assert signed_in.get("/api/profile").json()["avatarUrl"] is not None


def test_removing_it_goes_back_to_nothing(signed_in):
    signed_in.post("/api/profile", json=CONFIRMED)
    _upload(signed_in, PNG)

    assert signed_in.delete("/api/profile/avatar").status_code == 204
    assert signed_in.get("/api/profile").json()["avatarUrl"] is None


def test_replacing_a_photo_deletes_the_old_file(signed_in, store):
    """Otherwise every change leaves a copy behind and the uploads volume grows
    by one image per edit, forever."""
    url = _upload(signed_in, PNG).json()["avatarUrl"]
    old_path = Path(store.user_by_id(_user_id(signed_in))["avatar_path"])
    assert old_path.exists()

    _upload(signed_in, JPG, name="new.jpg", mime="image/jpeg")

    assert not old_path.exists(), "the replaced image was left on disk"
    # The URL is per-user and stable, so it now serves the new image.
    assert signed_in.get(url).content == JPG


# ---------------------------------------------------------------------------
# what must be refused — server side, not by the browser
# ---------------------------------------------------------------------------
def test_a_disguised_file_is_refused_by_its_bytes(signed_in):
    """THE security test.

    The extension and the Content-Type are both chosen by whoever is uploading.
    An HTML document named `me.png` and declared `image/png` would, if stored and
    served from our own origin, be stored XSS — so the decision is made from the
    file's own magic bytes.
    """
    html = b"<html><script>alert(document.cookie)</script></html>"
    res = _upload(signed_in, html, name="me.png", mime="image/png")

    assert res.status_code == 400
    assert "image" in res.json()["error"].lower()


def test_svg_is_refused_even_though_it_is_an_image(signed_in):
    """SVG is an XML document that may carry <script>. "It is an image" is true
    and irrelevant; serving one from our origin is the same hole as the test
    above wearing a better disguise."""
    svg = b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>'
    assert _upload(signed_in, svg, name="me.svg", mime="image/svg+xml").status_code == 400


def test_a_file_over_the_limit_is_refused(signed_in):
    from api.avatars import MAX_AVATAR_BYTES

    too_big = PNG + b"\x00" * MAX_AVATAR_BYTES
    res = _upload(signed_in, too_big)

    assert res.status_code == 400
    assert "MB" in res.json()["error"]


def test_an_empty_file_is_refused(signed_in):
    assert _upload(signed_in, b"").status_code == 400


def test_a_rejected_upload_leaves_the_existing_photo_alone(signed_in):
    """A failed replacement must not cost someone the picture they already had.

    The old file is deleted only AFTER the new one is safely written, so a
    refusal is a no-op rather than a removal.
    """
    _upload(signed_in, PNG)
    url = f"/api/profile/avatar/{_user_id(signed_in)}"

    assert _upload(signed_in, b"not an image at all", name="x.png").status_code == 400

    still_there = signed_in.get(url)
    assert still_there.status_code == 200 and still_there.content == PNG


def _user_id(client) -> str:
    return client.get("/api/session").json()["user"]["id"]


# ---------------------------------------------------------------------------
# auth and lifecycle
# ---------------------------------------------------------------------------
def test_uploading_requires_a_session(client):
    assert _upload(client, PNG).status_code == 401


def test_a_user_with_no_photo_serves_404(signed_in):
    assert signed_in.get("/api/profile/avatar/" + _user_id(signed_in)).status_code == 404


def test_an_unknown_user_serves_404(signed_in):
    assert signed_in.get("/api/profile/avatar/nobody-at-all").status_code == 404


def test_a_recorded_photo_missing_from_disk_is_404_not_a_crash(signed_in, store):
    """A restored database, or a hand-cleaned volume. The row says there is a
    photo and there is not — which is a 404, not a stack trace."""
    _upload(signed_in, PNG)
    user_id = _user_id(signed_in)
    path = store.user_by_id(user_id)["avatar_path"]
    Path(path).unlink()

    assert signed_in.get(f"/api/profile/avatar/{user_id}").status_code == 404


def test_a_profile_save_cannot_blank_the_photo(signed_in):
    """Contract, not preference: avatars are off `PUT /api/profile` so that
    correcting a graduation date can never remove someone's picture."""
    signed_in.post("/api/profile", json=CONFIRMED)
    _upload(signed_in, PNG)

    signed_in.put("/api/profile", json={**CONFIRMED, "fullName": "Maryam A."})

    assert signed_in.get("/api/profile").json()["avatarUrl"] is not None


def test_the_stored_path_is_confined_to_the_avatar_directory(signed_in, store):
    """Belt and braces: the path comes from our own database, but a corrupted or
    hand-edited row must not turn the serve route into "read any file"."""
    _upload(signed_in, PNG)
    user_id = _user_id(signed_in)
    store.set_avatar_path(user_id, "/etc/passwd")

    assert signed_in.get(f"/api/profile/avatar/{user_id}").status_code == 404
