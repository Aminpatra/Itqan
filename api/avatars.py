"""Profile photos: validate, store on disk, serve back.

Kept out of `main.py` because the interesting part is not the routing, it is the
validation — and validation that lives in a route handler is validation nobody
reads.

**Type and size are enforced HERE, server side.** The UI rejects non-images and
anything over 5 MB before uploading, but BACKEND.md is explicit that this is "a
courtesy, not a control": a client check is a convenience for honest users and no
obstacle at all to anyone else.

**Sniffed, not trusted.** The declared `Content-Type` and the file extension are
both attacker-controlled, so the format is decided by the file's own magic bytes.
A `.png` that is really a PDF is refused, and — more to the point — an `.html`
renamed `.png` cannot be stored and later served back from our own origin, which
is how an avatar upload becomes stored XSS.

Files live under `ITQAN_OUTPUT_DIR`, the named volume that already holds CV
uploads and already survives `docker compose up -d --build`. The bytes are not in
Postgres: that database is 1.7 GB and its connection is what the HNSW queries
use.
"""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Optional

from shared.config import Config

# 5 MB, matching what the UI enforces so the two cannot disagree about what is
# acceptable — a file the client accepts and the server rejects is a confusing
# failure, and the reverse is a hole.
MAX_AVATAR_BYTES = 5 * 1024 * 1024

# Magic bytes -> extension. Deliberately a small allowlist of raster formats a
# browser will render inertly.
#
# SVG IS EXCLUDED ON PURPOSE and its absence is the security decision in this
# file: SVG is an XML document that may carry <script>, so serving one from our
# own origin is stored XSS with extra steps. "It is an image" is true of SVG and
# irrelevant.
_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", "jpg"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
)
_CONTENT_TYPES = {"jpg": "image/jpeg", "png": "image/png",
                  "gif": "image/gif", "webp": "image/webp"}


class AvatarRejected(Exception):
    """Why the file was refused, in words a caller can hand to the user."""


def _sniff(data: bytes) -> Optional[str]:
    """The real format, from the bytes. WEBP is checked separately because its
    signature is split: 'RIFF' then a four-byte size then 'WEBP'."""
    for magic, ext in _SIGNATURES:
        if data.startswith(magic):
            return ext
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def avatar_dir(config: Optional[Config] = None) -> Path:
    path = Path((config or Config()).output_dir) / "avatars"
    path.mkdir(parents=True, exist_ok=True)
    return path


def store_avatar(user_id: str, data: bytes, *, config: Optional[Config] = None) -> str:
    """Write the image and return its path. Raises `AvatarRejected` if unusable.

    A random filename rather than the user id: the path is served over HTTP, and
    a guessable one would let anyone enumerate whose photos exist by requesting
    ids. It also means a replacement never collides with a cached copy of the old
    one.
    """
    if not data:
        raise AvatarRejected("the file is empty")
    if len(data) > MAX_AVATAR_BYTES:
        raise AvatarRejected(
            f"images must be {MAX_AVATAR_BYTES // (1024 * 1024)} MB or smaller")

    ext = _sniff(data)
    if ext is None:
        raise AvatarRejected("that is not an image we can display (JPEG, PNG, GIF or WEBP)")

    target = avatar_dir(config) / f"{user_id}-{secrets.token_hex(8)}.{ext}"
    target.write_bytes(data)
    return str(target)


def remove_avatar(path: Optional[str]) -> None:
    """Delete the file if it is still there. Missing is success: the row is the
    record of whether a user has a photo, and a delete that fails because the
    file already went is not a problem to report to anyone."""
    if not path:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


def content_type(path: str) -> str:
    return _CONTENT_TYPES.get(Path(path).suffix.lstrip(".").lower(),
                              "application/octet-stream")


def is_within_avatar_dir(path: str, *, config: Optional[Config] = None) -> bool:
    """Is this stored path actually one of ours?

    The path comes from our own database, so this is belt and braces — but it is
    the check that stops a corrupted or hand-edited `avatar_path` turning the
    serve route into "read any file the process can reach".
    """
    try:
        return Path(path).resolve().parent == avatar_dir(config).resolve()
    except OSError:
        return False
