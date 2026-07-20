"""Artifact writing.

PaddleOCR hands back numpy arrays for polygons and numpy float32 for scores.
``json.dump`` raises on both under numpy 2.x — there is no implicit coercion.
``to_jsonable`` is the single choke point where that gets fixed, so no caller
has to remember.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def to_jsonable(value: Any) -> Any:
    """Recursively convert numpy scalars/arrays into plain Python."""
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return to_jsonable(value.tolist())
    if hasattr(value, "item") and value.__class__.__module__ == "numpy":
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any, *, indent: int = 2) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(payload), indent=indent, ensure_ascii=False),
        encoding="utf-8",
    )
    return str(path)


def new_run_id() -> str:
    """Fresh id per run.

    Doubles as the LangGraph ``thread_id``. It must be unique per run: reusing a
    thread_id resumes the *previous* run's checkpoint instead of starting clean.
    """
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"


def run_dir(output_dir: Path, run_id: str) -> Path:
    d = Path(output_dir) / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
