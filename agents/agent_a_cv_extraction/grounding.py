"""Pure-Python grounding: does this extracted string actually appear in the source?

No LLM here, deliberately. This is the load-bearing anti-hallucination check, and
a check implemented by the same class of system that produces the errors is not
much of a check. Everything here is deterministic and unit-testable offline.

Three outcomes per field, driven by ``Config`` thresholds:

    score >= grounded_threshold      -> accepted   (method "exact" / "fuzzy")
    adjudicate_threshold .. below    -> escalated to the LLM adjudicator
    score <  adjudicate_threshold    -> dropped, and recorded in dropped_fields

The middle band is intentionally narrow. Most fields resolve deterministically;
only the genuinely ambiguous handful cost a model call.
"""

from __future__ import annotations

import difflib
import re
from typing import Any, Iterator

# Keys whose values are model-assigned labels rather than quotations from the
# document. Grounding them would be a category error — there is nothing to match.
UNGROUNDABLE_KEYS = {"category", "source_span", "summary_statement"}

# Below this length, fuzzy matching is meaningless (a 2-char grade will fuzzy-
# match almost anywhere). These get an exact-substring test only.
MIN_FUZZY_LEN = 4

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s@.+\-/]")


def normalize(text: str) -> str:
    """Casefold, strip most punctuation, collapse whitespace.

    Kept deliberately gentle: ``@ . + - /`` survive because they carry meaning in
    emails, phone numbers, GPAs and date ranges — the exact fields most likely to
    be hallucinated.
    """
    text = _PUNCT.sub(" ", text.casefold())
    return _WS.sub(" ", text).strip()


def iter_leaf_strings(obj: Any, prefix: str = "") -> Iterator[tuple[str, str]]:
    """Walk a nested extraction dict, yielding ``(field_path, value)`` for every
    non-empty string leaf worth checking."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in UNGROUNDABLE_KEYS:
                continue
            yield from iter_leaf_strings(value, f"{prefix}.{key}" if prefix else key)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from iter_leaf_strings(value, f"{prefix}[{i}]")
    elif isinstance(obj, str) and obj.strip():
        yield prefix, obj


def best_window_score(needle: str, haystack: str) -> float:
    """Best similarity of ``needle`` against any same-length window of ``haystack``.

    Both arguments must already be normalized. Returns 0.0..1.0.
    """
    if not needle or not haystack:
        return 0.0
    if needle in haystack:
        return 1.0
    if len(needle) < MIN_FUZZY_LEN:
        return 0.0  # too short to fuzzy-match meaningfully; exact test already failed

    n = len(needle)
    # A little slack on the window so OCR insertions don't truncate the match.
    window_len = n + max(2, n // 5)
    step = max(1, n // 4)

    matcher = difflib.SequenceMatcher(autojunk=False)
    matcher.set_seq2(needle)  # b is cached across set_seq1 calls — keep needle here

    best = 0.0
    for start in range(0, max(1, len(haystack) - n + 1), step):
        matcher.set_seq1(haystack[start : start + window_len])
        score = matcher.ratio()
        if score > best:
            best = score
            if best >= 0.995:
                break
    return best


def ground_value(value: str, normalized_source: str) -> tuple[float, str]:
    """Score one value. Returns ``(score, method)``."""
    needle = normalize(value)
    if not needle:
        return 0.0, "dropped"
    if needle in normalized_source:
        return 1.0, "exact"
    return best_window_score(needle, normalized_source), "fuzzy"


def ground_extraction(
    extraction: dict[str, Any],
    source_text: str,
    *,
    grounded_threshold: float = 0.92,
    adjudicate_threshold: float = 0.75,
    skip_fields: set[str] | None = None,
) -> dict[str, Any]:
    """Score every leaf string in ``extraction`` against ``source_text``.

    ``skip_fields`` are treated as grounded by definition — used for values a
    human typed in, which have no reason to appear in the OCR text.

    Returns::

        {
          "report":     {field_path: {grounded, score, method, value}},
          "needs_llm":  [{field_path, value}, ...],
          "dropped":    [field_path, ...],
        }
    """
    skip_fields = skip_fields or set()
    normalized_source = normalize(source_text)

    report: dict[str, dict[str, Any]] = {}
    needs_llm: list[dict[str, str]] = []
    dropped: list[str] = []

    for field_path, value in iter_leaf_strings(extraction):
        if field_path in skip_fields:
            report[field_path] = {
                "grounded": True,
                "score": 1.0,
                "method": "human",
                "value": value,
            }
            continue

        score, method = ground_value(value, normalized_source)

        if score >= grounded_threshold:
            report[field_path] = {
                "grounded": True,
                "score": round(score, 4),
                "method": method,
                "value": value,
            }
        elif score >= adjudicate_threshold:
            report[field_path] = {
                "grounded": False,
                "score": round(score, 4),
                "method": "fuzzy",
                "value": value,
            }
            needs_llm.append({"field_path": field_path, "value": value})
        else:
            report[field_path] = {
                "grounded": False,
                "score": round(score, 4),
                "method": "dropped",
                "value": value,
            }
            dropped.append(field_path)

    return {"report": report, "needs_llm": needs_llm, "dropped": dropped}


# ---------------------------------------------------------------------------
# Applying results back onto the extraction
# ---------------------------------------------------------------------------
def _split_path(field_path: str) -> list[str | int]:
    parts: list[str | int] = []
    for chunk in field_path.split("."):
        for piece in re.split(r"\[(\d+)\]", chunk):
            if piece == "":
                continue
            parts.append(int(piece) if piece.isdigit() else piece)
    return parts


def prune_field(extraction: dict[str, Any], field_path: str) -> bool:
    """Set an ungrounded field to None in place. Returns True if it was pruned.

    List *elements* are not removed — that would shift every later index and
    invalidate the field paths already recorded in the report. Scalars inside
    them are nulled instead.
    """
    parts = _split_path(field_path)
    if not parts:
        return False

    cursor: Any = extraction
    for part in parts[:-1]:
        try:
            cursor = cursor[part]
        except (KeyError, IndexError, TypeError):
            return False

    last = parts[-1]
    try:
        if isinstance(last, int) and isinstance(cursor, list):
            cursor[last] = None
        else:
            cursor[last] = None
        return True
    except (KeyError, IndexError, TypeError):
        return False


def compact_nulls(value: Any) -> Any:
    """Drop nulls left inside lists by pruning.

    ``prune_field`` nulls list elements rather than removing them, because
    removing would shift every later index and invalidate the field paths already
    recorded in the grounding report. That leaves ``["Python", None]`` behind,
    which is correct internally but must never reach a consumer: it crashes any
    ``", ".join(...)`` and would appear in the published envelope as a bare null
    inside a string array.

    So the nulls survive until the report is final, and are swept here on the way
    out. Empty dicts left by pruning are dropped too.
    """
    if isinstance(value, dict):
        return {k: compact_nulls(v) for k, v in value.items()}
    if isinstance(value, list):
        cleaned = [compact_nulls(v) for v in value]
        return [
            v
            for v in cleaned
            if v is not None and v != {} and not (isinstance(v, str) and not v.strip())
        ]
    return value


def verify_quote(evidence_quote: str | None, source_text: str) -> bool:
    """Is the adjudicator's cited quote real?

    This is the backstop that stops the LLM self-certifying: it must produce a
    verbatim span, and we check that span against the source ourselves. A
    fabricated quote fails here and the field is treated as ungrounded.
    """
    if not evidence_quote or not evidence_quote.strip():
        return False
    return normalize(evidence_quote) in normalize(source_text)
