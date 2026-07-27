"""Pure-Python grounding: does this extracted string actually appear in the source?

No LLM here, deliberately. This is the load-bearing anti-hallucination check, and
a check implemented by the same class of system that produces the errors is not
much of a check. Everything here is deterministic and unit-testable offline.

Outcomes per field, driven by ``Config`` thresholds:

    score >= grounded_threshold      -> accepted   (method "exact" / "fuzzy")
    adjudicate_threshold .. below    -> escalated to the LLM adjudicator
    method == "too_short"            -> escalated to the LLM adjudicator
    score <  adjudicate_threshold    -> dropped, and recorded in dropped_fields

The middle band is intentionally narrow. Most fields resolve deterministically;
only the genuinely ambiguous handful cost a model call.

Two properties here were bought the hard way and must not be "simplified" back:

* **matching is token-aligned, never a bare substring.** A plain ``needle in
  haystack`` handed an invented value a perfect score whenever its letters sat
  inside a longer real word — measured, against a CV that said only *JavaScript*:
  ``Java`` 1.0, ``Script`` 1.0, and a fabricated grade ``A`` 1.0 against any text.
* **short values are adjudicated, not deleted.** They used to score a flat 0.0 and
  vanish, so an OCR misread of ``SQL`` as ``SQI`` was destroyed unseen.

Lives in ``shared/`` because it is genuinely cross-agent and entirely
domain-neutral — it takes an arbitrary nested dict and an arbitrary source
string, and knows nothing about CVs. Agent A grounds extracted fields against a
CV; Agent B reuses ``normalize()`` for content hashing and ``verify_quote()`` as
the backstop that stops its legitimacy classifier self-certifying. The coupling
rule forbids Agent B importing Agent A's internals, so anything shared has to
live here.
"""

from __future__ import annotations

import difflib
import re
from typing import Any, Iterator

from .text_norm import normalize_ar

# Keys whose values are model-assigned labels rather than quotations from the
# document. Grounding them would be a category error — there is nothing to match.
UNGROUNDABLE_KEYS = {"category", "source_span", "summary_statement"}

# Below this length, fuzzy matching is meaningless (a 2-char grade will fuzzy-
# match almost anywhere). These get an exact-substring test only.
MIN_FUZZY_LEN = 4

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s@.+\-/]")
# The marks normalize() keeps. They must also count as token boundaries — see
# contains_token_span.
_BOUNDARY_PUNCT = "@.+-/"


def normalize(text: str) -> str:
    """Fold script variation away, casefold, strip most punctuation, collapse
    whitespace.

    Kept deliberately gentle on ASCII: ``@ . + - /`` survive because they carry
    meaning in emails, phone numbers, GPAs and date ranges — the exact fields most
    likely to be hallucinated.

    Arabic folding (``shared.text_norm``) runs first and is not cosmetic. Measured
    before it was added: the extracted name ``احمد`` scored 0.6 against a source
    reading ``أحمد`` — the same name, one written with a hamza — which is below the
    adjudication floor, so a real candidate's real name was deleted as a
    hallucination. Arabic-Indic digits behaved the same way (``٣.٧١`` vs ``3.71``
    scored 0.25). On a bilingual Omani CV that is the common case, not an edge one.
    """
    text = normalize_ar(text)
    text = _PUNCT.sub(" ", text.casefold())
    return _WS.sub(" ", text).strip()


def contains_token_span(needle: str, haystack: str) -> bool:
    """Does ``needle`` occur in ``haystack`` on whole-token boundaries?

    Both arguments must already be normalized. This replaced a bare ``needle in
    haystack``, which was the single largest hole in the anti-hallucination layer:
    because ``normalize`` collapses everything to space-separated tokens, a plain
    substring test scored an invented value 1.0 "exact" whenever its letters
    happened to appear inside a longer real word. Measured against a CV that said
    only *JavaScript*: ``Java`` -> 1.0, ``Script`` -> 1.0; against any text at all,
    a fabricated grade ``A`` -> 1.0.

    Requiring a token boundary on each side means ``java`` no longer matches
    ``javascript``, while a genuine multi-word value still matches inside a longer
    phrase (``bank muscat`` in ``bank muscat head office``). A value whose tokens
    really are joined in the source (OCR running two words together) falls through
    to fuzzy scoring, which is what fuzzy is for.

    The boundary is whitespace *or* one of the punctuation marks ``normalize``
    deliberately keeps (``@ . + - /``). Whitespace alone was not enough: those
    marks survive normalization precisely because they carry meaning in emails,
    GPAs and dates, so a value ending a sentence ("...at Bank Muscat.") kept its
    trailing period and failed a whitespace-only test — demoting real data to
    fuzzy right at the accept threshold. Keeping them *inside* the needle still
    discriminates: ``3.1`` does not match ``3.10``, because ``0`` is not a boundary.
    """
    if not needle or not haystack:
        return False
    edge = rf"[\s{re.escape(_BOUNDARY_PUNCT)}]"
    pattern = rf"(?:^|(?<={edge})){re.escape(needle)}(?:$|(?={edge}))"
    return re.search(pattern, haystack) is not None


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
    # Token-aligned, NOT a bare substring: a plain `needle in haystack` here would
    # re-open the hole ground_value just closed (it would hand "java" a 1.0 off
    # "javascript" after the exact test had correctly refused it).
    if contains_token_span(needle, haystack):
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
    """Score one value. Returns ``(score, method)``.

    ``method`` may be ``"too_short"``: a value below ``MIN_FUZZY_LEN`` that is not
    present as a whole token. Fuzzy scoring is meaningless at that length, but
    "absent" is the wrong conclusion — an OCR misread of ``SQL`` as ``SQI`` used to
    score a flat 0.0 and be deleted without ever being looked at. The caller routes
    these to adjudication instead, so short values stop being silently binary.
    """
    needle = normalize(value)
    if not needle:
        return 0.0, "dropped"
    if contains_token_span(needle, normalized_source):
        return 1.0, "exact"
    if len(needle) < MIN_FUZZY_LEN:
        return 0.0, "too_short"
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

        if method == "too_short":
            # Not "absent" — "too short for string maths to have an opinion".
            # Grades, GPAs, SQL/C++/MBA and most Arabic given names live here. The
            # adjudicator can read the surrounding context and settle it; deleting
            # them unseen (the old behaviour) lost real data every run.
            report[field_path] = {
                "grounded": False,
                "score": 0.0,
                "method": "too_short",
                "value": value,
            }
            needs_llm.append({"field_path": field_path, "value": value})
        elif score >= grounded_threshold:
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


def verify_quote(
    evidence_quote: str | None,
    source_text: str,
    value: str | None = None,
    *,
    support_threshold: float = 0.75,
) -> bool:
    """Is the adjudicator's cited quote real — and does it support the value?

    This is the backstop that stops the LLM self-certifying: it must produce a
    verbatim span, and we check that span against the source ourselves. A
    fabricated quote fails here and the field is treated as ungrounded.

    Quoting real text was never sufficient on its own, though. "The quote exists"
    and "the quote supports this value" are different claims, and only the first
    was being checked — so a verdict could certify a fabricated employer by citing
    any true sentence from the document, which is precisely the self-certification
    the check exists to prevent. Pass ``value`` and the quote must also contain it
    (token-aligned, or fuzzily at ``support_threshold`` to tolerate OCR noise).

    ``value`` is optional so existing two-argument callers — Agent B's legitimacy
    adjudicator — keep their current semantics.
    """
    if not evidence_quote or not evidence_quote.strip():
        return False
    normalized_quote = normalize(evidence_quote)
    if normalized_quote not in normalize(source_text):
        return False
    if value is None:
        return True
    score, _ = ground_value(value, normalized_quote)
    return score >= support_threshold
