"""Script-aware text folding, shared by every agent that compares two strings.

Arabic is written with optional diacritics, a tatweel stretching character, and
several visually identical forms of the same letter (أ إ آ ٱ ا, ى/ي, ة/ه). Two
people typing the same name — or an OCR engine reading it — routinely produce
different code points for identical text. Comparing raw characters therefore
fails on exactly the data this project exists to serve: an Omani CV, in Arabic.

Measured before this module existed: the extracted name ``احمد`` scored **0.6**
against a source containing ``أحمد`` — below the adjudication floor, so a real
candidate's real name was deleted as a hallucination. Arabic-Indic digits were
worse: ``3.71`` against ``٣.٧١`` scored **0.25**.

Promoted here from ``agents/agent_b_job_ingest/legitimacy.py`` when Agent A's
grounder became a second consumer — the same rule the scraping layer followed
(``shared/scraping/``): promote on the second consumer rather than duplicate. The
digit folding is new; everything else is byte-for-byte the logic Agent B's scam
filter has been using.
"""

from __future__ import annotations

import re
import unicodedata

# Harakat, tanwin, superscript alef and the tatweel (ـ) stretching character.
# All are presentational: removing them never changes which word was written.
_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
_ALEF = re.compile(r"[آأإٱ]")
_WS = re.compile(r"\s+")

# Arabic-Indic (٠-٩) and Extended/Persian Arabic-Indic (۰-۹) digits. A GPA or a
# date written in either form must compare equal to its ASCII spelling.
_ARABIC_INDIC_DIGITS = {
    **{chr(0x0660 + i): str(i) for i in range(10)},
    **{chr(0x06F0 + i): str(i) for i in range(10)},
}
_DIGIT_TABLE = str.maketrans(_ARABIC_INDIC_DIGITS)


def fold_digits(text: str) -> str:
    """Arabic-Indic and Persian digits to ASCII. ``٣.٧١`` -> ``3.71``."""
    return (text or "").translate(_DIGIT_TABLE)


def normalize_ar(text: str) -> str:
    """Fold Arabic orthographic variation away, leaving other scripts intact.

    NFKC first (so presentation forms and ligatures collapse to their canonical
    letters), then diacritics/tatweel out, then the letter forms that carry no
    distinction for matching: آأإٱ -> ا, ى -> ي, ة -> ه. Digits are folded so a
    number reads the same in either script.
    """
    text = unicodedata.normalize("NFKC", text or "")
    text = _DIACRITICS.sub("", text)
    text = _ALEF.sub("ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = fold_digits(text)
    return _WS.sub(" ", text).strip()


# Languages written right-to-left. Used to decide reading order when a page is
# reassembled from positioned OCR blocks — sorting an Arabic line by ascending
# x-coordinate reverses its words.
RTL_LANGUAGES = frozenset({"ar", "fa", "ur", "he", "arabic", "persian", "urdu", "hebrew"})


def is_rtl(lang: str | None) -> bool:
    return (lang or "").strip().casefold() in RTL_LANGUAGES
