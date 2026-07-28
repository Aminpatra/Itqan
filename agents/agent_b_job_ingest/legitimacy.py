"""Rule-based legitimacy scoring.

PHASE 2b: **scoring only.** Nothing here gates, rejects or changes a posting's
status yet. The scorer runs so its output distribution can be measured against
real postings before it is given the power to delete anything — the gate metric
is the fraction of postings landing in the 0.40-0.70 risk band, and >20% means
the weights are wrong rather than that more LLM adjudication is needed.

Two design decisions carry most of the weight.

**Noisy-OR, not a sum.** Five independent 0.20 signals summed reach exactly 1.00
— certainty, from five weak hints — which makes any threshold meaningless.
``1 - Π(1-wᵢ)`` gives 0.67 for the same five: "several weak signals, still not
certain", which is the honest reading. It is also bounded below 1.0 by
construction, so no combination of rules can ever assert certainty.

**Every phrase signal carries a verbatim span from the source.** A risk score
whose reasons cannot be quoted back is not auditable, and this filter's false
positives are invisible by construction: a wrongly rejected posting is excluded
from the statistics anyone would use to notice it was wrong. ``assert_auditable``
re-verifies each span against the source text, so a scorer bug that invents
evidence fails loudly rather than producing a plausible number.

Structural signals (an absent employer, no described role) cannot quote a span,
because their evidence is an absence. They carry ``evidence=None`` and a stated
``basis`` instead — the distinction is explicit rather than papered over with a
misleading quote.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Sequence

from shared.config import Config
from shared.text_norm import normalize_ar as _shared_normalize_ar

# ---------------------------------------------------------------------------
# Weights. Ordered by how strongly each independently indicates a scam.
#
# advance_fee is by far the strongest because it is the mechanism of the fraud
# itself, not a correlate of it. The rest are correlates and are weighted to
# say so — no single one of them can reach the reject threshold alone, which is
# deliberate: each has a legitimate explanation in isolation.
# ---------------------------------------------------------------------------
# Rules describe evidence, never proof. Reserving the top sliver keeps that
# distinction expressible: a score of 1.0 would mean the rule set is certain,
# which no phrase list can earn.
MAX_RISK = 0.999999

SIGNAL_WEIGHTS: dict[str, float] = {
    "advance_fee": 0.45,
    "whatsapp_only_contact": 0.30,
    "no_employer_named": 0.25,
    "no_salary_deduction": 0.20,
    # Short AND silent about the role — the classic "WhatsApp me" stub.
    "contact_only_no_role": 0.20,
    # Long enough to have described the work and still didn't. Weaker evidence:
    # plenty of legitimate long postings are pure company boilerplate, whereas a
    # posting too short to say anything at all is the more suspicious shape.
    "no_role_described": 0.10,
    "free_accommodation_bait": 0.15,
    "mass_vacancy_no_detail": 0.10,
}


@dataclass(frozen=True)
class Signal:
    code: str
    weight: float
    # The verbatim source span, when the signal is phrase-based. None for
    # structural signals, whose evidence is an absence and cannot be quoted.
    evidence: str | None
    basis: str

    @property
    def is_structural(self) -> bool:
        return self.evidence is None


@dataclass(frozen=True)
class Assessment:
    risk: float
    signals: tuple[Signal, ...] = ()

    @property
    def score(self) -> float:
        """Stored as ``legitimacy_score``: 1.0 clean, 0.0 certainly a scam.

        Inverted from risk on purpose — the column reads as a quality measure
        alongside every other column, where a high number is a good thing."""
        return round(1.0 - self.risk, 3)

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(s.code for s in self.signals)

    def band(self, config: Config | None = None) -> str:
        """Which of the three paths this posting takes downstream.

        Thresholds live in Config in SCORE space (not risk), matching the column
        and the reject rule, so there is one direction to reason about.
        """
        config = config or Config()
        if self.score <= config.legitimacy_reject_at:
            return "reject"
        if self.score <= config.legitimacy_adjudicate_high:
            return "adjudicate"
        return "clean"


# ---------------------------------------------------------------------------
# Text normalization
#
# Arabic is written with optional diacritics and several visually identical
# forms of the same letter (أ إ آ ا, ى/ي, ة/ه). Matching raw text would make a
# rule silently miss the same phrase typed a different way — and a scam filter
# that fails on an orthographic variant fails exactly where it matters.
#
# The folding itself now lives in shared/text_norm.py: Agent A's grounder became a
# second consumer (an Arabic name was being deleted as a hallucination), and this
# project promotes on the second consumer rather than duplicating — the same move
# shared/scraping/ made. Imported back here so the scam filter and the grounder can
# never drift on what "the same text" means. The regexes below stay because
# _fold_with_map needs them character-by-character to keep spans verbatim.
# ---------------------------------------------------------------------------
_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
_ALEF = re.compile(r"[آأإٱ]")
_WS = re.compile(r"\s+")


def normalize_ar(text: str) -> str:
    """Delegates to the shared folding — see the note above."""
    return _shared_normalize_ar(text)


def _fold(text: str) -> str:
    return normalize_ar(text).casefold()


def _fold_with_map(text: str) -> tuple[str, list[int]]:
    """Fold, while recording which original index each folded character came from.

    Needed because folding is not length-preserving — dropping a diacritic
    shifts every subsequent offset — so a match found in the folded string
    cannot be sliced out of the original by its own offsets. Without the map the
    only options are storing a *normalized* span (not verbatim, so not evidence)
    or storing nothing (a rule that fires and cites nothing). The map keeps the
    quote exact for text written with diacritics.
    """
    chars: list[str] = []
    origin: list[int] = []
    prev_space = False

    for index, char in enumerate(text):
        if _DIACRITICS.match(char):
            continue
        if char.isspace():
            if not prev_space:
                prev_space = True
                chars.append(" ")
                origin.append(index)
            continue
        prev_space = False
        folded = unicodedata.normalize("NFKC", char)
        folded = _ALEF.sub("ا", folded).replace("ى", "ي").replace("ة", "ه").casefold()
        for piece in folded:
            chars.append(piece)
            origin.append(index)

    return "".join(chars), origin


# ---------------------------------------------------------------------------
# Phrase rules. Each entry is (code, pattern). Patterns match against folded
# text; the evidence span is sliced from the ORIGINAL string at the match
# offsets, so what gets stored is what the source actually wrote.
#
# These are detection patterns, not example postings. No file in this repo
# contains a realistic scam advertisement: a convincing fake in a source tree is
# a working template, and the tests are built from single clauses for that
# reason.
# ---------------------------------------------------------------------------
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "advance_fee",
        re.compile(
            r"(?:"
            r"registration\s+fee|processing\s+fee|application\s+fee|security\s+deposit"
            r"|pay(?:ment)?\s+(?:of\s+)?\d+\s*(?:omr|aed|sar|usd|riyal)"
            r"|refundable\s+(?:fee|amount|deposit)"
            r"|رسوم\s*(?:التسجيل|تسجيل|الاشتراك|التقديم)"
            r"|دفع\s*مبلغ|تحويل\s*مبلغ|مقابل\s*مادي"
            r")",
            re.IGNORECASE,
        ),
    ),
    (
        "no_salary_deduction",
        re.compile(
            r"(?:no\s+(?:salary\s+)?deduction(?:s)?\s+from\s+(?:the\s+)?salary"
            r"|بدون\s*خصم\s*من\s*الراتب|لا\s*يوجد\s*خصم\s*من\s*الراتب)",
            re.IGNORECASE,
        ),
    ),
    (
        "free_accommodation_bait",
        re.compile(
            r"(?:free\s+(?:accommodation|housing)\s+and\s+(?:food|meals)"
            r"|سكن\s*(?:و\s*)?(?:اعاشه|اكل|مواصلات)?\s*مجاني)",
            re.IGNORECASE,
        ),
    ),
)

# Contact-channel detection, used by whatsapp_only_contact. Presence of WhatsApp
# is not itself suspicious — it is the region's ordinary business channel. The
# signal fires only when it is the ONLY route to the employer.
_WHATSAPP = re.compile(r"(?:whats\s?app|واتس\s?اب|wa\.me|api\.whatsapp\.com)", re.IGNORECASE)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_APPLY_LINK = re.compile(r"https?://(?!wa\.me|api\.whatsapp\.com)[^\s<>\"]+", re.IGNORECASE)

# Vacancy counts without any accompanying detail.
_MASS_VACANCY = re.compile(
    r"(?:\b\d{2,}\s*(?:vacanc(?:y|ies)|positions?|posts?)\b|\b\d{2,}\s*(?:وظيفه|وظائف|شاغر))",
    re.IGNORECASE,
)

# Words that indicate an actual role is described, in either language. Absence
# of ALL of them is what no_role_described means.
#
# The `(?:ال)?` prefixes are load-bearing, not tidying. Every Arabic term here was
# written with the definite article while every English one is a bare stem, so the
# common indefinite forms — مهام الوظيفة, شروط التقديم, خبرة لا تقل عن — matched
# nothing. Measured on the same posting in three renderings: Arabic without the
# article scored 0.560 and went to the paid adjudicator, while the article-prefixed
# Arabic and the English both scored 0.700 and passed clean. A vacancy was being
# penalised for its orthography, in the project's primary language.
_ROLE_WORDS = re.compile(
    r"(?:responsibilit|requirement|qualification|experience|duties|skills?|role|shift"
    r"|(?:ال)?مهام|(?:ال)?مسؤوليات|(?:ال)?متطلبات|(?:ال)?مؤهلات|(?:ال)?خبره|(?:ال)?شروط"
    r"|(?:ال)?وظيفه|(?:ال)?واجبات)",
    re.IGNORECASE,
)

MIN_ROLE_CHARS = 120


def _find(pattern: re.Pattern[str], text: str) -> str | None:
    """Return the VERBATIM span from ``text`` for a match found in folded text.

    The match runs on the folded string so orthographic variants cannot evade a
    rule, but what gets stored is sliced from the original via the index map —
    so the evidence is always exactly what the source wrote, diacritics and all.
    """
    folded, origin = _fold_with_map(text)
    match = pattern.search(folded)
    if not match or match.end() == match.start():
        return None

    start = origin[match.start()]
    end = origin[match.end() - 1] + 1
    return text[start:end].strip() or None


# ---------------------------------------------------------------------------
def combine_signals(weights: Sequence[float]) -> float:
    """Noisy-OR: ``1 - Π(1-wᵢ)``.

    Monotone non-decreasing in each weight and strictly bounded below 1.0, both
    of which are property-tested. Those two facts are what make the reject
    threshold mean the same thing regardless of how many rules exist, so adding
    a rule later cannot silently move every posting toward rejection.
    """
    product = 1.0
    for weight in weights:
        product *= 1.0 - max(0.0, min(1.0, weight))

    risk = round(1.0 - product, 6)
    # Clamped, not merely bounded in the limit. With enough signals the product
    # underflows and `1 - product` rounds to exactly 1.0 — certainty, arrived at
    # by floating-point accident. The clamp keeps "no set of rules can assert
    # certainty" true of the implementation and not just of the mathematics.
    return min(risk, MAX_RISK)


def score_text(
    text: str,
    *,
    company: str | None = None,
    title: str = "",
    employer_extracted: bool = True,
) -> Assessment:
    """Score one posting from its text. Deterministic; no LLM, no network.

    ``employer_extracted=False`` means extraction has not run yet, so nothing is
    known about whether the posting names an employer. That is NOT the same
    claim as ``company=None``, which means extraction ran and found none — and
    conflating them was a real bug caught at the phase-2b gate: every posting
    scored ``no_employer_named`` purely because the adapters, correctly, do not
    set ``company``. The band distribution then measured phase ordering rather
    than the postings, which would have led to retuning weights against an
    artifact.
    """
    signals: list[Signal] = []
    body = text or ""

    for code, pattern in _PATTERNS:
        evidence = _find(pattern, body)
        if evidence:
            signals.append(
                Signal(
                    code=code,
                    weight=SIGNAL_WEIGHTS[code],
                    evidence=evidence,
                    basis=f"matched the {code} phrase rule",
                )
            )

    # --- contact channels ---
    whatsapp = _find(_WHATSAPP, body)
    if whatsapp:
        has_email = bool(_EMAIL.search(body))
        has_link = bool(_APPLY_LINK.search(body))
        if not has_email and not has_link:
            signals.append(
                Signal(
                    code="whatsapp_only_contact",
                    weight=SIGNAL_WEIGHTS["whatsapp_only_contact"],
                    evidence=whatsapp,
                    basis="WhatsApp is the only route to the employer: no email address "
                    "and no application link appear anywhere in the posting",
                )
            )

    # --- structural signals: evidence is an absence, so nothing is quoted ---
    if employer_extracted and not (company or "").strip():
        signals.append(
            Signal(
                code="no_employer_named",
                weight=SIGNAL_WEIGHTS["no_employer_named"],
                evidence=None,
                basis="no employer name was extracted from the posting",
            )
        )

    # Brevity and silence are different accusations, and joining them with OR made
    # the signal state a falsehood: "Chef needed. Duties: cooking. Requirements: 2y
    # experience. Muscat. Send CV to hr@x.com" is 86 characters and describes the
    # role completely, yet fired a signal whose own basis line said it described
    # none. Telegram vacancies are routinely under 120 characters, so this was the
    # highest-volume false positive in the filter — and a required ingredient in
    # most combinations that reach the reject threshold.
    #
    # A posting is only "contact only, no role" when it is BOTH short AND carries
    # none of the role vocabulary. A short posting that names duties keeps its
    # score; a long one that names none still fires on the vocabulary test.
    stripped = body.strip()
    has_role_words = bool(_ROLE_WORDS.search(_fold(stripped)))
    if not has_role_words:
        if len(stripped) < MIN_ROLE_CHARS:
            signals.append(
                Signal(
                    code="contact_only_no_role",
                    weight=SIGNAL_WEIGHTS["contact_only_no_role"],
                    evidence=None,
                    basis=f"under {MIN_ROLE_CHARS} characters and containing none of the "
                    "responsibility/requirement vocabulary",
                )
            )
        else:
            signals.append(
                Signal(
                    code="no_role_described",
                    weight=SIGNAL_WEIGHTS["no_role_described"],
                    evidence=None,
                    basis="contains none of the responsibility/requirement vocabulary",
                )
            )

    # Searched over title+body but audited against the body alone, so a span taken
    # from the title ("15 vacancies available now") made assert_auditable raise on
    # an ordinary posting — dormant only because nothing called it. The signal
    # legitimately reads the title, so the AUDIT source is title+body too; see
    # `audit_source` below, which is what callers must pass.
    mass = _find(_MASS_VACANCY, audit_source(title, body))
    if mass and not has_role_words:
        signals.append(
            Signal(
                code="mass_vacancy_no_detail",
                weight=SIGNAL_WEIGHTS["mass_vacancy_no_detail"],
                evidence=mass,
                basis="advertises many vacancies with no role detail",
            )
        )

    return Assessment(
        risk=combine_signals([s.weight for s in signals]),
        signals=tuple(signals),
    )


def audit_source(title: str | None, body: str | None) -> str:
    """The exact text every signal is allowed to quote from.

    One function so the scorer and ``assert_auditable`` cannot disagree about it.
    They did: ``mass_vacancy_no_detail`` searched title+body while the audit was
    handed the body alone, so a span quoted from a title failed verification — the
    backstop would have crashed on ordinary postings the moment it was wired in.
    """
    return f"{title or ''}\n{body or ''}"


def assert_auditable(assessment: Assessment, source_text: str) -> None:
    """Re-verify every quoted span against the source.

    The backstop for the failure this module is most exposed to: a scorer bug
    that produces a confident number attached to evidence that is not in the
    posting. Structural signals are exempt by construction, not by exception —
    they never claim a quote in the first place.
    """
    for signal in assessment.signals:
        if signal.is_structural:
            continue
        assert signal.evidence is not None
        if signal.evidence not in source_text and _fold(signal.evidence) not in _fold(source_text):
            raise ValueError(
                f"legitimacy signal {signal.code} cites {signal.evidence!r}, "
                "which does not appear in the posting"
            )


def distribution(assessments: Iterable[Assessment], config: Config | None = None) -> dict[str, int]:
    """Band counts. This is the phase-2b gate metric.

    More than ~20% in ``adjudicate`` means the weights are miscalibrated — the
    rules are producing ambiguity rather than resolving it — and the answer is to
    retune them, not to send more postings to an LLM.
    """
    config = config or Config()
    counts = {"clean": 0, "adjudicate": 0, "reject": 0}
    for assessment in assessments:
        counts[assessment.band(config)] += 1
    return counts
