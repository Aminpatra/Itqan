"""Cross-agent configuration.

Import this module *before* anything touches paddleocr. Two of the settings below
(the model-source check and the warning filters) only take effect if they are in
place before paddle's import side effects run.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# PaddleX otherwise probes its model host on *every* import, which makes the CLI
# feel hung on a slow connection. Our models are already cached locally.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

# This env emits a RequestsDependencyWarning on every invocation (urllib3/chardet
# version skew). Harmless, but it corrupts the gap-collection prompts.
warnings.filterwarnings("ignore", message=".*urllib3.*chardet.*")
warnings.filterwarnings("ignore", category=UserWarning, module="paddle.*")

load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class Config:
    """Tunables shared by every agent.

    Defaults suit English-language documents. ``ocr_lang`` is the one to change
    for CVs in another script — PaddleOCR's English models will misread Arabic,
    Chinese or Devanagari rather than fail loudly. Exposed as ``--ocr-lang``.

    Fields are grouped by the agent that reads them. Agent A ignores the Agent B
    block entirely and vice versa; they share one class because both are process
    configuration and splitting them would mean two dotenv loads and two objects
    threaded through every factory.
    """

    model: str = field(default_factory=lambda: os.getenv("ITQAN_MODEL", "gpt-4o-mini"))
    temperature: float = 0.0

    # --- grounding thresholds (see shared/grounding.py) ---
    # >= grounded_threshold          -> accepted outright
    # adjudicate_threshold .. below  -> escalated to the LLM adjudicator
    # < adjudicate_threshold         -> dropped as ungrounded
    grounded_threshold: float = 0.92
    adjudicate_threshold: float = 0.75

    # An OCR block below this confidence makes any field it supports a gap.
    low_ocr_confidence: float = 0.60

    # Bounded HITL loop — after this many rounds, unfilled gaps finalize as null.
    max_review_rounds: int = 2

    # Promote skills the candidate credibly gained from completed coursework
    # (transcript + CV courses/certificates) but never wrote on their CV, so a
    # job-matcher can see them. They are added to skills.accepted flagged
    # origin="coursework_derived" and capped at "medium" quality — always below a
    # CV skill backed by a project or certificate. Set False to skip entirely.
    derive_coursework_skills: bool = True
    # Ceiling on how many derived skills to add: a full degree transcript can
    # teach dozens, and past a point they dilute the profile rather than enrich it.
    max_coursework_derived_skills: int = 15

    ocr_lang: str = "en"
    pdf_raster_dpi: int = 200

    # A PDF needs at least this much extractable text to count as having a real
    # text layer; below it we treat the file as scanned and route to OCR.
    pdf_text_min_chars: int = 200
    pdf_text_min_chars_per_page: int = 50

    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "output")

    # ------------------------------------------------------------------
    # Agent B — job ingestion
    # ------------------------------------------------------------------
    database_url: str = field(
        default_factory=lambda: os.getenv("ITQAN_DATABASE_URL", "")
    )

    # Schema, not tunables: the pgvector column has a fixed dimension, and
    # vectors from different models are not comparable. Changing either after
    # anything is written requires re-embedding the corpus.
    embedding_model: str = "text-embedding-3-small"
    embedding_dims: int = 1536

    cycle_hours: int = 12

    # Staleness counts CYCLES, not elapsed time — a skipped cycle (machine
    # asleep, source blocked) must not age postings that were never checked.
    stale_after_cycles: int = 3
    prune_after_days: int = 60
    degraded_after_cycles: int = 3

    # --- aggregation ---
    window_days: int = 30
    # Per-skill floor for the trend label. Without it, 1 -> 2 reads as +100%
    # "rising", which is noise published as a finding.
    trend_min_volume: int = 5
    trend_rising_ratio: float = 1.20
    trend_falling_ratio: float = 0.80
    # Per-SECTOR floor, distinct from trend_min_volume above: below this many
    # deduped postings, no number in the row is trustworthy and every row for
    # that sector is flagged low_confidence.
    low_confidence_min_postings: int = 10
    # Postings outside this set are stored and retrievable but not aggregated.
    # Scope is deliberately NOT a legitimacy signal — see agents/agent_b/legitimacy.
    in_scope_countries: tuple[str, ...] = ("OM",)

    # --- near-duplicate detection ---
    # Asymmetric on purpose. In-group (one publisher republishing itself) a
    # wrong merge is cheap; cross-group it would erase a real demand signal from
    # a different employer, so that path also requires human review.
    #
    # 0.97, raised from the originally-approved 0.93 after the first full live
    # run: at 0.93 on this template-heavy source, ~28 of 29 auto-merges were
    # DIFFERENT vacancies (a CFO merged into a CEO), undercounting demand.
    # Similarity now runs on essence embeddings (title + extracted skills +
    # seniority + location, see pipeline._essence_text), where true reposts
    # score ~0.99 and different jobs at one employer separate cleanly.
    neardup_in_group_threshold: float = 0.97
    neardup_cross_group_threshold: float = 0.97
    # Not 30: a repost more than two weeks apart is plausibly a genuine
    # re-advertisement, and merging it erases the sustained-demand signal.
    neardup_recent_days: int = 14
    neardup_candidates: int = 5

    # --- legitimacy filter (score is 1.0 clean .. 0.0 certainly scam) ---
    legitimacy_reject_at: float = 0.30
    legitimacy_adjudicate_low: float = 0.30
    legitimacy_adjudicate_high: float = 0.60

    # ------------------------------------------------------------------
    # Agent D — course ingestion (the supply side)
    # ------------------------------------------------------------------
    # 3 days, not 12h: courses change far slower than job postings, so a shorter
    # cycle would spend LLM/embedding budget re-confirming an unchanged catalog.
    # Staleness still counts CYCLES, so "3 missed cycles" here is ~9 days.
    course_cycle_hours: int = 72
    course_stale_after_cycles: int = 3
    course_prune_after_days: int = 60
    course_window_days: int = 90         # supply moves slowly; a wider window than jobs
    # A skill taught by fewer than this many courses is real but thin supply.
    course_low_confidence_min_courses: int = 3
    course_neardup_recent_days: int = 30
    # Coursera's public catalog page size; also the freeCodeCamp cert cap.
    coursera_max_pages: int = 8
    coursera_page_size: int = 100
    # Enrich each kept Coursera course by fetching its public course page
    # (robots-allowed) for rating/review_count/enrollment — the API exposes none
    # of these. One extra ~0.5-1MB fetch per course every cycle; set False to
    # skip and leave those fields NULL for Coursera.
    coursera_enrich: bool = True
    coursera_enrich_interval_s: float = 1.0

    # --- ESCO mapping ---
    # Minimum cosine similarity for an embedding-based skill->ESCO mapping.
    # 0.80, lowered from the initial 0.85 guess on first-corpus evidence
    # (2026-07-23): ESCO phrases skills as verbs ("prepare cost estimates"),
    # postings as nouns ("cost estimation"), and that equivalence lands at
    # 0.80-0.85 — an audited sample of that band was 13/14 correct, so 0.85 was
    # rejecting real matches. NOTE: lowering this further does not retroactively
    # remap; clear the affected method='unmapped' rows (or bump the taxonomy
    # version) and the next cycle re-tries them.
    esco_map_threshold: float = 0.80
    # Where the user-downloaded ESCO bundle lives (gitignored; EU dataset).
    esco_csv_path: Path = field(default_factory=lambda: PROJECT_ROOT / "ESCO" / "skills_en.csv")

    # ------------------------------------------------------------------
    # Agent C — skill-gap analysis
    # ------------------------------------------------------------------
    # A retrieved posting counts as a usable match above this candidate-essence
    # similarity. 0.43, lowered from the spec's 0.80 starting value on the first
    # live measurement (2026-07-23): a real CS-graduate profile against the live
    # corpus scored 0.41-0.48 for postings a human judges RELEVANT (technology
    # roles, data specialist, Oracle DBA) — cross-type similarity compresses, as
    # the original comment predicted, because a candidate essence aggregates
    # many skills while a posting names few. Cross-domain check (synthetic nurse
    # and salesperson profiles): rankings correct, relevant matches at
    # 0.53-0.66, so 0.43 is conservative for most profiles — tool-name-heavy
    # technical CVs sit lowest. One caveat: for those higher-scoring profiles
    # the bar also admits weaker cross-domain hits (~0.52), which the per-job
    # gap scores then contextualize honestly. The CLI prints the distribution
    # every run.
    agent_c_match_threshold: float = 0.43
    # The stats fallback compares the candidate against sector skills demanded
    # at least this often. Without the floor the comparison runs against every
    # freq-1 phrase ever aggregated (463 rows in the first live sector, ~87% of
    # them noise) and gap_score saturates at ~1.0, which says nothing.
    agent_c_fallback_min_freq: int = 2
    # Per-skill comparison bands: >= match is matched, [possible, match) is
    # possible_match and is NEVER auto-resolved either way, < possible is
    # missing.
    agent_c_skill_match: float = 0.80
    agent_c_skill_possible: float = 0.60
    # Below this many usable postings, per-job matching says more about retrieval
    # luck than about the market, so the sector fallback is ALSO computed. It no
    # longer suppresses the per-job results — four excellent matches are better
    # evidence than a sector aggregate, and hiding them was pure loss.
    agent_c_min_usable_postings: int = 5
    # The one model call in Agent C: it settles requirements the deterministic
    # tiers cannot, because cosine similarity is symmetric and cannot express
    # "TensorFlow is an instance of machine learning". Measured cost of not
    # having it: an ML engineering role scored a 100% gap for a candidate with
    # TensorFlow, PyTorch and scikit-learn. Off = fully deterministic, as before.
    agent_c_llm_matching: bool = True

    # --- Agent E (course recommender) ---
    # When several courses tie on coverage value, break the tie by these fields
    # in order. A config LIST, not a hardcoded chain, so product can reorder it
    # without a code change (the task explicitly required this). Nulls always
    # sort LAST in every field (a missing rating never beats a real one, and is
    # never coerced to 0); a final tie breaks on the lowest course_id, so the
    # selection is always total and reproducible. Valid fields: rating,
    # review_count, last_updated, price.
    agent_e_tiebreak: tuple[str, ...] = ("rating", "review_count", "last_updated", "price")

    # --- scraping ---
    user_agent: str = field(
        default_factory=lambda: os.getenv(
            "ITQAN_USER_AGENT", "ItqanJobBot/0.1 (+contact-not-configured)"
        )
    )
    max_response_bytes: int = 5_000_000
    blogger_max_pages: int = 8
    # MUST be >= window_days. A posting is counted as demand for `window_days`, so
    # if the feed is only read back `blogger_lookback_days`, everything in the gap
    # is never re-seen — it accrues missed_cycles every cycle and goes stale while
    # still being counted, silently undercounting the tail of every window. This
    # was 21 against a 30-day window; the invariant is asserted in __post_init__.
    blogger_lookback_days: int = 30
    telegram_max_pages: int = 3

    # A "roundup" Blogger post is an index listing several distinct vacancies,
    # each linking to its own same-site detail page (e.g. "…5 Jobs in Marketing,
    # HR, …"). When a post carries at least this many same-site individual-post
    # links, expand it: fetch each linked job page and ingest it as its OWN
    # posting, instead of collapsing all the jobs into one merged row. The job
    # count is whatever the links yield — never assumed. Set expand False to keep
    # the old single-row behaviour.
    blogger_expand_roundups: bool = True
    blogger_roundup_min_links: int = 2
    # Politeness/safety cap: never fetch more than this many job pages from one
    # roundup, however many links it lists.
    blogger_max_links_per_post: int = 12

    # When a posting links to the employer's own SINGLE job page (a "root
    # source"), fetch it and take the real required skills from there instead of
    # the aggregator's thin summary. Bounded and robots-respecting: only single
    # job-detail links (never careers hubs), every host's robots.txt honored, and
    # enrich-only (a failed/blocked fetch just keeps the aggregator's skills — no
    # posting is ever deleted). Set False to skip all root-page fetching.
    enrich_from_root_source: bool = True
    # Per-host politeness floor for those root-page fetches.
    root_fetch_interval_s: float = 1.0
    # Ceiling on root-page fetches per cycle. Without it "bounded" meant only
    # one-per-posting at a 1s interval, which on a large batch is an unbounded
    # crawl of third-party sites. None disables the cap.
    max_root_fetches_per_cycle: int | None = 60

    def __post_init__(self) -> None:
        # An invariant, not a preference: a posting counted as demand for
        # `window_days` must remain re-fetchable for at least that long, or it
        # ages toward deletion while the aggregation still expects to count it.
        # Raised here so the two values cannot drift apart unnoticed again.
        if self.blogger_lookback_days < self.window_days:
            raise ValueError(
                f"blogger_lookback_days ({self.blogger_lookback_days}) is shorter than "
                f"window_days ({self.window_days}): postings in the gap would be counted "
                f"as demand but never re-seen, so they would go stale while still counted."
            )

    def require_api_key(self) -> str:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        return key

    def require_database_url(self) -> str:
        if not self.database_url:
            raise RuntimeError(
                "ITQAN_DATABASE_URL is not set. Agent B needs Postgres with pgvector:\n"
                "  docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=x pgvector/pgvector:pg16\n"
                "  ITQAN_DATABASE_URL=postgresql://postgres:x@localhost:5432/postgres"
            )
        return self.database_url

    def require_identified_user_agent(self) -> str:
        """Refuse to make live requests without a real contact string.

        A scraper that does not identify itself gives a site operator no option
        except to block it, and no way to reach us if our cadence is a problem.
        Enforced here rather than documented, so it cannot be missed — dry runs
        are exempt because they make no requests.
        """
        if "contact-not-configured" in self.user_agent:
            raise RuntimeError(
                "ITQAN_USER_AGENT has no contact address. Set it before any live run, e.g.\n"
                '  ITQAN_USER_AGENT="ItqanJobBot/0.1 (+you@example.com)"'
            )
        return self.user_agent
