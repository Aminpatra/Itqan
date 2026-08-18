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
from typing import Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# PaddleX otherwise probes its model host on *every* import, which makes the CLI
# feel hung on a slow connection. Our models are already cached locally.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

# Paddle's oneDNN kernels, OFF.
#
# Measured 2026-08-01 in the deployment container (python:3.13-slim, paddlepaddle
# 3.3.0): every inference raises
#
#     NotImplementedError: (Unimplemented) ConvertPirAttribute2RuntimeAttribute
#     not support [pir::ArrayAttribute<pir::DoubleAttribute>]
#         at .../new_executor/instruction/onednn/onednn_instruction.cc:116
#
# The models load and the engine constructs fine; only `predict()` explodes, so
# nothing catches it until a real scanned page arrives. `FLAGS_use_mkldnn` and
# `FLAGS_enable_pir_api` are both ignored — PaddleX chooses the backend itself,
# and this is the switch it reads.
#
# `setdefault`, so a machine where oneDNN works can opt back into it for the
# speed. A crash is worse than a slower page.
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")

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

    # Chosen on measurement, 2026-07-29, by running THIS system's real prompts and
    # schemas (n=5) rather than on reputation. Two probes, both places where a
    # silent failure is expensive: Agent C's subsumption boundary (TensorFlow
    # satisfies "machine learning", JavaScript does NOT satisfy Java) and Agent
    # D's canonical-naming fold ("MS Excel" -> excel, which is what stops the
    # demand<->supply join fragmenting).
    #
    #   model           canonical  subsumption  median   est. $/month
    #   gpt-4o-mini       5/5         5/5        1.50s      1.41     (previous)
    #   gpt-5.4-mini      5/5         5/5        0.93s      8.32     <- chosen
    #   gpt-5.4           5/5         5/5        1.38s     27.74
    #   gpt-5.4-nano      0/5         5/5        0.97s      2.26     disqualified
    #   gpt-5-nano         -          5/5      12-19s        -       disqualified
    #
    # gpt-5.4-mini strictly dominates the previous default on both measurable
    # axes — 1.6x faster, identical on every quality probe — for a price
    # difference that is immaterial: the WHOLE system costs single-digit dollars
    # a month, so speed and quality are the real axes, not price.
    #
    # Two traps worth recording. `gpt-5.4-nano` is cheap and fast and fails the
    # canonical rule 0/5, which would quietly undo an Agent D fix. And the
    # GPT-5.0 generation's list price lies: `gpt-5-nano` is priced BELOW
    # gpt-4o-mini yet cost 9.4x more per call, because it emitted 4,480 reasoning
    # tokens across two trivial calls — billed at the output rate — and took
    # 12-19s, which would turn a 1,335-course backfill into 4-7 hours.
    #
    # NOT tiered per agent: every candidate scored 5/5 on both probes, so there
    # is no measured evidence a stronger model improves any output here.
    model: str = field(default_factory=lambda: os.getenv("ITQAN_MODEL", "gpt-5.4-mini"))
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
    # The text DETECTION model. PaddleOCR defaults to `PP-OCRv5_server_det`,
    # which needed more than 3 GB to run one CV page and was OOM-killed in a 2 GB
    # container; the mobile variant reads the same page correctly at a measured
    # 1,228 MB peak. That difference is what makes OCR fit on a 4 GB box beside
    # Postgres. Set to "PP-OCRv5_server_det" if you have the memory and want the
    # accuracy on hard scans.
    ocr_detection_model: str = "PP-OCRv5_mobile_det"
    pdf_raster_dpi: int = 200

    # A PDF needs at least this much extractable text to count as having a real
    # text layer; below it we treat the file as scanned and route to OCR.
    pdf_text_min_chars: int = 200
    pdf_text_min_chars_per_page: int = 50

    # Where uploads and per-run artifacts land. Env-settable because a deployment
    # mounts a volume wherever it likes, and the default — inside the source tree
    # — is exactly the wrong place when the source tree is a read-only image
    # layer. Unset, the behaviour is unchanged.
    output_dir: Path = field(
        default_factory=lambda: Path(os.getenv("ITQAN_OUTPUT_DIR") or (PROJECT_ROOT / "output"))
    )

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

    # How many aggregation snapshots to keep, for BOTH agents' stats tables.
    #
    # Every consumer filters with `window_end = (SELECT max(window_end) …)`, and
    # nothing was pruning across windows: measured 2026-07-28, skill_demand_stats
    # held 4 windows / 4,120 rows to serve 1,139 current ones, and
    # skill_supply_stats 3 windows / 18,334 to serve 10,202. At a 12h cycle that
    # is ~730 windows a year of dead history under every query.
    #
    # HARD FLOOR OF 2, asserted in code rather than trusted here: Agent B's trend
    # calculation reads `prior_frequency_count` from the STORED prior window, so
    # a retention of 1 would silently resurrect the fabricated-trend bug its own
    # audit fixed — every row labelled from a prior of zero.
    stats_retention_windows: int = 8

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
    # The supply snapshot's nominal span. NOTE this does NOT filter which courses
    # are counted: supply is a STOCK (a course from last year still teaches its
    # skills), unlike demand, which is a flow of vacancies open in a window. It
    # once did filter, which would have silently emptied the table as the corpus
    # aged past it. See agents/agent_d_course_ingest/aggregate.py.
    course_window_days: int = 90
    # A skill taught by fewer than this many courses is real but thin supply.
    # Also what Agent E calls "thin" when flagging a recommendation.
    course_low_confidence_min_courses: int = 3
    course_neardup_recent_days: int = 30
    # Below this many characters of name+description there is nothing to extract
    # from, and the call is skipped rather than spent finding that out. Generous:
    # freeCodeCamp's one-line descriptions are real courses and must pass.
    course_min_text_chars: int = 30
    # Courses per committed transaction. A normal cycle is well under this and
    # commits once; a backfill commits as it walks, so a failure at course 1,900
    # costs one chunk instead of the whole run, and no connection sits 'idle in
    # transaction' for twenty minutes of LLM calls.
    course_ingest_chunk_size: int = 200
    # Coursera's public catalog page size; also the freeCodeCamp cert cap.
    coursera_max_pages: int = 8
    coursera_page_size: int = 100
    # Set by `agent-d --backfill N` for one run: how many catalog pages to walk,
    # overriding coursera_max_pages. None outside a backfill. When set, the
    # per-course quality-signal page fetch is skipped — the API pass costs about
    # $0.0001 per course, while the enrichment fetch is ~1MB at a polite interval
    # each and would dominate a large walk. Those fields fill in on later cycles.
    course_backfill_pages: Optional[int] = None
    # Enrich each kept Coursera course by fetching its public course page
    # (robots-allowed) for rating/review_count/enrollment — the API exposes none
    # of these. One extra ~0.5-1MB fetch per course every cycle; set False to
    # skip and leave those fields NULL for Coursera.
    coursera_enrich: bool = True
    coursera_enrich_interval_s: float = 1.0
    # Stored courses to enrich per cycle, chosen from the DATABASE rather than
    # from this cycle's fetch. A backfill walks past the per-cycle page cap while
    # the adapter restarts at page 0 every cycle, so without this the ~1,300
    # courses beyond the cap are never revisited and never get a rating — the
    # deferred-enrichment promise the backfill makes would simply be false.
    # Measured backlog after the 2026-07-28 backfill: 1,749 courses.
    coursera_enrich_budget_per_cycle: int = 150

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
    # review_count, enrollment_count, level, last_updated, price.
    #
    # Order changed 2026-07-28 on live evidence. `rating` was first and RAW, so a
    # 5.0 from 10 reviews outranked a 4.9 from 30,000 — review_count only ever
    # broke an exact rating tie. It is now a confidence-shrunk score (see
    # agent_e_rating_prior_reviews), which folds volume into the rating itself,
    # and `enrollment_count` — collected by Agent D for 252 courses and
    # previously read by nothing — sits above the two fields that are almost
    # always null on this corpus (last_updated: 0 of 2,097; price: ~0).
    agent_e_tiebreak: tuple[str, ...] = (
        "rating", "enrollment_count", "review_count", "level", "last_updated", "price",
    )
    # Prior weight for the shrunk rating: the number of reviews at which a
    # course's own average carries as much weight as the corpus mean. Higher =
    # more sceptical of thin review counts. 50 is a conventional starting point
    # and, unlike the raw ordering it replaces, it cannot indict a well-reviewed
    # 4.9 in favour of a 5.0 nobody has taken.
    agent_e_rating_prior_reviews: int = 50
    # Rating differences below this do not decide anything. Measured on the real
    # gap file, two `project management` candidates scored 4.8070 and 4.8031 and
    # that 0.004 was choosing which course a person is told to take. Courses that
    # close are indistinguishable; below this resolution they tie and a real
    # signal (enrolments) decides, or the pick is reported as `arbitrary`.
    agent_e_rating_resolution: float = 0.1
    # Safety valve on how many candidate courses one missing skill may pull back.
    # Deliberately far above any realistic answer, so it never shapes a
    # recommendation — it only stops a catalog-scale corpus from materializing
    # tens of thousands of rows for a single gap. NOTE: if it ever does bind,
    # `supply.courses_available` saturates here rather than being exact.
    agent_e_max_candidates_per_skill: int = 200

    # ------------------------------------------------------------------
    # Email — password recovery only, for now
    # ------------------------------------------------------------------
    # Unset means no relay, and in PRODUCTION that fails the boot
    # (`assert_deployable`). An app that accepts every reset request and sends
    # nothing is worse than one that will not start: the endpoint answers 200
    # either way to avoid leaking who is registered, so a missing relay is
    # invisible to everyone except the person waiting for an email that is never
    # coming.
    smtp_host: str = field(default_factory=lambda: os.getenv("ITQAN_SMTP_HOST", ""))
    smtp_port: int = field(default_factory=lambda: int(os.getenv("ITQAN_SMTP_PORT", "587")))
    smtp_user: str = field(default_factory=lambda: os.getenv("ITQAN_SMTP_USER", ""))
    smtp_password: str = field(default_factory=lambda: os.getenv("ITQAN_SMTP_PASSWORD", ""))
    # The From address. Falls back to the login user, which is what most relays
    # require them to match anyway.
    smtp_from: str = field(default_factory=lambda: os.getenv("ITQAN_SMTP_FROM", ""))
    smtp_starttls: bool = field(
        default_factory=lambda: os.getenv("ITQAN_SMTP_STARTTLS", "1") not in ("0", "false", "False"))
    # Short: the send is on a background thread, but a relay that never answers
    # would otherwise hold one open for the life of the process.
    smtp_timeout_s: float = 20.0

    # Where the reset link points. The marketing site owns the forgot-password
    # page, per BACKEND.md, so this is the SITE origin and not the API's.
    site_url: str = field(
        default_factory=lambda: os.getenv("ITQAN_SITE_URL", "http://localhost:4321"))

    # Long enough to walk to another device and find the mail, short enough that
    # a link left in an inbox is not a standing key to the account.
    reset_token_minutes: int = 10

    # An endpoint that emails an arbitrary address on request is a bombing vector
    # aimed at people who are not our users, so it is bounded both ways: by
    # address, so one victim cannot be flooded, and by IP, so one attacker cannot
    # spray many. Enforced by the same guarded UPDATE the assistant's quota uses.
    reset_requests_per_email_hour: int = 3
    reset_requests_per_ip_hour: int = 10

    # ------------------------------------------------------------------
    # Email verification at signup (user decisions, 2026-08-17)
    # ------------------------------------------------------------------
    # Six digits is a million combinations, which is not a lot, and it is worth
    # being clear about what carries the weight here: NOT the length, and not the
    # sha256 the code is stored under. It is `verification_max_attempts`. Five
    # wrong answers kill the code, so guessing is capped at 5 in 1,000,000 per
    # issued code, and a resend replaces the code rather than granting a fresh
    # allowance against the old one.
    #
    # Six rather than eight because the limit already makes brute force
    # impractical, and every extra digit is real friction for someone reading a
    # code off one device and typing it into another.
    verification_code_digits: int = 6

    # The same ten minutes the reset link uses, so the two flows expire alike and
    # the email copy cannot drift apart from the code that enforces it.
    verification_code_minutes: int = 10
    verification_max_attempts: int = 5

    # Resends are bounded like every other outbound-mail path. Looser than the
    # reset limits because this endpoint requires a SESSION — it cannot be aimed
    # at a stranger's inbox, so the risk it bounds is a user hammering their own,
    # not a bombing vector. Per IP as well, since one attacker with many accounts
    # is the case the per-user limit cannot see.
    verification_resends_per_user_hour: int = 5
    verification_resends_per_ip_hour: int = 20

    # ------------------------------------------------------------------
    # Agent S — the assistant over a user's own results
    # ------------------------------------------------------------------
    # DECISIONS, not measurements (user, 2026-08-15), and labelled that way
    # because every other calibrated constant in this file carries the evidence
    # that set it and these two carry none yet.
    #
    # 30/day (was 10, raised 2026-08-17). Ten was set when this was a Q&A box;
    # on the chat screen a real conversation is easily ten turns, so the cap was
    # being hit mid-thought at exactly the moment the feature is most useful. It
    # is still a cap, and it is still the only thing between ordinary use and an
    # unbounded model bill — which is why it stays enforced in SQL.
    #
    # What to re-measure after a week of real use: how often a user reaches 30,
    # and how often a rerun actually CHANGED anything. A rerun returning
    # identical results is a credit spent for nothing, and at one per week that
    # is the whole allowance.
    #
    # Both are enforced by a guarded UPDATE in `AppStore.claim_quota`, never by
    # asking the model to respect them — see the module docstring there.
    assistant_daily_messages: int = 30
    assistant_weekly_reruns: int = 1

    # Quota periods are local days and weeks, NOT UTC ones. Telling a user in
    # Muscat that their quota resets at midnight and having it reset at 4am is a
    # bug, and taking UTC because it is the default is how you get it.
    assistant_tz: str = "Asia/Muscat"

    # How much of the conversation the model is shown. Ten messages a day means
    # a session is short, so this is the whole day rather than a sliding window
    # — and it bounds the prompt, which is the cost control that matters when a
    # user controls how often we call a model.
    assistant_history_turns: int = 10

    # --- scraping ---
    user_agent: str = field(
        default_factory=lambda: os.getenv(
            "ITQAN_USER_AGENT", "ItqanJobBot/0.1 (+contact-not-configured)"
        )
    )
    max_response_bytes: int = 5_000_000

    # --- browser transport (shared/scraping/browser.py) --------------------
    # Chromium instead of httpx, for sources that render their content in
    # JavaScript. Off per-source by default: a browser costs the target a full
    # page render and costs us ~300-500 MB of RAM, so it is opted into by the
    # sources that need it rather than imposed on the ones that do not.
    browser_enabled: bool = field(
        default_factory=lambda: os.getenv("ITQAN_BROWSER", "1") not in ("0", "false", "False")
    )
    # Floor between requests to one source. Higher than the httpx default on
    # purpose — a rendered page is far more work for the target than a document.
    browser_min_interval_s: float = 1.5
    # Added at random on top of the interval. Mechanically-spaced requests are
    # what make an otherwise polite crawler read as a script.
    browser_jitter_s: float = 0.7
    # Hard ceiling per URL, so one unresponsive host cannot stall an unattended
    # cycle behind it.
    browser_timeout_s: float = 90.0
    # Render the employer pages `root_fetch` follows.
    #
    # MEASURED 2026-08-08, on the 9 destination links one live el7far cycle
    # produced. Of the 6 that robots did not refuse, THREE are unreadable to
    # httpx and readable to Chromium — and none of the three fail in a way a
    # retry would fix:
    #
    #   sah.om            SSL: CERTIFICATE_VERIFY_FAILED  ->  5,003 chars
    #   careers.dhl.com   HTTP 410 Gone                   ->  4,934 chars
    #   sjscareers        empty shell                     ->  (still empty)
    #
    # An incomplete certificate chain is common on small Omani hosts and
    # browsers repair it; a 410 to a non-browser client is a site declining to
    # talk to scripts. The two hosts httpx CAN read (careers.oq.com,
    # erp.uob.edu.om) return byte-identical text rendered, so nothing regresses.
    browser_fetch_destinations: bool = True
    # Source adapters that should fetch through Chromium instead of httpx.
    #
    # EMPTY, and that is a measured decision rather than an oversight. The same
    # probe found NO gain for any of them: el7far's article pages 1.00x,
    # telegram byte-identical, dubizzle 0.86x (a browser at `domcontentloaded`
    # sees FEWER listing cards than the served HTML carries). And el7far's feed
    # is Atom XML, which a browser hands back wrapped in Chrome's XML viewer —
    # rendering it would break the parse for a gain that does not exist.
    #
    # The knob stays because a source can start refusing scripts overnight, and
    # then this is a one-value change rather than a code change.
    #
    # AGENT D's course sources were wired to the same factory on 2026-08-16 and
    # are empty here for the same reason — measured, not assumed:
    #
    #   coursera  a course we hold with NO rating serves HTML containing no
    #             `ratingValue` and no `AggregateRating` at all, and enrichment
    #             has already run on 2,000 of 2,000 rows. The 1,750 missing
    #             ratings are missing because the publisher never issued them.
    #             A browser cannot conjure a rating that does not exist.
    #   edx       552 KB of server-rendered HTML with full JSON-LD to httpx.
    #   fcc       a JSON file in a GitHub repo; no browser was ever involved.
    #
    # Enabling one here costs the target a full page render and costs us
    # 300-500 MB of RAM on a 4 GB box that already peaks at 1.2 GB during OCR.
    browser_sources: tuple[str, ...] = ()
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

    def stats_windows_to_keep(self) -> int:
        """Retention, with the floor that protects the trend calculation.

        Enforced here rather than left to whoever edits the config, because the
        failure it prevents is silent: with one window retained, Agent B's
        `prior_frequency_count` lookup finds no prior, every skill is labelled
        from a prior of zero, and the fabricated-trend bug its audit fixed comes
        straight back — with no error and plausible-looking numbers.
        """
        return max(2, int(self.stats_retention_windows))

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
