"""Agent E nodes — deterministic course selection, then one fenced rationale.

The selection is arithmetic: a weighted greedy set-cover over Agent C's missing
skills and Agent D's courses. It is fully reproducible — every tie is broken by a
configured field chain and finally by ``course_id``, so the same inputs always
produce the same assignment. No LLM touches selection.

The ONLY LLM call is ``generate_rationale``: one short message per recommended
course, built from the ALREADY-FINALIZED record. It is fenced by construction —
it receives a rendered fact sheet (never the raw numbers behind it: no
priority_score float, no ESCO code, no gap_score), so it can explain the
decision but cannot alter it or invent beyond it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from shared.config import Config
from shared.contracts import load_gap
from shared.course_market import courses_for_skills

from .state import RecommendState

# 1.1 adds, all additively: `supply` (how many courses back each pick),
# `selection` (whether anything about the course decided it), `calibration`, and
# `course.rationale_source`. Nothing was removed or re-typed.
SCHEMA_VERSION = "itqan.course_reco/1.1"

_INF = float("inf")

# Verbatim from the task brief. The rationale model sees ONLY the rendered user
# message below — it never receives priority_score, esco_code, or gap_score.
RATIONALE_SYSTEM_PROMPT = (
    "You are writing a short message to a job-seeking candidate explaining why a "
    "specific course was recommended to close a specific skill gap.\n\n"
    "Rules:\n"
    "- Only reference facts explicitly given below. Never invent course content, "
    "instructor names, syllabus details, or outcomes not present in the input. "
    "Never guess why a course has its rating or what reviews say — you are given "
    "a count and a number, not review text.\n"
    "- If the course covers additional missing skills beyond the primary one, "
    "mention that.\n"
    "- If rating, review_count, or price are null/not available, omit them — "
    "never substitute a plausible-sounding guess.\n"
    "- ONLY if 'Recommendation basis' says general demand, use softer language "
    "(\"a solid option based on general demand in your field\"). If it names "
    "specific postings, do NOT use that hedge — it would understate real "
    "evidence.\n"
    "- The 'Courses available' line is a COUNT, and it tells you whether the "
    "field is thin. Call it small ONLY when the line itself says few courses "
    "cover the skill. Nine, twelve, twenty-two or thirty-eight courses are NOT "
    "'a small number of options' — claiming otherwise is false and is checked in "
    "code. If the line does not flag it, say nothing about how many exist.\n"
    "- If 'Recommendation basis' reports that several courses matched equally "
    "well, you MUST say so: the choice between them was not based on anything "
    "about the courses, and presenting it as a ranked best is the one thing this "
    "message must never do.\n"
    "- 2-3 sentences, plain and direct, no marketing language.\n"
    "- Never state a NUMBER that is not in the input above — no durations, module "
    "counts, instructor counts, completion times, or ratings beyond what is "
    "given. An invented figure is the most authoritative-sounding thing you can "
    "write; every number is checked against the input in code, and a rationale "
    "that adds one is discarded.\n"
    "- Never mention priority_score as a number, ESCO codes, gap_score, or "
    "internal pipeline terms. 'Demand level' ranks this gap against THE "
    "CANDIDATE'S OTHER GAPS — it is not a measurement of the job market. Say "
    "\"one of the bigger gaps in your profile\"; do NOT say \"a skill that came up "
    "often in roles you'd be a good fit for\", which is a claim about the labour "
    "market that this ranking does not support.\n"
    "- Output plain text only — this is the message itself, not JSON or markdown."
)


@dataclass
class Deps:
    """Injected once; nodes never construct their own I/O. Fakes go in here for
    the offline tests, exactly like Agent C's Deps."""

    config: Config
    llm: Any = None                                   # chat model; None in --dry-run
    courses_reader: Callable[..., dict[str, Any]] = courses_for_skills


# ===========================================================================
# 1. load_missing_skills
# ===========================================================================
def make_load_missing_skills(deps: Deps) -> Callable[[RecommendState], dict]:
    def load_missing_skills(state: RecommendState) -> dict:
        # Validated through the published contract rather than duck-typed. This
        # used to be raw `.get()` chains, so a renamed field on Agent C's side
        # would not fail — it would silently fall through to the far lossier
        # `most_common_missing_skills` path and quietly produce worse
        # recommendations. Unknown fields are still ignored, so Agent C can add
        # to the envelope freely; what fails now is a field that VANISHES.
        gap = load_gap(state["gap_path"])
        aggregate = gap.aggregate
        used_fallback = gap.used_fallback
        warnings: list[str] = []

        details = aggregate.missing_skill_details
        missing: list[dict[str, Any]] = []
        seen: set[str] = set()
        if details:
            # The enriched Agent C aggregate: each skill carries its esco_code and
            # inherited priority_score. Already deduped upstream; dedup again
            # defensively (a skill triggers exactly one course search).
            for d in details:
                key = (d.skill or "").strip().lower()
                if not key or key in seen:
                    continue
                seen.add(key)
                missing.append({
                    "skill": key,
                    "esco_code": d.esco_code,
                    "priority_score": float(d.priority_score or 0.0),
                    # Agent C's actual market evidence, previously dropped on the
                    # floor here. `jobs_missing_in` is a plain, checkable count of
                    # retrieved postings that asked for this skill — the true
                    # version of the claim `priority_bucket` was being asked to
                    # imply. `demand_rate`/`low_confidence` say how much the
                    # demand side trusts its own numbers.
                    "jobs_missing_in": d.jobs_missing_in,
                    "demand_rate": d.demand_rate,
                    "demand_low_confidence": bool(d.low_confidence),
                })
        else:
            # Older gap file, before missing_skill_details existed: fall back to
            # the bare skill list — no ESCO codes, no weights — and say so.
            for s in aggregate.most_common_missing_skills:
                key = (s or "").strip().lower()
                if not key or key in seen:
                    continue
                seen.add(key)
                missing.append({"skill": key, "esco_code": None, "priority_score": 1.0,
                                "jobs_missing_in": None, "demand_rate": None,
                                "demand_low_confidence": False})
            if missing:
                warnings.append(
                    "gap file has no missing_skill_details; using "
                    "most_common_missing_skills without ESCO codes or priority "
                    "weights (regenerate with Agent C for full fidelity)"
                )

        if not missing:
            warnings.append("no missing skills in the gap file — nothing to recommend")

        return {
            "missing": missing,
            "used_fallback": used_fallback,
            "user_id": state.get("user_id") or gap.user_id or "",
            "warnings": warnings,
        }

    return load_missing_skills


# ===========================================================================
# 2. retrieve_candidate_courses
# ===========================================================================
def make_retrieve_candidate_courses(deps: Deps) -> Callable[[RecommendState], dict]:
    def retrieve_candidate_courses(state: RecommendState) -> dict:
        missing = state.get("missing", [])
        # ESCO-coded skills retrieve by concept; unmapped skills fall back to an
        # EXACT normalized skill_key match (never a fuzzy guess).
        esco_codes = sorted({m["esco_code"] for m in missing if m["esco_code"]})
        skill_keys = sorted({m["skill"] for m in missing if not m["esco_code"]})

        out = deps.courses_reader(esco_codes, skill_keys, config=deps.config)
        by_esco = out.get("by_esco", {})
        by_key = out.get("by_key", {})

        courses_by_id: dict[str, Any] = {}
        course_covers: dict[str, list[str]] = {}
        skill_candidates: dict[str, list[str]] = {}
        for m in missing:
            skill, esco = m["skill"], m["esco_code"]
            cands = by_esco.get(esco, []) if esco else by_key.get(skill, [])
            ids: list[str] = []
            for cc in cands:
                courses_by_id[cc.course_id] = cc
                course_covers.setdefault(cc.course_id, []).append(skill)
                ids.append(cc.course_id)
            # A zero-candidate skill stays in the set (empty list), never dropped.
            skill_candidates[skill] = ids

        return {
            "courses_by_id": courses_by_id,
            "course_covers": {cid: sorted(sk) for cid, sk in course_covers.items()},
            "skill_candidates": skill_candidates,
        }

    return retrieve_candidate_courses


# ===========================================================================
# 3. greedy_cover_assign  (pure, deterministic, no LLM)
# ===========================================================================
def _epoch(iso: Optional[str]) -> Optional[float]:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


# Courses are labelled by their own provider; a missing label is unknown, not
# "advanced". A MISSING skill is by definition one the candidate has shown no
# evidence in, so an introductory course is the better default — this is the one
# place Agent E can reason about fit without needing the candidate's profile.
_LEVEL_ORDER = {"beginner": 0, "intermediate": 1, "advanced": 2}


def shrunk_rating(cc: Any, prior_mean: float, prior_reviews: int) -> Optional[float]:
    """A rating weighted by how much evidence stands behind it.

    ``(v/(v+m))·R + (m/(v+m))·C`` — the course's own average R pulled toward the
    corpus mean C, with the pull decided by its review count v against a prior
    weight m.

    Raw rating ordering put a **5.0 from 10 reviews above a 4.9 from 30,000**,
    because review_count only ever broke an *exact* rating tie. On this corpus
    that is not hypothetical: every top-rated row is a 5.0 with 10-14 reviews.
    Shrinkage folds volume into the number itself, so confidence and score stop
    being separate lexicographic stages.

    A course with a rating but no review count is treated as v=0 — all prior, no
    evidence — rather than being trusted at face value.
    """
    if cc.rating is None:
        return None
    v = cc.review_count or 0
    return (v * cc.rating + prior_reviews * prior_mean) / (v + prior_reviews)


def _field_key(cc: Any, field_name: str, ctx: dict[str, Any] | None = None) -> float:
    """One tie-break field as an ascending, smaller-is-better number. A null
    always yields +inf, so it sorts LAST in every field and is NEVER treated as
    0 (a 0 rating is a real, distinct value that would sometimes win)."""
    ctx = ctx or {}
    if field_name == "rating":
        s = shrunk_rating(cc, ctx.get("prior_mean", 0.0), ctx.get("prior_reviews", 50))
        if s is None:
            return _INF
        # Rounded to a resolution a learner could act on. Measured on the real
        # gap file, the top two `project management` candidates scored 4.8070 and
        # 4.8031 — a gap of 0.004 that was silently deciding which course a
        # person is told to take. Two courses that close are not distinguishable,
        # and letting noise pick between them is the same false precision as a
        # gap_score of 0.0 meaning "we parsed nothing". Below the resolution they
        # tie and the next real signal (enrolments) decides — or, if nothing
        # does, the pick is reported as `arbitrary` rather than dressed up.
        step = ctx.get("resolution") or 0.1
        return -round(s / step)
    if field_name == "review_count":
        return -cc.review_count if cc.review_count is not None else _INF
    if field_name == "enrollment_count":
        # Collected by Agent D for 252 courses and, until now, read by nothing.
        return -cc.enrollment_count if cc.enrollment_count is not None else _INF
    if field_name == "level":
        rank = _LEVEL_ORDER.get((getattr(cc, "level", None) or "").strip().lower())
        return rank if rank is not None else _INF      # beginner first; unknown last
    if field_name == "last_updated":
        e = _epoch(cc.last_updated)
        return -e if e is not None else _INF          # more recent = smaller = better
    if field_name == "price":
        amt = cc.price.amount if (cc.price and cc.price.amount is not None) else None
        return amt if amt is not None else _INF        # cheaper = smaller = better; free(0) wins
    if field_name == "price_is_free":
        # The candidate asked for free courses. This is a RANKING field, not a
        # filter (user decision 2026-07-30): a gap whose only course is paid still
        # needs an answer, and on the measured corpus a filter would cut 2,099
        # courses to the 98 freeCodeCamp ones.
        #
        # Three-valued on purpose, and the middle value is the point. 0 of 1,999
        # Coursera courses publish a price, so "not free" and "we do not know"
        # are overwhelmingly the common cases and they are NOT the same claim.
        # An unknown price sorts between a known-free and a known-paid course:
        # ahead of something we know costs money, behind something we know does
        # not. Coercing unknown to "paid" would bury most of the catalogue on a
        # fact nobody established.
        if cc.price is None:
            return 1.0                                 # unknown
        return 0.0 if cc.price.is_free else 2.0
    raise ValueError(f"unknown agent_e_tiebreak field {field_name!r}")


def rating_prior_mean(courses: list[Any]) -> float:
    """The corpus mean the shrunk rating pulls toward, over the rated candidates
    in THIS run — self-contained and deterministic, no extra query. With nothing
    rated it is unused (every shrunk rating is None and the field is skipped)."""
    rated = [c.rating for c in courses if c.rating is not None]
    return sum(rated) / len(rated) if rated else 0.0


def greedy_assign(
    missing: list[dict[str, Any]],
    skill_candidates: dict[str, list[str]],
    courses_by_id: dict[str, Any],
    tiebreak: tuple[str, ...],
    *,
    prior_reviews: int = 50,
    resolution: float = 0.1,
) -> tuple[dict[str, list[str]], list[str], dict[str, dict[str, Any]]]:
    """Weighted greedy set-cover.

    Returns ``(assigned, no_course_found, basis)``. ``assigned`` maps a course_id
    to the skills it was chosen to cover (each skill covered exactly once, each
    course appearing once); ``basis`` records HOW each course won.

    The objective is ``n × Σpriority`` — the **product** the design specified.
    It was implemented as lexicographic ``(-n, -weight)``, which is a different
    rule: under it a course covering three trivial gaps always beat one covering
    two critical gaps, because priority could only ever break a tie between equal
    coverage counts. The product lets breadth and importance trade off, which is
    what "weighted" was supposed to mean.
    """
    priority = {m["skill"]: m["priority_score"] for m in missing}
    prior_mean = rating_prior_mean(list(courses_by_id.values()))
    ctx = {"prior_mean": prior_mean, "prior_reviews": prior_reviews,
           "resolution": resolution}

    course_can_cover: dict[str, set[str]] = {}
    for skill, ids in skill_candidates.items():
        for cid in ids:
            course_can_cover.setdefault(cid, set()).add(skill)

    uncovered = {s for s, ids in skill_candidates.items() if ids}
    no_course_found = sorted(s for s, ids in skill_candidates.items() if not ids)

    assigned: dict[str, list[str]] = {}
    basis: dict[str, dict[str, Any]] = {}
    while uncovered:
        candidate_ids = {cid for s in uncovered for cid in skill_candidates[s]}

        def objective(cid: str) -> tuple:
            covers = course_can_cover[cid] & uncovered
            # min() picks the smallest key, so the value is negated: highest
            # coverage-value first, then the configured tie-break fields, then
            # the lowest course_id for total determinism.
            value = len(covers) * sum(priority[s] for s in covers)
            cc = courses_by_id[cid]
            return (-value, *(_field_key(cc, f, ctx) for f in tiebreak), cid)

        scored = {cid: objective(cid) for cid in candidate_ids}
        best = min(candidate_ids, key=lambda c: scored[c])

        # Everything except the final course_id element. If more than one
        # candidate matches the winner here, NOTHING distinguished them and the
        # pick came down to a SHA-256 hash. Measured on the live corpus:
        # `communication skills` had 13 candidates, none with a rating, price or
        # date — so the "recommendation" was hash order. Saying so is the same
        # instinct as Agent C's `insufficient_data`; presenting it as a
        # considered choice is the dishonesty.
        winning = scored[best][:-1]
        equivalent = sum(1 for key in scored.values() if key[:-1] == winning)

        newly = sorted(course_can_cover[best] & uncovered)
        assigned[best] = newly
        basis[best] = {
            "basis": "arbitrary" if equivalent > 1 else "quality",
            "equivalent_candidates": equivalent,
            "candidates_considered": len(candidate_ids),
        }
        uncovered.difference_update(newly)

    return assigned, no_course_found, basis


def effective_tiebreak(configured: tuple[str, ...], *, prefer_free: bool) -> tuple[str, ...]:
    """The tiebreak chain for this run, with the candidate's answer applied.

    Prepending rather than replacing is the whole design: "free" decides FIRST,
    and every configured signal still decides among the courses that tie on it. A
    preference reorders the ranking; it never throws the ranking away.
    """
    if not prefer_free:
        return configured
    return ("price_is_free", *(f for f in configured if f != "price_is_free"))


def make_greedy_cover_assign(deps: Deps) -> Callable[[RecommendState], dict]:
    def greedy_cover_assign(state: RecommendState) -> dict:
        assigned, no_course, basis = greedy_assign(
            state.get("missing", []),
            state.get("skill_candidates", {}),
            state.get("courses_by_id", {}),
            effective_tiebreak(tuple(deps.config.agent_e_tiebreak),
                               prefer_free=bool(state.get("prefer_free"))),
            prior_reviews=deps.config.agent_e_rating_prior_reviews,
            resolution=deps.config.agent_e_rating_resolution,
        )
        return {"assigned": assigned, "no_course_found_skills": no_course,
                "selection_basis": basis}

    return greedy_cover_assign


# ===========================================================================
# 4. attach_flags  (pure)
# ===========================================================================
def _compute_buckets(missing: list[dict[str, Any]]) -> dict[str, str]:
    """Rank each gap against THIS candidate's other gaps — high/moderate/some.

    Computed in code so a raw priority float never reaches the LLM.

    **This is a relative rank, not a market claim.** It says "your biggest gap",
    not "in demand": a candidate whose gaps are all marginal still gets a "high".
    The prompt used to render it as *"a skill that came up often in roles you'd be
    a good fit for"*, which is a statement about the labour market that this
    number cannot support — see RATIONALE_SYSTEM_PROMPT.

    Two degenerate cases the thirds-of-the-range arithmetic got wrong, both live:
    with a single missing skill `lo == hi`, so every threshold collapses and the
    one gap is ALWAYS "high"; and when every gap carries the same weight, all of
    them came out "high" while being, by definition, indistinguishable.
    """
    scores = [m["priority_score"] for m in missing]
    if not scores:
        return {}
    lo, hi = min(scores), max(scores)
    span = hi - lo

    # Nothing separates them (one skill, or all equal). Ranking is meaningless
    # here, so every gap gets the same middle label rather than a fabricated top.
    if span <= 0:
        return {m["skill"]: "moderate" for m in missing}

    t_high = lo + 2 * span / 3
    t_mod = lo + span / 3

    def bucket(s: float) -> str:
        if s >= t_high:
            return "high"
        if s >= t_mod:
            return "moderate"
        return "some"

    return {m["skill"]: bucket(m["priority_score"]) for m in missing}


def make_attach_flags(deps: Deps) -> Callable[[RecommendState], dict]:
    def attach_flags(state: RecommendState) -> dict:
        missing = state.get("missing", [])
        priority = {m["skill"]: m["priority_score"] for m in missing}
        esco_of = {m["skill"]: m["esco_code"] for m in missing}
        demand_of = {m["skill"]: {
            "jobs_missing_in": m.get("jobs_missing_in"),
            "rate": m.get("demand_rate"),
            "low_confidence": m.get("demand_low_confidence", False),
        } for m in missing}
        buckets = _compute_buckets(missing)
        courses_by_id = state.get("courses_by_id", {})
        candidates = state.get("skill_candidates", {})
        selection = state.get("selection_basis", {})
        prior_reviews = deps.config.agent_e_rating_prior_reviews
        prior_mean = rating_prior_mean(list(courses_by_id.values()))

        # How much supply exists for each gap. This is Agent D's whole reason for
        # aggregating a supply side, and it costs nothing here: retrieval already
        # fetched every eligible course per skill, so the depth is the length of
        # that list — no second query, no join to the stats table.
        #
        # It matters to the candidate because "one obscure course exists" and
        # "forty courses teach this" are very different situations behind the same
        # single recommendation, and only one of them is a safe plan.
        thin_below = deps.config.course_low_confidence_min_courses

        def supply_for(skill: str) -> dict[str, Any]:
            n = len(candidates.get(skill, []))
            return {"courses_available": n, "thin": n < thin_below}

        recs: list[dict[str, Any]] = []
        for cid, covered in state.get("assigned", {}).items():
            # The entry leads with the highest-priority skill this course covers;
            # the rest go in covers_other_skills so the course appears ONCE.
            primary = min(covered, key=lambda s: (-priority[s], s))
            others = sorted(s for s in covered if s != primary)
            cc = courses_by_id[cid]
            recs.append({
                "skill": primary,
                "esco_code": esco_of.get(primary),
                "priority_score": priority[primary],
                # A rank among THIS candidate's gaps, not a market claim.
                "priority_bucket": buckets.get(primary, "some"),
                # The real market evidence, straight from Agent C.
                "demand": demand_of.get(primary, {}),
                "supply": supply_for(primary),
                # Did anything about the course decide this, or did it come down
                # to a hash? Published so a reader can tell.
                "selection": selection.get(cid, {"basis": "quality"}),
                "course": {
                    "course_id": cc.course_id,
                    "title": cc.title,
                    "provider": cc.provider,
                    "url": cc.url,
                    "covers_other_skills": others,
                    "quality": {
                        "rating": cc.rating,
                        # What the ranking actually used: raw rating alone put a
                        # 5.0 from 10 reviews above a 4.9 from 30,000.
                        "rating_shrunk": (round(s, 4)
                                          if (s := shrunk_rating(cc, prior_mean, prior_reviews))
                                          is not None else None),
                        "review_count": cc.review_count,
                        "enrollment_count": cc.enrollment_count,
                        "level": getattr(cc, "level", None),
                        "price": cc.price.model_dump() if cc.price else None,
                        "last_updated": cc.last_updated,
                    },
                    "rationale": None,          # filled by generate_rationale
                },
                "no_course_found": False,
            })

        for skill in state.get("no_course_found_skills", []):
            recs.append({
                "skill": skill,
                "esco_code": esco_of.get(skill),
                "priority_score": priority.get(skill, 0.0),
                "priority_bucket": buckets.get(skill, "some"),
                "demand": demand_of.get(skill, {}),
                # Explicit rather than implied by `course: null` — a consumer
                # should not have to infer a zero.
                "supply": {"courses_available": 0, "thin": True},
                "course": None,
                "no_course_found": True,
            })

        # Most important first, then by skill for a stable order.
        recs.sort(key=lambda r: (-r["priority_score"], r["skill"]))
        return {"recommendations": recs}

    return attach_flags


# ===========================================================================
# 5. generate_rationale  (the ONLY LLM call — one per recommended course)
# ===========================================================================
def _asked_for_line(rec: dict[str, Any], prefer_free: bool) -> str:
    """What the candidate asked for, and whether THIS course satisfies it.

    The second half is the part that matters. Telling a model "the learner wants
    free courses" without also telling it what this course costs is an invitation
    to congratulate itself: measured on this corpus, 0 of 1,999 Coursera courses
    publish a price, so the most likely course in front of it has an unknown price
    and "free, as you asked" would be a fabrication with a friendly tone.
    """
    if not prefer_free:
        return "nothing stated"
    price = (rec["course"]["quality"] or {}).get("price") or {}
    if price.get("is_free"):
        return "free courses — this one IS free, so you may say so"
    if price.get("amount") is not None:
        return ("free courses — this one is NOT free, and it was chosen because "
                "nothing free covers the skill. Do NOT call it free")
    return ("free courses — this course's price is NOT published, so it may or may "
            "not be free. Do NOT call it free")


def _render_user_message(rec: dict[str, Any], used_fallback: bool,
                         prefer_free: bool = False) -> str:
    """The verbatim USER MESSAGE TEMPLATE, filled from the finalized record.
    Nulls become 'not available' (never a guess); priority_score and esco_code
    are deliberately absent — only priority_bucket's plain word appears."""
    course = rec["course"]
    q = course["quality"]
    others = course["covers_other_skills"]
    also = ", ".join(others) if others else "nothing else — single-skill match"

    rating = q["rating"] if q["rating"] is not None else "not available"
    reviews = q["review_count"] if q["review_count"] is not None else "not available"

    price = q["price"]
    if not price:
        price_line = "not available"
    elif price.get("is_free"):
        price_line = "free"
    elif price.get("amount") is not None:
        currency = price.get("currency")
        price_line = f"{price['amount']} {currency}" if currency else f"{price['amount']}"
    else:
        price_line = "not available"

    last_updated = q["last_updated"] if q["last_updated"] else "not available"
    basis = (
        "specific job postings you matched with" if not used_fallback
        else "general demand data for your field (fewer specific postings were available)"
    )
    # A plain count of alternatives, not an internal score — it tells the reader
    # whether this recommendation is a pick from many or the only thing there is.
    supply = rec.get("supply") or {}
    available = supply.get("courses_available")
    if available is None:
        supply_line = "not available"
    elif supply.get("thin"):
        supply_line = f"{available} (few courses cover this skill — say so)"
    else:
        # Stated explicitly, because the model otherwise read any count as small
        # and called a field of 38 courses "one of a small number of options".
        supply_line = f"{available} (a normal range — do NOT call this scarce)"

    selection = rec.get("selection") or {}
    if selection.get("basis") == "arbitrary":
        basis = (f"{selection.get('equivalent_candidates', 0)} courses matched this skill "
                 f"equally well and nothing distinguished them, so this is a representative "
                 f"pick, NOT a ranked best — you must say so")

    # The REAL market evidence, which Agent C measures and Agent E used to drop.
    # A plain count of retrieved postings that asked for this skill is something
    # the model can state truthfully; `priority_bucket` is only a rank among the
    # candidate's own gaps and cannot support a claim about the market.
    demand = rec.get("demand") or {}
    jobs = demand.get("jobs_missing_in")
    if jobs:
        jobs_line = f"{jobs} of the roles you matched asked for it"
        if demand.get("low_confidence"):
            jobs_line += " (thin data — say so)"
    else:
        jobs_line = "not available"

    return "\n".join([
        f"Skill: {rec['skill']}",
        f"Also covers: {also}",
        f"Course: {course['title']} ({course['provider'] or 'the provider'})",
        f"Rating: {rating}",
        f"Reviews: {reviews}",
        f"Price: {price_line}",
        f"Last updated: {last_updated}",
        f"How often this skill was asked for: {jobs_line}",
        f"Rank among your own gaps: {rec['priority_bucket']}",
        f"Courses available for this skill: {supply_line}",
        f"Recommendation basis: {basis}",
        f"What the learner asked for: {_asked_for_line(rec, prefer_free)}",
    ])


def deterministic_rationale(rec: dict[str, Any], used_fallback: bool) -> str:
    """The rationale built in code from the finalized record.

    Used when the model is absent, fails, or says something the fact sheet does
    not support. Every clause is conditional on the value actually existing, so a
    null is silently omitted rather than rendered as a guess — the same rule the
    prompt gives the model, enforced instead of requested.
    """
    course, q = rec["course"], rec["course"]["quality"]
    others = course["covers_other_skills"]
    provider = f" from {course['provider']}" if course.get("provider") else ""

    lead = f"{course['title']}{provider} covers {rec['skill']}"
    if others:
        lead += f", and also {', '.join(others)}"
    parts = [lead + "."]

    demand = rec.get("demand") or {}
    if demand.get("jobs_missing_in"):
        n = demand["jobs_missing_in"]
        hedge = " though that is from thin data" if demand.get("low_confidence") else ""
        parts.append(f"{n} of the roles you matched asked for it{hedge}.")

    if q.get("rating") is not None and q.get("review_count"):
        parts.append(f"It is rated {q['rating']} from {q['review_count']} reviews.")
    elif q.get("rating") is not None:
        parts.append(f"It is rated {q['rating']}.")
    elif q.get("enrollment_count"):
        parts.append(f"It has {q['enrollment_count']} enrolments.")

    price = q.get("price") or {}
    if price.get("is_free"):
        parts.append("It is free.")

    if (rec.get("supply") or {}).get("thin"):
        parts.append("Few courses cover this skill, so there is little to choose from.")
    elif (rec.get("selection") or {}).get("basis") == "arbitrary":
        n = rec["selection"].get("equivalent_candidates", 0)
        parts.append(
            f"{n} courses matched this skill equally well and none carries ratings or "
            f"pricing, so this one is a representative pick rather than a ranked best.")

    if used_fallback:
        parts.append("This is based on general demand in your field rather than "
                     "specific postings you matched.")
    return " ".join(parts)


_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_FORBIDDEN = ("esco", "gap_score", "priority_score", "skill_key", "duplicate_of")


def _numbers(text: str) -> set[str]:
    """Numeric tokens, comma-stripped and trailing-zero-normalized, so '1,919'
    and '1919' — and '4.70' and '4.7' — compare equal."""
    out: set[str] = set()
    for raw in _NUMBER.findall(text or ""):
        token = raw.replace(",", "")
        if "." in token:
            token = token.rstrip("0").rstrip(".")
        out.add(token or "0")
    return out


# Claims the RECORD can adjudicate, as opposed to prose it cannot. Each of these
# was written by the real model on the real gap file while passing a
# numbers-only check — which is how they were found.
_SCARCITY = re.compile(
    r"\b(few|small number|limited (number|options|selection)|only a (few|handful)|"
    r"scarce|not many)\b", re.I)
_FALLBACK_HEDGE = re.compile(r"\bgeneral demand\b", re.I)
# A price claim, which the record settles exactly. `\bfree\b` deliberately does
# NOT match "freeCodeCamp" (no word boundary between "free" and "C"), which is a
# real provider name on this corpus and appears in the fact sheet.
_FREE_CLAIM = re.compile(
    r"\b(free|no cost|costs? nothing|free of charge|zero cost|at no charge)\b", re.I)


def verify_claims(text: str, rec: dict[str, Any], used_fallback: bool) -> Optional[str]:
    """Check the qualitative claims the finalized record can actually settle.

    A numbers-only check passes all of these, and on the first real-model run it
    did — 6 of 8 rationales called a field of 9-38 courses "one of a small number
    of options", and 3 offered the "general demand" hedge on a run where
    ``used_fallback`` was False. Neither is a hallucinated *figure*; both are
    false statements to someone choosing what to study.

    Only claims with a deterministic answer in the record are checked. Vague
    praise ("a solid course") still passes — that remains the honest limit.
    """
    supply = rec.get("supply") or {}
    if _SCARCITY.search(text or "") and not supply.get("thin"):
        return (f"calls the field scarce when {supply.get('courses_available')} courses "
                f"cover the skill")

    if _FALLBACK_HEDGE.search(text or "") and not used_fallback:
        return "offers the general-demand hedge on a run matched to specific postings"

    # "It's free" is the claim most likely to be acted on and the easiest to get
    # wrong, because the fact sheet now tells the model the learner ASKED for free
    # courses. Only `is_free` licenses it: an unpublished price is not evidence of
    # a zero one, which is the same null-is-not-0.0 rule the price column itself
    # follows. The course title is excised first so a course legitimately called
    # "Free Fall Physics" cannot fail its own recommendation.
    price = (rec.get("course", {}).get("quality") or {}).get("price") or {}
    if not price.get("is_free"):
        title = (rec.get("course", {}) or {}).get("title") or ""
        body = re.sub(re.escape(title), " ", text or "", flags=re.I) if title else (text or "")
        if _FREE_CLAIM.search(body):
            stated = "no price is published for it" if price.get("amount") is None \
                else f"it costs {price['amount']}"
            return f"calls the course free when {stated}"

    # An arbitrary pick that reads as a considered one is the exact dishonesty
    # `selection_basis` exists to prevent, so the caveat is mandatory rather than
    # merely offered. The template always states it.
    if (rec.get("selection") or {}).get("basis") == "arbitrary":
        if not re.search(r"\b(equal|equally|representative|nothing to choose|"
                         r"arbitrar|any of|interchangeab)", text or "", re.I):
            return "presents an arbitrary pick as a ranked recommendation"
    return None


def verify_rationale(text: str, fact_sheet: str) -> Optional[str]:
    """Return a reason the rationale must not be published, or None if it holds.

    Agent A verifies every extracted value against the document, Agent B verifies
    the adjudicator's quote, Agent C verifies the skill the model cites. This was
    the one model output in the pipeline published on trust — with a prompt that
    says "never invent course content" and nothing checking whether it did.

    What this catches: **invented specifics**. A model writing "40 hours of video"
    or "taught by 3 industry experts" or "rated 4.9" introduces numbers the fact
    sheet never contained, and those are the fabrications that read as most
    authoritative. Every number in the text must appear in the input.

    What it does NOT catch, stated plainly: unquantified invention ("hands-on
    projects", "beginner friendly"). Numbers are the tractable, high-value half.
    A stricter check would need the course syllabus, which Agent D does not store.
    """
    if not (text or "").strip():
        return "empty"
    invented = _numbers(text) - _numbers(fact_sheet)
    if invented:
        return f"asserts figures absent from the source: {sorted(invented)}"
    lowered = text.lower()
    for token in _FORBIDDEN:
        if token in lowered:
            return f"leaks the internal term {token!r}"
    if len(text) > 700:
        return "far longer than the 2-3 sentences asked for"
    return None


def make_generate_rationale(deps: Deps) -> Callable[[RecommendState], dict]:
    def generate_rationale(state: RecommendState) -> dict:
        from langchain_core.messages import HumanMessage, SystemMessage

        recs = state.get("recommendations", [])
        used_fallback = bool(state.get("used_fallback"))
        warnings: list[str] = []

        for rec in recs:
            # Skipped entirely for no_course_found — there is nothing to explain.
            if rec.get("no_course_found") or not rec.get("course"):
                continue

            human = _render_user_message(rec, used_fallback,
                                        bool(state.get("prefer_free")))
            # The deterministic sentence is the FLOOR, not an error path. Every
            # recommendation gets a real rationale even with no model wired
            # (--no-rationale), where the field used to be published as "" — an
            # empty string that reads as "no reason given".
            text = deterministic_rationale(rec, used_fallback)
            source = "template"

            if deps.llm is not None:
                try:
                    result = deps.llm.invoke([
                        SystemMessage(content=RATIONALE_SYSTEM_PROMPT),
                        HumanMessage(content=human),
                    ])
                    generated = (getattr(result, "content", None) or str(result)).strip()
                except Exception as exc:  # noqa: BLE001 - must not sink the run
                    generated, reason = "", f"{type(exc).__name__}: {exc}"
                else:
                    reason = (verify_rationale(generated, human)
                              or verify_claims(generated, rec, used_fallback))

                if generated and reason is None:
                    text, source = generated, "model"
                else:
                    # The model may phrase; it may never assert. A rationale that
                    # states something its fact sheet does not is discarded whole
                    # and the deterministic sentence stands — the same discipline
                    # as Agent A's dropped fields and Agent C's voided verdicts.
                    warnings.append(
                        f"rationale for '{rec['skill']}' fell back to the template: {reason}")

            rec["course"]["rationale"] = text
            rec["course"]["rationale_source"] = source

        return {"recommendations": recs, "warnings": warnings}

    return generate_rationale


# ===========================================================================
# 6. persist
# ===========================================================================
def _count_sources(recs: list[dict[str, Any]]) -> dict[str, int]:
    """How many rationales the model wrote vs how many fell back to the template.
    A sudden shift toward `template` means the model started asserting things its
    fact sheet did not contain — worth seeing without reading every entry."""
    counts: dict[str, int] = {}
    for r in recs:
        course = r.get("course")
        if not course:
            continue
        key = course.get("rationale_source") or "none"
        counts[key] = counts.get(key, 0) + 1
    return counts


def make_persist(deps: Deps) -> Callable[[RecommendState], dict]:
    def persist(state: RecommendState) -> dict:
        recs = state.get("recommendations", [])
        candidates = state.get("skill_candidates", {})
        arbitrary = [r["skill"] for r in recs
                     if (r.get("selection") or {}).get("basis") == "arbitrary"]
        out = {
            "user_id": state.get("user_id") or "",
            "used_fallback": bool(state.get("used_fallback")),
            "recommendations": recs,
            # additive envelope, per repo convention
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            # An output that cannot be reproduced or re-interpreted from itself is
            # not auditable. Agent C's `calibration` block is the precedent.
            "calibration": {
                # The chain THIS run used, not the configured default: with
                # `prefer_free` the two differ, and the ranking is only
                # reproducible from the one that was actually applied.
                "tiebreak": list(effective_tiebreak(
                    tuple(deps.config.agent_e_tiebreak),
                    prefer_free=bool(state.get("prefer_free")))),
                "prefer_free": bool(state.get("prefer_free")),
                "rating_prior_reviews": deps.config.agent_e_rating_prior_reviews,
                "max_candidates_per_skill": deps.config.agent_e_max_candidates_per_skill,
                "thin_supply_below": deps.config.course_low_confidence_min_courses,
                "candidates_per_skill": {s: len(ids) for s, ids in sorted(candidates.items())},
                # The headline honesty number: how many picks nothing about the
                # courses actually decided.
                "recommendations_by_arbitrary_pick": arbitrary,
                "rationale_sources": _count_sources(recs),
            },
        }
        run_id = state.get("run_id") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        base = Path(state.get("output_dir") or deps.config.output_dir) / run_id
        base.mkdir(parents=True, exist_ok=True)
        path = base / "course_recommendations.json"
        path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"output_path": str(path), "run_calibration": out["calibration"]}

    return persist
