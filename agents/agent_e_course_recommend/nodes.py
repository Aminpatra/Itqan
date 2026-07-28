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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from shared.config import Config
from shared.course_market import courses_for_skills

from .state import RecommendState

SCHEMA_VERSION = "itqan.course_reco/1.0"

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
    "- If used_fallback is true, use softer confidence language (\"a solid option "
    "based on general demand in your field\") rather than confident job-specific "
    "claims, since this came from aggregate demand stats, not matched postings.\n"
    "- If 'Courses available' is low, say plainly that few courses cover this "
    "skill, so this is one of a small number of options — do not dress a thin "
    "field up as a strong endorsement of the single course found.\n"
    "- 2-3 sentences, plain and direct, no marketing language.\n"
    "- Never mention priority_score as a number, ESCO codes, gap_score, or "
    "internal pipeline terms — translate priority_bucket into plain language if "
    "referenced at all (e.g. \"a skill that came up often in roles you'd be a "
    "good fit for\").\n"
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
        data = json.loads(Path(state["gap_path"]).read_text(encoding="utf-8"))
        aggregate = data.get("aggregate") or {}
        used_fallback = bool(data.get("used_fallback"))
        warnings: list[str] = []

        details = aggregate.get("missing_skill_details")
        missing: list[dict[str, Any]] = []
        seen: set[str] = set()
        if details:
            # The enriched Agent C aggregate: each skill carries its esco_code and
            # inherited priority_score. Already deduped upstream; dedup again
            # defensively (a skill triggers exactly one course search).
            for d in details:
                key = (d.get("skill") or "").strip().lower()
                if not key or key in seen:
                    continue
                seen.add(key)
                missing.append({
                    "skill": key,
                    "esco_code": d.get("esco_code"),
                    "priority_score": float(d.get("priority_score") or 0.0),
                })
        else:
            # Older gap file, before missing_skill_details existed: fall back to
            # the bare skill list — no ESCO codes, no weights — and say so.
            for s in aggregate.get("most_common_missing_skills") or []:
                key = (s or "").strip().lower()
                if not key or key in seen:
                    continue
                seen.add(key)
                missing.append({"skill": key, "esco_code": None, "priority_score": 1.0})
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
            "user_id": state.get("user_id") or data.get("user_id") or "",
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


def _field_key(cc: Any, field_name: str) -> float:
    """One tie-break field as an ascending, smaller-is-better number. A null
    always yields +inf, so it sorts LAST in every field and is NEVER treated as
    0 (a 0 rating is a real, distinct value that would sometimes win)."""
    if field_name == "rating":
        return -cc.rating if cc.rating is not None else _INF
    if field_name == "review_count":
        return -cc.review_count if cc.review_count is not None else _INF
    if field_name == "last_updated":
        e = _epoch(cc.last_updated)
        return -e if e is not None else _INF          # more recent = smaller = better
    if field_name == "price":
        amt = cc.price.amount if (cc.price and cc.price.amount is not None) else None
        return amt if amt is not None else _INF        # cheaper = smaller = better; free(0) wins
    raise ValueError(f"unknown agent_e_tiebreak field {field_name!r}")


def greedy_assign(
    missing: list[dict[str, Any]],
    skill_candidates: dict[str, list[str]],
    courses_by_id: dict[str, Any],
    tiebreak: tuple[str, ...],
) -> tuple[dict[str, list[str]], list[str]]:
    """Weighted greedy set-cover. Returns ``(assigned, no_course_found)`` where
    ``assigned`` maps a course_id to the skills it was chosen to cover (each skill
    covered exactly once, each course appearing once)."""
    priority = {m["skill"]: m["priority_score"] for m in missing}

    course_can_cover: dict[str, set[str]] = {}
    for skill, ids in skill_candidates.items():
        for cid in ids:
            course_can_cover.setdefault(cid, set()).add(skill)

    uncovered = {s for s, ids in skill_candidates.items() if ids}
    no_course_found = sorted(s for s, ids in skill_candidates.items() if not ids)

    assigned: dict[str, list[str]] = {}
    while uncovered:
        candidate_ids = {cid for s in uncovered for cid in skill_candidates[s]}

        def objective(cid: str) -> tuple:
            covers = course_can_cover[cid] & uncovered
            n = len(covers)
            weight = sum(priority[s] for s in covers)
            # min() picks the smallest key: most skills first (-n), then most
            # summed priority (-weight), then the configured tie-break fields,
            # then the lowest course_id for total determinism.
            cc = courses_by_id[cid]
            return (-n, -weight, *(_field_key(cc, f) for f in tiebreak), cid)

        best = min(candidate_ids, key=objective)
        newly = sorted(course_can_cover[best] & uncovered)
        assigned[best] = newly
        uncovered.difference_update(newly)

    return assigned, no_course_found


def make_greedy_cover_assign(deps: Deps) -> Callable[[RecommendState], dict]:
    def greedy_cover_assign(state: RecommendState) -> dict:
        assigned, no_course = greedy_assign(
            state.get("missing", []),
            state.get("skill_candidates", {}),
            state.get("courses_by_id", {}),
            tuple(deps.config.agent_e_tiebreak),
        )
        return {"assigned": assigned, "no_course_found_skills": no_course}

    return greedy_cover_assign


# ===========================================================================
# 4. attach_flags  (pure)
# ===========================================================================
def _compute_buckets(missing: list[dict[str, Any]]) -> dict[str, str]:
    """priority_bucket by splitting THIS candidate's priority_scores into thirds
    of their own [min, max] range — high/moderate/some. Computed in code so a raw
    priority float never reaches the LLM."""
    scores = [m["priority_score"] for m in missing]
    if not scores:
        return {}
    lo, hi = min(scores), max(scores)
    span = hi - lo
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
        buckets = _compute_buckets(missing)
        courses_by_id = state.get("courses_by_id", {})
        candidates = state.get("skill_candidates", {})

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
                "priority_bucket": buckets.get(primary, "some"),
                "supply": supply_for(primary),
                "course": {
                    "course_id": cc.course_id,
                    "title": cc.title,
                    "provider": cc.provider,
                    "url": cc.url,
                    "covers_other_skills": others,
                    "quality": {
                        "rating": cc.rating,
                        "review_count": cc.review_count,
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
def _render_user_message(rec: dict[str, Any], used_fallback: bool) -> str:
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
    supply_line = "not available" if available is None else str(available)
    if supply.get("thin"):
        supply_line += " (few courses cover this skill)"

    return "\n".join([
        f"Skill: {rec['skill']}",
        f"Also covers: {also}",
        f"Course: {course['title']} ({course['provider'] or 'the provider'})",
        f"Rating: {rating}",
        f"Reviews: {reviews}",
        f"Price: {price_line}",
        f"Last updated: {last_updated}",
        f"Demand level for this skill: {rec['priority_bucket']}",
        f"Courses available for this skill: {supply_line}",
        f"Recommendation basis: {basis}",
    ])


def make_generate_rationale(deps: Deps) -> Callable[[RecommendState], dict]:
    def generate_rationale(state: RecommendState) -> dict:
        from langchain_core.messages import HumanMessage, SystemMessage

        recs = state.get("recommendations", [])
        # No model wired (--no-rationale): run selection only, leave rationales
        # empty. This is a deliberate mode, not a failure — no warnings.
        if deps.llm is None:
            return {}

        used_fallback = bool(state.get("used_fallback"))
        warnings: list[str] = []

        for rec in recs:
            # Skipped entirely for no_course_found — there is nothing to explain.
            if rec.get("no_course_found") or not rec.get("course"):
                continue
            human = _render_user_message(rec, used_fallback)
            try:
                result = deps.llm.invoke([
                    SystemMessage(content=RATIONALE_SYSTEM_PROMPT),
                    HumanMessage(content=human),
                ])
                text = (getattr(result, "content", None) or str(result)).strip()
            except Exception as exc:  # noqa: BLE001 - a failed rationale must not sink the run
                text = ""
                warnings.append(f"rationale generation failed for '{rec['skill']}': {exc}")
            rec["course"]["rationale"] = text

        return {"recommendations": recs, "warnings": warnings}

    return generate_rationale


# ===========================================================================
# 6. persist
# ===========================================================================
def make_persist(deps: Deps) -> Callable[[RecommendState], dict]:
    def persist(state: RecommendState) -> dict:
        out = {
            "user_id": state.get("user_id") or "",
            "used_fallback": bool(state.get("used_fallback")),
            "recommendations": state.get("recommendations", []),
            # additive envelope, per repo convention
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        run_id = state.get("run_id") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        base = Path(state.get("output_dir") or deps.config.output_dir) / run_id
        base.mkdir(parents=True, exist_ok=True)
        path = base / "course_recommendations.json"
        path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"output_path": str(path)}

    return persist
