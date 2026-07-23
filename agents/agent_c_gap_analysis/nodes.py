"""Agent C nodes — candidate-to-job gap analysis, arithmetic only.

There is deliberately NO LLM anywhere in this agent, not even fenced: matching
and scoring are cosine comparisons and weighted ratios over data two other
systems already verified (Agent A's grounded profile, Agent B's filtered
tables). An LLM here would add nothing but non-determinism to what is,
end to end, arithmetic.

The three honesty rules the arithmetic follows:

* the **possible_match band** ([0.6, 0.8) similarity) is never resolved in
  either direction — it appears in the denominator of ``gap_score`` (it is a
  real requirement) but never in the numerator (we do not know it is missing);
* **weights** come from ``skill_demand_stats.frequency_count`` (demand as an
  importance proxy — this system has no essential/optional tags and does not
  invent them), with a **floor of 1** so a skill absent from the stats still
  exists in the score instead of silently vanishing;
* the **fallback sector is never guessed**: modal sector of the retrieved
  postings, or the operator's ``--sector``, or the fallback is skipped with a
  warning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from shared.config import Config
from shared.contracts import load_profile
from shared.job_market import export_for_agent_c, map_skills_to_esco

from .state import GapState

SCHEMA_VERSION = "itqan.skill_gap/1.0"

# Below this many usable postings, per-job matching says more about retrieval
# luck than about the market, and the aggregated stats are the honest basis.
MIN_USABLE_POSTINGS = 5

GAP_SCORE_FORMULA = (
    "sum(weight of missing) / sum(weight of matched + missing + possible_match); "
    "weight = frequency_count in the latest skill_demand_stats window "
    "(by esco_code, else by skill_key), floor 1"
)


@dataclass
class Deps:
    """Injected once; nodes never construct their own I/O. This is what lets the
    offline tests drive the whole graph with fakes, exactly like Agent B's
    GraphDeps."""

    config: Config
    embedder: Any
    exporter: Callable[..., dict[str, list[Any]]] = export_for_agent_c
    mapper: Callable[..., list[Any]] = map_skills_to_esco


# ---------------------------------------------------------------------------
# 1. build_query_embedding
# ---------------------------------------------------------------------------
def make_build_query_embedding(deps: Deps) -> Callable[[GapState], dict]:
    def build_query_embedding(state: GapState) -> dict:
        profile = load_profile(state["profile_path"])
        warnings: list[str] = []

        skills = [
            s.get("name", "").strip()
            for s in profile.skills.get("accepted", [])
            if s.get("name", "").strip()
        ]
        if not skills:
            warnings.append(
                "profile has no accepted skills; retrieval will rest on the headline alone"
            )

        # The SAME essence shape Agent B embeds postings with (title line,
        # scalars line, skills line) — same shape and same embedder is what
        # makes candidate-vs-posting similarity a meaningful number at all.
        # No seniority line: the profile schema does not state one, and
        # inferring it would be a guess feeding a similarity score.
        headline = (profile.summary or {}).get("headline", "").strip()
        location = ((profile.candidate or {}).get("contact") or {}).get("location") or ""
        parts = [headline]
        if location.strip():
            parts.append(location.strip())
        if skills:
            parts.append("skills: " + ", ".join(skills))
        essence = "\n".join(p for p in parts if p)

        if not essence:
            raise ValueError(
                f"{state['profile_path']}: nothing to embed — no headline, no "
                "location, no accepted skills"
            )

        return {
            "profile": profile,
            "candidate_skills": skills,
            "essence": essence,
            "query_embedding": list(deps.embedder.embed_query(essence)),
            "warnings": warnings,
        }

    return build_query_embedding


# ---------------------------------------------------------------------------
# 2. retrieve_postings
# ---------------------------------------------------------------------------
def make_retrieve_postings(deps: Deps) -> Callable[[GapState], dict]:
    def retrieve_postings(state: GapState) -> dict:
        config = deps.config
        out = deps.exporter(
            state["query_embedding"],
            top_k=state.get("top_k") or 15,
            config=config,
        )
        postings = out["job_postings"]
        stats = out["skill_demand_stats"]

        usable = [p for p in postings if p.similarity >= config.agent_c_match_threshold]
        used_fallback = len(usable) < MIN_USABLE_POSTINGS

        warnings: list[str] = []
        sector = state.get("sector_override")
        if sector is None:
            # Modal sector over ALL retrieved rows (even sub-threshold ones):
            # deterministic, and derived from data rather than assumption. Ties
            # break toward the lower sector code for reproducibility.
            counts: dict[str, int] = {}
            for p in postings:
                if p.sector:
                    counts[p.sector] = counts.get(p.sector, 0) + 1
            if counts:
                sector = min(sorted(counts), key=lambda s: (-counts[s], s))
        if used_fallback and sector is None:
            warnings.append(
                "fallback needed but no sector could be inferred (zero retrievals, "
                "no --sector); sector-level gap analysis skipped rather than guessed"
            )

        return {
            "postings": postings,
            "stats": stats,
            "usable_postings": usable,
            "used_fallback": used_fallback,
            "inferred_sector": sector,
            "warnings": warnings,
        }

    return retrieve_postings


# ---------------------------------------------------------------------------
# 3. map_candidate_skills
# ---------------------------------------------------------------------------
def make_map_candidate_skills(deps: Deps) -> Callable[[GapState], dict]:
    def map_candidate_skills(state: GapState) -> dict:
        mappings = deps.mapper(
            state.get("candidate_skills", []),
            embedder=deps.embedder,
            config=deps.config,
        )
        return {"candidate_mappings": mappings}

    return map_candidate_skills


# ---------------------------------------------------------------------------
# 4. gap_analysis
# ---------------------------------------------------------------------------
def make_gap_analysis(deps: Deps) -> Callable[[GapState], dict]:
    def gap_analysis(state: GapState) -> dict:
        config = deps.config
        stats = state.get("stats", [])
        mappings = state.get("candidate_mappings", [])
        candidate_skills = state.get("candidate_skills", [])

        candidate_esco = {m.esco_uri for m in mappings if m.esco_uri}

        # Demand-stat lookups (latest window, served by the export):
        #   skill_key -> esco_code, for resolving a JOB skill to the vocabulary
        #   esco/skill_key -> summed frequency, for weights
        key_to_esco: dict[str, Optional[str]] = {}
        weight_by_esco: dict[str, int] = {}
        weight_by_key: dict[str, int] = {}
        for row in stats:
            key_to_esco.setdefault(row.skill_key, row.esco_code)
            if row.esco_code:
                weight_by_esco[row.esco_code] = (
                    weight_by_esco.get(row.esco_code, 0) + row.frequency_count
                )
            weight_by_key[row.skill_key] = (
                weight_by_key.get(row.skill_key, 0) + row.frequency_count
            )

        def weight(skill_key: str, esco: Optional[str]) -> int:
            # Floor 1: a skill nobody has aggregated yet still exists. Without
            # the floor, un-aggregated skills would vanish from gap_score.
            if esco and esco in weight_by_esco:
                return max(1, weight_by_esco[esco])
            return max(1, weight_by_key.get(skill_key, 0))

        # One embedding batch for every phrase the banding will compare —
        # candidate skills, every usable job's skills, and (for fallback) the
        # sector's stat skills. Cached; a phrase is embedded exactly once.
        phrases: set[str] = {s.lower() for s in candidate_skills}
        for job in state.get("usable_postings", []):
            phrases.update(s.lower() for s in job.required_skills)
        sector = state.get("inferred_sector")
        # The noise floor: a phrase aggregated once is not sector demand, and
        # comparing the candidate against every such phrase saturates the
        # fallback gap at ~1.0 (measured live: 463 sector rows, ~87% freq-1).
        sector_rows = (
            [r for r in stats
             if r.sector == sector
             and r.frequency_count >= config.agent_c_fallback_min_freq]
            if sector else []
        )
        if state.get("used_fallback"):
            phrases.update(r.skill_key for r in sector_rows)
        vectors = _embed_phrases(deps, sorted(phrases))

        def band(job_phrase: str) -> str:
            """matched / possible_match / missing by best cosine against the
            candidate's skills. The middle band is a refusal to decide, and it
            stays one."""
            best = max(
                (
                    _dot(vectors[job_phrase.lower()], vectors[c.lower()])
                    for c in candidate_skills
                ),
                default=0.0,
            )
            if best >= config.agent_c_skill_match:
                return "matched"
            if best >= config.agent_c_skill_possible:
                return "possible_match"
            return "missing"

        def classify(skill: str, esco: Optional[str]) -> str:
            if esco and esco in candidate_esco:
                return "matched"
            return band(skill)

        def score(entries: list[tuple[str, Optional[str], str]]) -> float:
            total = sum(weight(k, e) for k, e, _ in entries)
            miss = sum(weight(k, e) for k, e, verdict in entries if verdict == "missing")
            return round(miss / total, 4) if total else 0.0

        # ---- direct path -------------------------------------------------
        matched_jobs: list[dict[str, Any]] = []
        if not state.get("used_fallback"):
            for job in state.get("usable_postings", []):
                entries = []
                for skill in job.required_skills:
                    key = skill.lower()
                    esco = key_to_esco.get(key)
                    entries.append((key, esco, classify(skill, esco)))
                buckets: dict[str, list[str]] = {"matched": [], "possible_match": [], "missing": []}
                for (key, _e, verdict), skill in zip(entries, job.required_skills):
                    buckets[verdict].append(skill)
                matched_jobs.append({
                    "job_id": job.posting_id,
                    "job_title": job.title,
                    "similarity": round(job.similarity, 4),
                    "matched_skills": buckets["matched"],
                    "missing_skills": buckets["missing"],
                    "possible_match_skills": buckets["possible_match"],
                    "gap_score": score(entries),
                })

        # ---- fallback path ----------------------------------------------
        fallback_gap: Optional[dict[str, Any]] = None
        if state.get("used_fallback") and sector is not None:
            entries = [
                (row.skill_key, row.esco_code, classify(row.skill_key, row.esco_code))
                for row in sector_rows
            ]
            buckets = {"matched": [], "possible_match": [], "missing": []}
            for (key, _e, verdict), row in zip(entries, sector_rows):
                buckets[verdict].append(row.skill)
            fallback_gap = {
                "sector": sector,
                "matched_skills": buckets["matched"],
                "missing_skills": buckets["missing"],
                "possible_match_skills": buckets["possible_match"],
                "gap_score": score(entries),
            }

        # ---- aggregate ---------------------------------------------------
        missing_weight: dict[str, int] = {}
        for job in matched_jobs:
            for skill in job["missing_skills"]:
                key = skill.lower()
                missing_weight[key] = missing_weight.get(key, 0) + weight(
                    key, key_to_esco.get(key)
                )
        if fallback_gap:
            for skill in fallback_gap["missing_skills"]:
                key = skill.lower()
                missing_weight[key] = missing_weight.get(key, 0) + weight(
                    key, key_to_esco.get(key)
                )
        top_missing = [
            k for k, _w in sorted(missing_weight.items(), key=lambda kv: (-kv[1], kv[0]))
        ][:10]

        scores = [j["gap_score"] for j in matched_jobs]
        if fallback_gap:
            scores.append(fallback_gap["gap_score"])
        aggregate = {
            "most_common_missing_skills": top_missing,
            "average_gap_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        }

        return {
            "matched_jobs": matched_jobs,
            "fallback_sector_gap": fallback_gap,
            "aggregate": aggregate,
        }

    def _embed_phrases(deps: Deps, phrases: list[str]) -> dict[str, list[float]]:
        if not phrases:
            return {}
        vectors = deps.embedder.embed_documents(phrases)
        return dict(zip(phrases, vectors))

    return gap_analysis


def _dot(a: list[float], b: list[float]) -> float:
    """Vectors from both embedders are unit-normalised, so this IS cosine."""
    return sum(x * y for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# 5. persist
# ---------------------------------------------------------------------------
def make_persist(deps: Deps) -> Callable[[GapState], dict]:
    def persist(state: GapState) -> dict:
        import json
        from pathlib import Path

        out = {
            # ---- the requested contract, verbatim keys -------------------
            "user_id": state.get("user_id") or getattr(state.get("profile"), "run_id", ""),
            "used_fallback": bool(state.get("used_fallback")),
            "matched_jobs": state.get("matched_jobs", []),
            "fallback_sector_gap": state.get("fallback_sector_gap"),
            "aggregate": state.get("aggregate", {}),
            # ---- additive envelope, per repo convention ------------------
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "gap_score_formula": GAP_SCORE_FORMULA,
            "retrieval": {
                "threshold": deps.config.agent_c_match_threshold,
                "retrieved": len(state.get("postings", [])),
                "usable": len(state.get("usable_postings", [])),
                "similarities": [round(p.similarity, 4) for p in state.get("postings", [])],
                "inferred_sector": state.get("inferred_sector"),
            },
            # The user-decided home for candidate-side ESCO evidence,
            # including near-miss scores for unmapped skills.
            "candidate_skill_mappings": [
                m.model_dump() for m in state.get("candidate_mappings", [])
            ],
            "warnings": state.get("warnings", []),
        }

        run_id = state.get("run_id") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        base = Path(state.get("output_dir") or deps.config.output_dir) / run_id
        base.mkdir(parents=True, exist_ok=True)
        path = base / "skill_gap.json"
        path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"output_path": str(path)}

    return persist
