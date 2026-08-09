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

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from shared.config import Config
from shared.contracts import SkillGap, load_profile
from shared.job_market import export_for_agent_c, map_skills_to_esco

from .skill_resolver import resolve_skills
from .state import GapState

SCHEMA_VERSION = "itqan.skill_gap/1.0"

# Below this many usable postings, per-job matching says more about retrieval
# luck than about the market, and the aggregated stats are the honest basis.
GAP_SCORE_FORMULA = (
    "sum(weight of missing) / sum(weight of matched + missing + possible_match); "
    "weight = log1p(frequency_count in the latest skill_demand_stats window, "
    "scoped to THIS JOB'S sector, by esco_code else by skill_key, floor 1). "
    "null when the posting listed no parsable requirements — never 0.0, which "
    "would read as a perfect fit. gap_score_range gives [lower, upper] where the "
    "upper bound counts every unresolved possible_match as missing."
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
    # The ONE model in this agent, and it is optional: absent (or with
    # agent_c_llm_matching off) the whole agent stays deterministic, exactly as
    # it was. Injected like every other dependency so offline tests never reach a
    # network.
    llm: Any = None

    def resolve_skills(self, **kwargs: Any) -> dict[str, dict[str, Any]]:
        return resolve_skills(llm=self.llm, **kwargs)


# ---------------------------------------------------------------------------
# 1. build_query_embedding
# ---------------------------------------------------------------------------
def make_build_query_embedding(deps: Deps) -> Callable[[GapState], dict]:
    def build_query_embedding(state: GapState) -> dict:
        profile = load_profile(state["profile_path"])
        warnings: list[str] = []

        accepted = [
            s for s in profile.skills.get("accepted", [])
            if isinstance(s, dict) and s.get("name", "").strip()
        ]
        skills = [s["name"].strip() for s in accepted]
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

        # The candidate's own answers, from the onboarding questions. The role
        # goes in the TITLE slot deliberately: that is the slot Agent B fills
        # with a posting's job title, so this is the one place a stated
        # preference can genuinely change which postings come back — rather
        # than being recorded somewhere and ignored, which is what it was.
        role = (state.get("preferred_role") or "").strip()
        roles_only = bool(state.get("roles_only"))
        arrangement = (state.get("preferred_arrangement") or "").strip()

        if role and roles_only:
            # "Not open to other roles": the CV headline is what the candidate
            # HAS been; the role is what they want to be compared against, and
            # they have said the two should not be pooled.
            title = role
        elif role:
            title = f"{headline} / {role}" if headline else role
        else:
            title = headline

        parts = [title]
        if location.strip():
            parts.append(location.strip())
        if arrangement:
            # A bias in the embedded text, and nothing more. `job_postings` has
            # no work-arrangement column, so there is nothing to filter on;
            # `calibration.preferences.arrangement_applied` records that this is
            # retrieval bias so no consumer can read it as a guarantee.
            parts.append(f"preferred work arrangement: {arrangement}")
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
            # The full records, not just names. Agent A publishes quality,
            # evidence_type and origin per skill and Agent C was discarding every
            # one of them — so a skill the candidate merely claimed cancelled a
            # requirement as forcefully as one demonstrated in a project.
            "candidate_skill_records": accepted,
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
        # "Too thin for a stable market read" — it no longer means "discard the
        # per-job results", which conflated a statistical-stability question with
        # an evidence question and threw away the best evidence available.
        used_fallback = len(usable) < config.agent_c_min_usable_postings

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
        warnings: list[str] = []

        # Candidate skills whose evidence is too weak to CANCEL a requirement.
        # Agent A publishes quality/evidence_type/origin per skill and Agent C
        # was discarding all of it, so an unverified `claim_only` claim closed a
        # gap exactly as forcefully as a project-evidenced skill. The error is
        # asymmetric in the harmful direction — it shrinks the gap — so weak
        # evidence is capped at `possible_match` and can never read `matched`.
        weak_skills = {
            (s.get("name") or "").strip().lower()
            for s in (state.get("candidate_skill_records") or [])
            if s.get("quality") == "low"
            or s.get("evidence_type") in {"claim_only", "adjacent"}
        }

        # Demand-stat lookups (latest window, served by the export). Weights are
        # SECTOR-SCOPED: summing a skill's frequency across all nine ISCO groups
        # made "professional communication" weigh 95 when its largest single
        # sector is 58 — a nationwide generic phrase outranking a candidate's
        # real sector-specific gaps, and the distortion pulls every candidate
        # toward the largest sector regardless of their own.
        key_to_esco: dict[str, Optional[str]] = {}
        w_sector_esco: dict[tuple[str, str], int] = {}
        w_sector_key: dict[tuple[str, str], int] = {}
        w_any_esco: dict[str, int] = {}
        w_any_key: dict[str, int] = {}
        sector_volume: dict[str, int] = {}
        low_conf_sectors: set[str] = set()
        for row in stats:
            # Never let a NULL overwrite a real code (setdefault alone would pin
            # the first row seen, even if a later row carries the mapping).
            if row.esco_code and not key_to_esco.get(row.skill_key):
                key_to_esco[row.skill_key] = row.esco_code
            key_to_esco.setdefault(row.skill_key, row.esco_code)

            if row.esco_code:
                k = (row.sector, row.esco_code)
                w_sector_esco[k] = w_sector_esco.get(k, 0) + row.frequency_count
                w_any_esco[row.esco_code] = w_any_esco.get(row.esco_code, 0) + row.frequency_count
            k2 = (row.sector, row.skill_key)
            w_sector_key[k2] = w_sector_key.get(k2, 0) + row.frequency_count
            w_any_key[row.skill_key] = w_any_key.get(row.skill_key, 0) + row.frequency_count

            if row.sector_volume:
                sector_volume[row.sector] = max(
                    sector_volume.get(row.sector, 0), row.sector_volume
                )
            if row.low_confidence:
                low_conf_sectors.add(row.sector)

        def raw_demand(skill_key: str, esco: Optional[str], sector: Optional[str]) -> int:
            """This skill's demand IN THIS JOB'S SECTOR, falling back to the
            nationwide total only when the sector has no row for it."""
            if sector:
                if esco and (sector, esco) in w_sector_esco:
                    return w_sector_esco[(sector, esco)]
                if (sector, skill_key) in w_sector_key:
                    return w_sector_key[(sector, skill_key)]
            if esco and esco in w_any_esco:
                return w_any_esco[esco]
            return w_any_key.get(skill_key, 0)

        def weight(skill_key: str, esco: Optional[str], sector: Optional[str] = None) -> float:
            # Floor 1: a skill nobody has aggregated yet still exists. Without
            # the floor, un-aggregated skills would vanish from gap_score.
            #
            # log1p damping: raw counts span 1..95 here, so one boilerplate
            # phrase outweighed two dozen specific skills and dictated the whole
            # ranking. Damping preserves the ORDER of demand while stopping the
            # mode from being tyrannical.
            return math.log1p(max(1, raw_demand(skill_key, esco, sector)))

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

        def nearest(job_phrase: str) -> tuple[float, Optional[str]]:
            """Best cosine against the candidate's skills, and which skill it was."""
            best, who = 0.0, None
            target = vectors.get(job_phrase.lower())
            if target is None:
                return 0.0, None
            for c in candidate_skills:
                vec = vectors.get(c.lower())
                if vec is None:
                    continue
                sim = _dot(target, vec)
                if sim > best:
                    best, who = sim, c
            return best, who

        def classify(skill: str, esco: Optional[str]) -> dict[str, Any]:
            """Resolve one requirement, recording HOW it was resolved.

            Tiers, strongest first. Each is deterministic; the LLM tier that runs
            afterwards only ever looks at what these could not settle.
            """
            key = skill.strip().lower()
            best, who = nearest(skill)
            out = {
                "skill": skill, "key": key, "esco": esco,
                "best_similarity": round(best, 4), "nearest_candidate_skill": who,
            }

            # 1. Same ESCO concept — the shared vocabulary, strongest evidence.
            if esco and esco in candidate_esco:
                by_esco = next(
                    (m.skill for m in mappings if m.esco_uri == esco), who
                )
                return {**out, "verdict": "matched", "resolved_by": "esco",
                        "satisfied_by": by_esco}

            # 2. Exact normalised string equality.
            for c in candidate_skills:
                if c.strip().lower() == key:
                    return {**out, "verdict": "matched", "resolved_by": "exact",
                            "satisfied_by": c}

            # 3. Asymmetric token containment: the candidate's phrase contains the
            #    requirement as whole tokens ("data analytics engineering" satisfies
            #    "data analytics"). Cosine cannot express this direction at all.
            for c in candidate_skills:
                if _contains_tokens(c, key):
                    return {**out, "verdict": "matched", "resolved_by": "containment",
                            "satisfied_by": c}

            # 4. Cosine bands. Measured: at 0.80 this is an exact-string detector
            #    (zero non-identical matches on the live run), so it is kept as a
            #    conservative confirmer, not relied on to find synonyms.
            if best >= config.agent_c_skill_match:
                return {**out, "verdict": "matched", "resolved_by": "cosine",
                        "satisfied_by": who}
            if best >= config.agent_c_skill_possible:
                return {**out, "verdict": "possible_match", "resolved_by": "cosine",
                        "satisfied_by": who}
            return {**out, "verdict": "missing", "resolved_by": "cosine",
                    "satisfied_by": None}

        def cap_weak_evidence(entry: dict[str, Any]) -> dict[str, Any]:
            """A requirement cancelled by a weak claim is not `matched`.

            An unverified claim_only/low/adjacent skill closing a gap outright is
            the asymmetric error: it makes the candidate look readier than the
            evidence supports. Capped at possible_match so the uncertainty stays
            visible instead of being silently resolved in their favour.
            """
            who = (entry.get("satisfied_by") or "").strip().lower()
            if entry["verdict"] == "matched" and who in weak_skills:
                return {**entry, "verdict": "possible_match", "weak_evidence": True}
            return entry

        def score(entries: list[dict[str, Any]]) -> tuple[Optional[float], Optional[list]]:
            """(gap_score, [lower, upper]) — or (None, None) when undefined.

            None, never 0.0. Returning the BEST value on the scale for "this
            posting listed nothing we could parse" was a false claim of a perfect
            fit: 5 of 15 jobs on the live run scored 0.0 having matched nothing,
            two of them listing no requirements at all.

            The interval is what `possible_match` honestly implies. Putting it in
            the denominator only is not neutral — it quietly decides "not
            missing", and every unresolved requirement therefore lowers the score.
            The bounds say plainly: at best `lower`, at worst `upper`.
            """
            if not entries:
                return None, None
            total = sum(e["weight"] for e in entries)
            if total <= 0:
                return None, None
            miss = sum(e["weight"] for e in entries if e["verdict"] == "missing")
            poss = sum(e["weight"] for e in entries if e["verdict"] == "possible_match")
            # Everything unresolved -> the point estimate is arithmetically 0.0 but
            # means nothing: not one requirement was settled either way. Publishing
            # 0.0 there reads as "perfect fit" at a glance, which is the same lie
            # the empty-denominator case told, so it gets the same answer: null,
            # with the [0, 1] range saying exactly how little is known.
            if poss >= total:
                return None, [0.0, 1.0]
            return round(miss / total, 4), [
                round(miss / total, 4), round((miss + poss) / total, 4)
            ]

        # ---- classify every requirement, then let the LLM tier settle what the
        # deterministic tiers could not -------------------------------------
        # matched_jobs is built for whatever cleared the retrieval bar, WHETHER OR
        # NOT the fallback also runs. Suppressing it under `used_fallback` threw
        # away the per-job evidence entirely: four excellent matches at 0.85
        # produced an empty list while five mediocre ones became "the market".
        # "Too few for a stable average" and "not worth showing" are different
        # claims, and only the first was ever true.
        job_entries: list[tuple[Any, list[dict[str, Any]]]] = []
        for job in state.get("usable_postings", []):
            entries = []
            for skill in job.required_skills:
                key = skill.strip().lower()
                entries.append(classify(skill, key_to_esco.get(key)))
            job_entries.append((job, entries))

        fallback_entries: list[dict[str, Any]] = []
        if state.get("used_fallback") and sector is not None:
            fallback_entries = [
                classify(row.skill, row.esco_code) for row in sector_rows
            ]

        all_entries = [e for _job, es in job_entries for e in es] + fallback_entries
        resolutions = deps.resolve_skills(
            unresolved=[e for e in all_entries if e["verdict"] != "matched"],
            candidate_skills=candidate_skills,
            skill_records=state.get("candidate_skill_records") or [],
            config=config,
        )
        for entry in all_entries:
            verdict = resolutions.get(entry["key"])
            if verdict is not None:
                entry.update(verdict)
        # Weak evidence is capped AFTER resolution, so neither tier can promote an
        # unverified claim to a confident match.
        for i, entry in enumerate(all_entries):
            all_entries[i] = cap_weak_evidence(entry)
        by_key = {e["key"]: e for e in all_entries}
        job_entries = [
            (job, [by_key.get(e["key"], e) for e in es]) for job, es in job_entries
        ]
        fallback_entries = [by_key.get(e["key"], e) for e in fallback_entries]

        def _bucket(entries: list[dict[str, Any]]) -> dict[str, list[str]]:
            out: dict[str, list[str]] = {"matched": [], "possible_match": [], "missing": []}
            for e in entries:
                out[e["verdict"]].append(e["skill"])
            return out

        # ---- direct path -------------------------------------------------
        matched_jobs: list[dict[str, Any]] = []
        for job, entries in job_entries:
            for e in entries:
                e["weight"] = weight(e["key"], e["esco"], job.sector)
            gap, interval = score(entries)
            buckets = _bucket(entries)
            matched_jobs.append({
                "job_id": job.posting_id,
                "job_title": job.title,
                # A report whose purpose is "roles you nearly fit" was unusable
                # without a way to open the job: all four of these were retrieved
                # and discarded.
                "source_url": job.source_url,
                # The employer's own page, when Agent B could resolve it. Kept
                # ALONGSIDE `source_url` rather than replacing it: the consumer
                # decides which to link, and "where we found it" stays auditable.
                # None on every posting ingested before Agent B's migration 0011.
                "final_url": job.final_url,
                "posted_date": job.posted_date,
                # Stated by the employer or not stated at all. A consumer may
                # show these; nothing may default them — an absent salary is
                # unknown pay, not free labour, and an absent arrangement is not
                # a claim that the role is onsite.
                "work_arrangement": job.work_arrangement,
                "employment_type": job.employment_type,
                "salary": None if job.salary_min is None and job.salary_max is None else {
                    "min": job.salary_min,
                    "max": job.salary_max,
                    "currency": job.salary_currency,
                    "period": job.salary_period,
                },
                # The employer. Agent B extracted and grounded it from the start
                # but had no column to keep it in, so every consumer saw None;
                # persisted from Agent B's migration 0010. A job card without an
                # employer is not something anyone acts on.
                "company": job.company,
                "source": job.source,
                # How the publisher must be CREDITED, and the terms that require
                # it. Carried rather than derived: a consumer showing a job card
                # has to name GulfTalent because GulfTalent's terms make that a
                # condition of our having the row at all.
                "attribution": job.attribution,
                "terms_url": job.terms_url,
                "seniority_level": job.seniority_level,
                "location": job.location,
                "similarity": round(job.similarity, 4),
                "matched_skills": buckets["matched"],
                "missing_skills": buckets["missing"],
                "possible_match_skills": buckets["possible_match"],
                "gap_score": gap,
                "gap_score_range": interval,
                # Says plainly WHY the number is absent, instead of implying a
                # perfect fit.
                "insufficient_data": gap is None,
                "low_confidence_demand": job.sector in low_conf_sectors,
                "skill_resolution": [
                    {k: e[k] for k in ("skill", "verdict", "resolved_by",
                                       "satisfied_by", "best_similarity")}
                    for e in entries
                ],
            })

        # ---- fallback path ----------------------------------------------
        fallback_gap: Optional[dict[str, Any]] = None
        if state.get("used_fallback") and sector is not None:
            for e in fallback_entries:
                e["weight"] = weight(e["key"], e["esco"], sector)
            gap, interval = score(fallback_entries)
            buckets = _bucket(fallback_entries)
            if gap is None:
                warnings.append(
                    f"sector {sector} had no demand rows above the floor, so no "
                    f"sector-level gap could be computed (reported as null, not zero)."
                )
            fallback_gap = {
                "sector": sector,
                "matched_skills": buckets["matched"],
                "missing_skills": buckets["missing"],
                "possible_match_skills": buckets["possible_match"],
                # Renamed from gap_score: this answers a DIFFERENT question from
                # the per-job scores ("what fraction of everything demanded in this
                # ISCO major group do you lack?"), so it must not be pooled with
                # them or read as comparable.
                "sector_coverage_gap": gap,
                "sector_coverage_range": interval,
                "insufficient_data": gap is None,
                "sector_volume": sector_volume.get(sector),
                "low_confidence": sector in low_conf_sectors,
                "skills_considered": len(fallback_entries),
            }

        # ---- aggregate ---------------------------------------------------
        # Demand and occurrence are SEPARATE facts. Adding the market-wide
        # frequency once per job produced freq x job_count — a squared prevalence
        # term, part of which merely measured Agent B's dedup quality (5 of 8
        # titles in the live run appeared twice). It ranked "communication
        # skills" (24x3=72) above "project management" (8), which is how an
        # interview-skills course became the top recommendation for a data
        # engineer. Demand is now counted ONCE; occurrence is its own field and
        # only breaks ties.
        missing_entries: list[dict[str, Any]] = []
        for job, entries in job_entries:
            missing_entries.extend(
                {**e, "sector": job.sector} for e in entries if e["verdict"] == "missing"
            )
        if fallback_gap:
            missing_entries.extend(
                {**e, "sector": sector} for e in fallback_entries if e["verdict"] == "missing"
            )

        groups = _canonical_groups(missing_entries, key_to_esco, vectors, config)
        ranked = sorted(
            groups.values(),
            key=lambda g: (-g["demand_weight"], -g["jobs_missing_in"], g["skill"]),
        )[:10]

        missing_skill_details = [
            {
                "skill": g["skill"],
                "esco_code": g["esco_code"],
                # Kept as the name Agent E consumes, but now it is demand counted
                # once — not demand multiplied by how often retrieval repeated it.
                "priority_score": round(g["demand_weight"], 4),
                "jobs_missing_in": g["jobs_missing_in"],
                # Demand as a RATE. "58 postings ask for X" is uninterpretable
                # without "out of how many"; the denominator was published by
                # Agent B and ignored here.
                "demand_rate": g["demand_rate"],
                "low_confidence": g["low_confidence"],
                "best_similarity": g["best_similarity"],
                "nearest_candidate_skill": g["nearest_candidate_skill"],
                # Phrasings merged into this one gap, so a consumer can see that
                # "communication" and "communication skills" were one concept.
                "also_phrased_as": g["variants"],
            }
            for g in ranked
        ]

        # Weight-pooled, and only over jobs that produced a defined score. The
        # old mean averaged a 1-skill posting equally with a 15-skill one AND
        # included the undefined 0.0s, publishing 0.5533 where the honest figure
        # over real jobs was 0.757.
        scored = [j for j in matched_jobs if j["gap_score"] is not None]
        pooled_total = sum(
            e["weight"] for _job, es in job_entries for e in es if "weight" in e
        )
        pooled_miss = sum(
            e["weight"] for _job, es in job_entries for e in es
            if e.get("verdict") == "missing" and "weight" in e
        )
        aggregate = {
            "most_common_missing_skills": [g["skill"] for g in ranked],
            "missing_skill_details": missing_skill_details,
            "average_gap_score": (
                round(pooled_miss / pooled_total, 4) if pooled_total > 0 else None
            ),
            "jobs_scored": len(scored),
            "jobs_without_parsable_requirements": len(matched_jobs) - len(scored),
            # possible_match used to die inside each job. Published so a consumer
            # can see what we declined to resolve rather than inferring silence.
            "unresolved_skills": sorted(
                {e["skill"] for e in all_entries if e["verdict"] == "possible_match"}
            ),
        }

        if len(matched_jobs) - len(scored):
            warnings.append(
                f"{len(matched_jobs) - len(scored)} retrieved job(s) listed no parsable "
                f"requirements; they are reported with a null gap_score and excluded "
                f"from the average rather than counted as a perfect fit."
            )

        return {
            "matched_jobs": matched_jobs,
            "fallback_sector_gap": fallback_gap,
            "aggregate": aggregate,
            "warnings": warnings,
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


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9+#.]+", (text or "").lower()) if t]


def _contains_tokens(container: str, needle_key: str) -> bool:
    """Does ``container`` contain ``needle_key`` as a whole-token subsequence?

    The asymmetric relation cosine cannot express. "Data analytics engineering"
    genuinely satisfies a requirement for "data analytics"; symmetric similarity
    only knows the two phrases are 0.83 alike, which is the same number it gives
    pairs that do NOT satisfy each other. Whole tokens, so "java" never matches
    inside "javascript".
    """
    hay, needle = _tokens(container), _tokens(needle_key)
    if not needle or len(needle) > len(hay):
        return False
    return any(
        hay[i:i + len(needle)] == needle for i in range(len(hay) - len(needle) + 1)
    )


def _canonical_groups(
    entries: list[dict[str, Any]],
    key_to_esco: dict[str, Optional[str]],
    vectors: dict[str, list[float]],
    config: Config,
) -> dict[str, dict[str, Any]]:
    """Merge the same gap expressed several ways into one entry.

    Missing skills were deduped on the job posting's raw phrasing, so one concept
    occupied several of the ten slots a candidate actually sees: on the live run
    `professional communication` (190), `communication skills` (55) and
    `communication` (33) were three separate "top gaps", and Agent E duly spent
    three of its ten recommendations on them.

    Merge key: the ESCO concept where the vocabulary has one, else the strongest
    already-seen phrase this one is a token-superset of or highly similar to.
    Demand is the MAXIMUM over the group (not a sum — the phrasings describe one
    demand, and adding them would re-inflate exactly what this removes).
    """
    groups: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    for e in sorted(entries, key=lambda x: -x.get("weight", 0)):
        key, esco = e["key"], e["esco"] or key_to_esco.get(e["key"])
        gid = esco or None
        if gid is None:
            for existing in order:
                g = groups[existing]
                if g["esco_code"]:
                    continue
                if _contains_tokens(key, g["key"]) or _contains_tokens(g["key"], key):
                    gid = existing
                    break
                a, b = vectors.get(key), vectors.get(g["key"])
                if a is not None and b is not None and _dot(a, b) >= config.agent_c_skill_match:
                    gid = existing
                    break
            gid = gid or key

        if gid not in groups:
            groups[gid] = {
                "key": key, "skill": e["skill"], "esco_code": esco,
                "demand_weight": e.get("weight", 0.0), "jobs_missing_in": 0,
                "demand_rate": e.get("demand_rate"), "low_confidence": False,
                "best_similarity": e.get("best_similarity"),
                "nearest_candidate_skill": e.get("nearest_candidate_skill"),
                "variants": [],
            }
            order.append(gid)

        g = groups[gid]
        g["jobs_missing_in"] += 1
        g["demand_weight"] = max(g["demand_weight"], e.get("weight", 0.0))
        if e["skill"] != g["skill"] and e["skill"] not in g["variants"]:
            g["variants"].append(e["skill"])
        if (e.get("best_similarity") or 0) > (g["best_similarity"] or 0):
            g["best_similarity"] = e.get("best_similarity")
            g["nearest_candidate_skill"] = e.get("nearest_candidate_skill")
    return groups


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
            # Every threshold that shaped the numbers above. Without these an
            # output cannot be reproduced or even re-interpreted from itself —
            # the one archived fallback run is only readable today because its
            # threshold happened to be recorded.
            "calibration": {
                "agent_c_match_threshold": deps.config.agent_c_match_threshold,
                "agent_c_skill_match": deps.config.agent_c_skill_match,
                "agent_c_skill_possible": deps.config.agent_c_skill_possible,
                "agent_c_fallback_min_freq": deps.config.agent_c_fallback_min_freq,
                "agent_c_min_usable_postings": deps.config.agent_c_min_usable_postings,
                "agent_c_llm_matching": deps.config.agent_c_llm_matching,
                "top_k": state.get("top_k"),
                # What the candidate asked for, and — just as important — how far
                # each answer was actually acted on. `arrangement_applied` is
                # "retrieval_bias" rather than "filter" because nothing in the
                # corpus records a posting's work arrangement: the answer nudges
                # the embedded query text and that is the whole of its effect.
                # Recording the weakness is what stops a reader assuming the
                # stronger claim.
                "preferences": {
                    "preferred_role": state.get("preferred_role") or None,
                    "roles_only": bool(state.get("roles_only")),
                    "preferred_arrangement": state.get("preferred_arrangement") or None,
                    "role_applied": (
                        "replaced_headline" if state.get("preferred_role")
                        and state.get("roles_only")
                        else "joined_headline" if state.get("preferred_role") else None
                    ),
                    "arrangement_applied": (
                        "retrieval_bias" if state.get("preferred_arrangement") else None
                    ),
                },
            },
            # The user-decided home for candidate-side ESCO evidence,
            # including near-miss scores for unmapped skills.
            "candidate_skill_mappings": [
                m.model_dump() for m in state.get("candidate_mappings", [])
            ],
            "warnings": state.get("warnings", []),
        }

        # Validate against the PUBLISHED contract before writing. Agent E reads
        # this file through the same model, so a field that silently stopped
        # being emitted here would previously surface downstream as quietly worse
        # recommendations rather than as an error. Failing at the producer is the
        # only place the problem is cheap to see.
        SkillGap.model_validate(out)

        run_id = state.get("run_id") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        base = Path(state.get("output_dir") or deps.config.output_dir) / run_id
        base.mkdir(parents=True, exist_ok=True)
        path = base / "skill_gap.json"
        path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"output_path": str(path)}

    return persist
