"""The fenced tier: settle requirements the deterministic tiers could not.

Fenced in four ways, because this is the only place in Agent C where a model can
influence a published number:

1. **It never sees free text.** Input is two lists of skill NAMES plus a weak-
   evidence marker. No CV prose, no job description — so there is nothing to be
   injected by, and nothing to invent detail from.
2. **It cannot introduce a skill.** Every ``satisfied_by`` is checked against the
   candidate's actual list; a name that is not there voids that verdict and the
   deterministic answer stands. Same discipline as Agent A's grounded spans and
   Agent B's verified legitimacy quotes.
3. **It cannot manufacture confidence.** ``uncertain`` keeps the existing verdict,
   and weak-evidence skills are barred from producing ``satisfied`` at all.
4. **It is optional.** With ``agent_c_llm_matching`` off, or no LLM injected, the
   agent behaves exactly as it did before — deterministic end to end.

One batched call per run, not one per skill: the candidate's list is fixed, so
every unresolved requirement can be judged together. On the live corpus that is a
single call for ~40 requirements.
"""

from __future__ import annotations

from typing import Any, Optional

from shared.config import Config


def _label(record: dict[str, Any]) -> str:
    """A skill as the model sees it: name, plus a marker when the candidate only
    claimed it. The marker is what stops an unevidenced claim closing a gap."""
    name = (record.get("name") or "").strip()
    weak = record.get("quality") == "low" or record.get("evidence_type") in {
        "claim_only", "adjacent"
    }
    return f"{name} [weak evidence]" if weak else name


def resolve_skills(
    *,
    unresolved: list[dict[str, Any]],
    candidate_skills: list[str],
    skill_records: list[dict[str, Any]],
    config: Config,
    llm: Any = None,
) -> dict[str, dict[str, Any]]:
    """Judge unresolved requirements. Returns ``{requirement_key: patch}``.

    A patch only ever carries ``verdict``/``resolved_by``/``satisfied_by``. An
    empty result means "nothing changed", which is the correct outcome whenever
    the tier is disabled, unavailable, or unsure.
    """
    if llm is None or not unresolved or not getattr(config, "agent_c_llm_matching", False):
        return {}

    from shared.llm import as_dict, structured

    from .prompts import SKILL_MATCH_PROMPT
    from .schemas import SkillMatchBatch

    by_name = {(s or "").strip().lower(): s for s in candidate_skills}
    records = {(r.get("name") or "").strip().lower(): r for r in skill_records}
    weak = {
        k for k, r in records.items()
        if r.get("quality") == "low" or r.get("evidence_type") in {"claim_only", "adjacent"}
    }

    labelled = [
        _label(records.get(k, {"name": name})) for k, name in by_name.items()
    ]
    # Deduped: several jobs asking for the same thing is one question.
    requirements = list(dict.fromkeys(e["skill"] for e in unresolved))

    try:
        batch = as_dict(
            (SKILL_MATCH_PROMPT | structured(llm, SkillMatchBatch)).invoke({
                "candidate_skills": ", ".join(labelled),
                "requirements": "; ".join(requirements),
            })
        )
    except Exception:  # noqa: BLE001 - advisory tier; the deterministic verdict stands
        return {}

    wanted = {r.strip().lower(): r for r in requirements}
    patches: dict[str, dict[str, Any]] = {}

    for verdict in batch.get("verdicts", []):
        req = (verdict.get("requirement") or "").strip().lower()
        if req not in wanted:
            continue                     # a requirement nobody asked about
        decision = verdict.get("decision")
        if decision == "not_satisfied":
            # The deterministic tiers already have it as missing or possible; the
            # model agreeing it is absent is not new information, and letting it
            # DOWNGRADE possible_match to missing would let it manufacture gaps.
            continue
        if decision != "satisfied":
            continue                     # uncertain -> keep what we had

        cited = (verdict.get("satisfied_by") or "").strip().lower()
        if cited not in by_name:
            # Invented a skill. The whole verdict is void — this is the fence.
            continue
        if cited in weak:
            # Weak evidence cannot close a gap, whatever the model concluded.
            continue

        patches[req] = {
            "verdict": "matched",
            "resolved_by": "llm",
            "satisfied_by": by_name[cited],
            "llm_reason": (verdict.get("reason") or "").strip() or None,
        }

    return patches
