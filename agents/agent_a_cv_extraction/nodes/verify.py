"""Grounding verification.

Order matters here. The cheap deterministic pass runs first and resolves most
fields; only the ambiguous band reaches the LLM. That keeps cost down, but the
real reason is that a deterministic check cannot itself hallucinate — the LLM is
the fallback, not the primary authority.

The adjudicator's answer is then checked against the source in Python. If it
claims a field is grounded but cannot produce a quote that actually appears in
the document, the field is dropped anyway.
"""

from __future__ import annotations

import json
from typing import Any

from shared.config import Config
from shared.llm import as_dict, structured

from shared.grounding import ground_extraction, prune_field, verify_quote
from ..prompts import GROUNDING_ADJUDICATION_PROMPT
from ..schemas import GroundingReport
from ..state import AgentState

# Transcript field paths are namespaced in the grounding report so they cannot
# collide with the CV's (both have `courses[0].title`), and so pruning can route
# each path back to the object it came from.
TRANSCRIPT_PREFIX = "transcript."


def _flag_unverified_spans(
    extraction: dict[str, Any], source_text: str, report: dict[str, Any],
    warnings: list[str],
) -> None:
    """Record which skills cited a ``source_span`` we could not find in the source.

    ``source_span`` is demanded by the extraction prompt and was checked nowhere,
    so it was pure token cost. It is worth recording — but it is NOT worth acting
    on, and an earlier version of this function learned that expensively: it
    dropped any skill whose span failed to verify, and on a real PDF CV that
    deleted **all 24** of the candidate's skills. Every one of those names had
    scored a perfect 1.0 against the document. The spans were merely paraphrased.

    The lesson is about relative evidence. A skill's NAME matching the document
    verbatim is strong, direct evidence; a model's ability to reproduce a quote
    exactly is a property of the model, not of the candidate. Letting the weak
    signal override the strong one is backwards, so this now annotates and never
    condemns: the flag rides along in the grounding report for anyone who wants
    it, and the name grounding continues to decide the skill's fate.
    """
    unverified: list[str] = []
    for index, skill in enumerate(extraction.get("skills") or []):
        if not isinstance(skill, dict):
            continue
        span = skill.get("source_span")
        if not isinstance(span, str) or not span.strip():
            continue
        path = f"skills[{index}].name"
        entry = report.get(path)
        if entry is None:
            continue
        verified = verify_quote(span, source_text)
        entry["span_verified"] = verified
        if not verified:
            unverified.append(str(skill.get("name") or path))

    if unverified:
        # One line, not one per skill: this is a note about extraction quality,
        # not a per-skill problem the operator must act on.
        warnings.append(
            f"{len(unverified)} skill(s) cited a source span that could not be matched "
            f"verbatim; the skills themselves were still checked against the document "
            f"on their own merit ({', '.join(unverified[:6])})."
        )


def make_verify_node(llm: Any, config: Config):
    chain = GROUNDING_ADJUDICATION_PROMPT | structured(llm, GroundingReport)

    def verify_grounding(state: AgentState) -> dict[str, Any]:
        cv_doc = state["cv_doc"]
        transcript_doc = state.get("transcript_doc")

        # Both documents are legitimate evidence for the merged profile: a name on
        # the transcript grounds the same name on the CV.
        source_text = cv_doc.get("text", "")
        if transcript_doc and transcript_doc.get("text"):
            source_text = f"{source_text}\n\n{transcript_doc['text']}"

        extraction = dict(state.get("cv_extraction") or {})
        transcript = dict(state.get("transcript_extraction") or {})
        human_fields = set(state.get("human_supplied_fields") or [])
        warnings: list[str] = []

        result = ground_extraction(
            extraction,
            source_text,
            grounded_threshold=config.grounded_threshold,
            adjudicate_threshold=config.adjudicate_threshold,
            skip_fields=human_fields,
        )
        report = dict(result["report"])
        dropped = list(result["dropped"])
        needs_llm = list(result["needs_llm"])

        # --- the transcript is published, so it must be verified too ----------
        # persist merges transcript courses into candidate.courses and publishes
        # academic_record (institution / program / CGPA). Those were reaching the
        # envelope with no provenance at all — CGPA, the highest-stakes number in
        # the profile, was never checked against anything. The transcript's own
        # text is already part of source_text, so this costs one extra pass and
        # almost never drops anything: it is about recording that we looked.
        if transcript:
            t_result = ground_extraction(
                transcript,
                source_text,
                grounded_threshold=config.grounded_threshold,
                adjudicate_threshold=config.adjudicate_threshold,
            )
            for path, entry in t_result["report"].items():
                report[f"{TRANSCRIPT_PREFIX}{path}"] = entry
            dropped.extend(f"{TRANSCRIPT_PREFIX}{p}" for p in t_result["dropped"])
            needs_llm.extend(
                {"field_path": f"{TRANSCRIPT_PREFIX}{i['field_path']}", "value": i["value"]}
                for i in t_result["needs_llm"]
            )

        # --- record span quality; it annotates, it does not condemn ----------
        _flag_unverified_spans(extraction, source_text, report, warnings)

        # --- LLM adjudication for the ambiguous band only --------------------
        if needs_llm:
            try:
                adjudication = as_dict(
                    chain.invoke(
                        {
                            "source_text": source_text,
                            "candidate_fields": json.dumps(
                                needs_llm, indent=2, ensure_ascii=False
                            ),
                        }
                    )
                )
                for verdict in adjudication.get("verdicts", []):
                    path = verdict.get("field_path")
                    if path not in report:
                        continue  # model invented a field path; ignore it

                    # The quote must exist in the source AND actually support the
                    # value. Checking only that it exists let a verdict certify a
                    # fabricated field by citing any true sentence in the document
                    # — the self-certification this backstop exists to prevent.
                    quote_ok = verify_quote(
                        verdict.get("evidence_quote"), source_text, report[path].get("value")
                    )
                    grounded = bool(verdict.get("grounded")) and quote_ok

                    if verdict.get("grounded") and not quote_ok:
                        warnings.append(
                            f"Adjudicator claimed '{path}' was grounded but could not "
                            f"quote source text supporting it; treating as ungrounded."
                        )

                    report[path]["grounded"] = grounded
                    report[path]["method"] = "llm"
                    report[path]["evidence_quote"] = verdict.get("evidence_quote")
                    if not grounded:
                        dropped.append(path)
            except Exception as exc:
                # Fail closed: without adjudication the ambiguous fields stay
                # unverified, so they are dropped rather than trusted.
                warnings.append(f"Grounding adjudication failed ({exc}); dropping unverified fields.")
                for item in needs_llm:
                    dropped.append(item["field_path"])

        # --- prune everything that failed ------------------------------------
        # Each path is pruned from the object it was grounded against; the prefix
        # is what routes a transcript field back to the transcript.
        for path in dropped:
            if path.startswith(TRANSCRIPT_PREFIX):
                pruned = prune_field(transcript, path[len(TRANSCRIPT_PREFIX):])
            else:
                pruned = prune_field(extraction, path)
            if pruned:
                report[path]["method"] = "dropped"
                report[path]["grounded"] = False

        if dropped:
            warnings.append(
                f"{len(dropped)} field(s) could not be verified against the source "
                f"and were removed: {', '.join(sorted(set(dropped))[:8])}"
                + ("…" if len(set(dropped)) > 8 else "")
            )

        updates: dict[str, Any] = {
            "cv_extraction": extraction,
            "grounding_report": report,
            "trace": ["verify_grounding"],
        }
        if transcript:
            updates["transcript_extraction"] = transcript
        if dropped:
            updates["dropped_fields"] = list(dict.fromkeys(dropped))
        if warnings:
            updates["warnings"] = warnings
        return updates

    return verify_grounding
