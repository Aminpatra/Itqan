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

from ..grounding import ground_extraction, prune_field, verify_quote
from ..prompts import GROUNDING_ADJUDICATION_PROMPT
from ..schemas import GroundingReport
from ..state import AgentState


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
        human_fields = set(state.get("human_supplied_fields") or [])

        result = ground_extraction(
            extraction,
            source_text,
            grounded_threshold=config.grounded_threshold,
            adjudicate_threshold=config.adjudicate_threshold,
            skip_fields=human_fields,
        )
        report = result["report"]
        dropped = list(result["dropped"])
        warnings: list[str] = []

        # --- LLM adjudication for the ambiguous band only --------------------
        if result["needs_llm"]:
            try:
                adjudication = as_dict(
                    chain.invoke(
                        {
                            "source_text": source_text,
                            "candidate_fields": json.dumps(
                                result["needs_llm"], indent=2, ensure_ascii=False
                            ),
                        }
                    )
                )
                for verdict in adjudication.get("verdicts", []):
                    path = verdict.get("field_path")
                    if path not in report:
                        continue  # model invented a field path; ignore it

                    quote_ok = verify_quote(verdict.get("evidence_quote"), source_text)
                    grounded = bool(verdict.get("grounded")) and quote_ok

                    if verdict.get("grounded") and not quote_ok:
                        warnings.append(
                            f"Adjudicator claimed '{path}' was grounded but could not "
                            f"quote it verbatim; treating as ungrounded."
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
                for item in result["needs_llm"]:
                    dropped.append(item["field_path"])

        # --- prune everything that failed ------------------------------------
        for path in dropped:
            if prune_field(extraction, path):
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
        if dropped:
            updates["dropped_fields"] = list(dict.fromkeys(dropped))
        if warnings:
            updates["warnings"] = warnings
        return updates

    return verify_grounding
