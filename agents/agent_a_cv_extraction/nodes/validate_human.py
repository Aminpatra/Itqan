"""Validate and merge manually-entered values.

Human input skips the grounding pass by design — a phone number the user types
will not appear in the OCR text, and requiring it to would defeat the purpose of
asking. This node is the compensating gate, so that the HITL path is not simply a
hole through every other check in the pipeline.

It also increments ``review_rounds``, which is what makes the loop back to
``assess_gaps`` terminate.
"""

from __future__ import annotations

import json
import re
from typing import Any

from shared.config import Config
from shared.llm import as_dict, structured

from ..prompts import HUMAN_INPUT_VALIDATION_PROMPT
from ..schemas import HumanInputValidation
from ..state import AgentState

_INDEX = re.compile(r"^(.*?)\[(\d+)\]$")


def _set_path(data: dict[str, Any], path: str, value: str) -> None:
    """Write ``value`` at ``path``, creating containers as needed.

    Handles ``contact.email`` and ``education[0].institution`` alike.
    """
    parts = path.split(".")
    cursor: Any = data

    for i, part in enumerate(parts):
        last = i == len(parts) - 1
        match = _INDEX.match(part)

        if match:
            key, index = match.group(1), int(match.group(2))
            container = cursor.setdefault(key, [])
            while len(container) <= index:
                container.append({})
            if container[index] is None:
                container[index] = {}
            if last:
                container[index] = value
            else:
                cursor = container[index]
        elif last:
            cursor[part] = value
        else:
            nxt = cursor.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[part] = nxt
            cursor = nxt


def _split_list(raw: str) -> list[str]:
    return [item.strip() for item in re.split(r"[,;]", raw) if item.strip()]


def make_validate_human_node(llm: Any, config: Config):
    chain = HUMAN_INPUT_VALIDATION_PROMPT | structured(llm, HumanInputValidation)

    def validate_human_input(state: AgentState) -> dict[str, Any]:
        raw = {k: v for k, v in (state.get("human_input") or {}).items() if str(v).strip()}
        rounds = state.get("review_rounds", 0) + 1

        if not raw:
            # User skipped everything. Bump the round so the loop ends.
            return {
                "review_rounds": rounds,
                "human_input": {},
                "trace": ["validate_human_input(empty)"],
            }

        gaps_by_path = {g["field_path"]: g for g in (state.get("gaps") or [])}
        field_specs = [
            {
                "field_path": path,
                "expects": gaps_by_path.get(path, {}).get("prompt_hint", ""),
                "why_asked": gaps_by_path.get(path, {}).get("reason", ""),
            }
            for path in raw
        ]

        try:
            validation = as_dict(
                chain.invoke(
                    {
                        "field_specs": json.dumps(field_specs, indent=2, ensure_ascii=False),
                        "raw_answers": json.dumps(raw, indent=2, ensure_ascii=False),
                    }
                )
            )
            fields = validation.get("fields", [])
        except Exception as exc:
            return {
                "review_rounds": rounds,
                "human_input": {},
                "warnings": [f"Could not validate manual input ({exc}); discarding it."],
                "trace": ["validate_human_input(failed)"],
            }

        extraction = dict(state.get("cv_extraction") or {})
        transcript = dict(state.get("transcript_extraction") or {})
        accepted_paths: list[str] = []
        notes: list[str] = []
        transcript_touched = False

        for field in fields:
            path = field.get("field_path")
            if path not in raw:
                continue  # model invented a path

            if not field.get("accepted"):
                notes.append(
                    f"Rejected '{path}': {field.get('issue') or 'failed validation'}"
                )
                continue

            value = field.get("normalized_value")
            if not value or not str(value).strip():
                continue
            value = str(value).strip()

            # Two synthetic paths don't map onto the schema directly.
            if path == "skills_freeform":
                skills = list(extraction.get("skills") or [])
                for name in _split_list(value):
                    skills.append(
                        {"name": name, "source_span": None, "category": "unknown"}
                    )
                extraction["skills"] = skills
                accepted_paths.extend(
                    f"skills[{i}].name"
                    for i in range(len(skills) - len(_split_list(value)), len(skills))
                )
                notes.append(f"Added {len(_split_list(value))} skill(s) from manual entry")
                continue

            if path == "transcript.courses":
                courses = list(transcript.get("courses") or [])
                for title in _split_list(value):
                    courses.append({"title": title})
                transcript["courses"] = courses
                transcript_touched = True
                accepted_paths.append(path)
                notes.append(f"Added {len(_split_list(value))} course(s) from manual entry")
                continue

            _set_path(extraction, path, value)
            accepted_paths.append(path)
            notes.append(f"Accepted '{path}' = {value!r}")

        updates: dict[str, Any] = {
            "cv_extraction": extraction,
            "review_rounds": rounds,
            "human_input": {},
            "trace": [f"validate_human_input({len(accepted_paths)} accepted)"],
        }
        if transcript_touched:
            updates["transcript_extraction"] = transcript
        if accepted_paths:
            updates["human_supplied_fields"] = accepted_paths
        if notes:
            updates["human_validation_notes"] = notes
        return updates

    return validate_human_input
