"""Choose the model by running THIS system's prompts against it.

    python main.py bakeoff
    python main.py bakeoff --models google/gemini-3.7-flash,openai/gpt-5.6-luna
    python main.py bakeoff --runs 5

Read-only: no database, no writes, no corpus touched. It builds the real chains —
the same `ChatPromptTemplate | structured(llm, Schema)` every agent builds — and
asks whether the answers are the ones this system depends on.

**Why this is a committed tool and not a scratchpad script.** The model was chosen
this way on 2026-07-29 and the numbers went into `shared/config.py`; when the
question came up again the harness was gone and had to be written from scratch.
The same thing happened three times with source probing before `--probe-source`
existed. A measurement you cannot cheaply repeat is one that silently stops being
re-checked, and then a year-old table decides today's model.

**The probes are the failures this system has actually had**, not a benchmark:

* CANONICAL — Agent D must fold `MS Excel` to `excel`. `gpt-5.4-nano` failed this
  0/5 and would have refragmented the demand<->supply join, which is the entire
  point of the shared vocabulary.
* SUBSUMPTION — Agent C's fenced matcher must know TensorFlow satisfies "machine
  learning" and that JavaScript does NOT satisfy Java. Cosine cannot express
  this; it is why the tier exists.
* ARABIC — new, and the reason for this round. Every probe in the 2026-07-29
  bake-off was English, for a product whose job corpus is Arabic, whose CVs may be
  Arabic, and whose assistant answers in the language it was asked in.
* BINDS — can the model be constrained to our Optional-heavy schemas at all? A
  null means "not in the document" here, which is the anti-hallucination
  mechanism, and strict schema modes are picky about exactly that. Everything
  except Agent E's rationale is structured, so a model that cannot bind is
  unusable rather than merely worse.

Latency is reported because it has disqualified a whole generation before: 12-19s
per call turns a 1,335-course backfill into 4-7 hours.
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Any, Callable

from shared.config import Config

# --- the fixtures, each one a real shape this system handles ---------------

# Agent D: a description that names the tool the way a marketer does.
COURSE = {
    "name": "Data Analysis Essentials",
    "provider": "Example University",
    "body": ("Learn to clean and report on business data. The course covers MS Excel "
             "for spreadsheets, Microsoft Power BI for dashboards, and an "
             "introduction to SQL for querying databases."),
}

# Agent C: the boundary cases, including the one that used to score 1.0 gap.
CANDIDATE_SKILLS = ["TensorFlow", "PyTorch", "scikit-learn", "JavaScript", "SQL", "Python"]
REQUIREMENTS = ["machine learning", "Java", "software development"]

# Agent B: a real-shaped Arabic posting. Arabic job ads are most of this corpus.
ARABIC_POSTING = {
    "title": "مطلوب محاسب لشركة في مسقط",
    "body": ("تعلن شركة الخليج للتجارة عن حاجتها لتوظيف محاسب في مسقط، سلطنة عمان.\n"
             "المهام: إعداد التقارير المالية، ومتابعة الحسابات، وتدقيق الفواتير.\n"
             "الشروط: بكالوريوس محاسبة، وخبرة سنتين، وإجادة برنامج إكسل.\n"
             "الدوام: دوام كامل. للتقديم أرسل سيرتك الذاتية إلى hr@example.com"),
}


def _probe_canonical(chain_for: Callable[..., Any]) -> tuple[bool, str]:
    """`MS Excel` must arrive as `excel`, not as the vendor's spelling."""
    from agents.agent_d_course_ingest.prompts.extraction import EXTRACTION_PROMPT
    from agents.agent_d_course_ingest.schemas import CourseExtraction

    out = chain_for(EXTRACTION_PROMPT, CourseExtraction).invoke(COURSE)
    skills = [s.lower().strip() for s in (getattr(out, "taught_skills", None) or [])]
    ok = any(s == "excel" for s in skills)
    return ok, ("excel" if ok else f"got {skills[:4]}")


def _probe_subsumption(chain_for: Callable[..., Any]) -> tuple[bool, str]:
    """Two different questions, and conflating them would rank a useless model first.

    `uncertain` is a first-class answer in this schema, not a failure: the
    deterministic verdict stands, so the model declining to guess costs nothing.
    Scoring it as simply wrong would mean a model that answered "uncertain" to
    everything looked safe — and it IS safe, and it is also the reason the LLM
    tier exists at all, so it cannot be the top score.

    So this measures both, and only one of them can fail the probe:

    * UNSAFE — a verdict that moves a published gap score the wrong way. Claiming
      JavaScript satisfies Java invents a match; claiming TensorFlow does not
      satisfy machine learning recreates the 1.0-gap bug this tier was built for.
      Any of these fails, however much else is right.
    * RESOLVED — how many of the genuinely satisfiable requirements it actually
      settled. Not a failure, but it is what the tier is FOR, and a model that
      resolves nothing is one to reject on the strength of this column even
      though it never says anything false.
    """
    from agents.agent_c_gap_analysis.prompts.skill_match import SKILL_MATCH_PROMPT
    from agents.agent_c_gap_analysis.schemas import SkillMatchBatch

    out = chain_for(SKILL_MATCH_PROMPT, SkillMatchBatch).invoke({
        "candidate_skills": "\n".join(f"- {s}" for s in CANDIDATE_SKILLS),
        "requirements": "\n".join(f"- {r}" for r in REQUIREMENTS),
    })
    got = {v.requirement.strip().lower(): v.decision
           for v in (getattr(out, "verdicts", None) or [])}

    satisfiable = ("machine learning", "software development")
    unsafe = []
    if got.get("java") == "satisfied":
        unsafe.append("java=satisfied (JavaScript is not Java)")
    for key in satisfiable:
        if got.get(key) == "not_satisfied":
            unsafe.append(f"{key}=not_satisfied")

    resolved = sum(1 for key in satisfiable if got.get(key) == "satisfied")
    detail = f"resolved {resolved}/{len(satisfiable)}"
    if unsafe:
        return False, "UNSAFE: " + "; ".join(unsafe)
    return True, detail


def _probe_arabic_posting(chain_for: Callable[..., Any]) -> tuple[bool, str]:
    """An Arabic ad must yield one vacancy with real skills, not an empty shell."""
    from agents.agent_b_job_ingest.prompts.extraction import EXTRACTION_PROMPT
    from agents.agent_b_job_ingest.schemas import JobExtractionBatch

    out = chain_for(EXTRACTION_PROMPT, JobExtractionBatch).invoke(ARABIC_POSTING)
    jobs = getattr(out, "jobs", None) or []
    if len(jobs) != 1:
        return False, f"{len(jobs)} vacancies from a single-vacancy ad"
    skills = [s.lower() for s in (jobs[0].required_skills or [])]
    ok = bool(skills) and any("excel" in s or "account" in s or "financial" in s
                              or "محاسب" in s for s in skills)
    return ok, (f"{len(skills)} skills: {skills[:3]}" if skills else "no skills extracted")


def _probe_arabic_assistant(chain_for: Callable[..., Any]) -> tuple[bool, str]:
    """Hud answers in Arabic AND survives its own verifier.

    Both halves matter. An answer that is fluent but states a figure the fact
    sheet does not contain is discarded and the person gets the template, so a
    model that is chatty in Arabic but loose with numbers is worse here than one
    that is plainer.
    """
    from agents.agent_s_assistant.facts import build_fact_sheet, verify_answer
    from agents.agent_s_assistant.prompts import ASSISTANT_PROMPT
    from agents.agent_s_assistant.schemas import AssistantReply

    sheet = build_fact_sheet(readiness=60, jobs=[], courses=[], gaps=[],
                             suggested_role=None, matched_at=None)
    out = chain_for(ASSISTANT_PROMPT, AssistantReply).invoke({
        "facts": sheet, "about": "", "history": "",
        "question": "ما هي درجة جاهزيتي؟",
    })
    text = getattr(out, "answer", "") or ""
    arabic = sum(1 for ch in text if "؀" <= ch <= "ۿ")
    if arabic < 5:
        return False, "answered a question asked in Arabic in another language"
    problem = verify_answer(text, sheet)
    return problem is None, ("published" if problem is None else f"REJECTED: {problem}")


ARABIC_CV = """السيرة الذاتية

الاسم: مريم البلوشي
البريد الإلكتروني: maryam.b@example.com
الهاتف: 96891234567

المؤهل العلمي:
بكالوريوس علوم الحاسوب، جامعة السلطان قابوس، 2025

المهارات:
Python، SQL، تحليل البيانات، Power BI، العمل الجماعي

المشاريع:
مشروع تخرج: نظام لتحليل بيانات المبيعات باستخدام Python وقواعد البيانات.
"""


def _probe_arabic_cv(chain_for: Callable[..., Any]) -> tuple[bool, str]:
    """An Arabic CV must yield values that GROUND against the Arabic source.

    The sharpest probe here, because its failure mode is silent and this project
    has already paid for it once: extracted fields are checked back against the
    document, and anything that will not ground is dropped. A model that
    paraphrases or transliterates instead of copying does not produce a wrong
    answer — it produces an EMPTY profile, and the candidate is simply told
    nothing was found. On a real CV that once deleted all 24 skills at once.

    So this asks the question the pipeline asks: of what came back, how much
    survives `ground_value` against the original Arabic?
    """
    from agents.agent_a_cv_extraction.prompts.extraction import CV_EXTRACTION_PROMPT
    from agents.agent_a_cv_extraction.schemas import CVExtraction
    from shared.grounding import ground_value, normalize

    out = chain_for(CV_EXTRACTION_PROMPT, CVExtraction).invoke(
        {"source_text": ARABIC_CV, "ocr_quality_note": ""})

    source = normalize(ARABIC_CV)
    values = [getattr(out, "full_name", None)] + [
        getattr(s, "name", None) for s in (getattr(out, "skills", None) or [])]
    values = [v for v in values if v]
    if not values:
        return False, "extracted nothing at all from an Arabic CV"

    grounded = sum(1 for v in values if ground_value(str(v), source)[0] >= 0.92)
    ok = grounded == len(values)
    return ok, f"{grounded}/{len(values)} values ground against the Arabic source"


PROBES: list[tuple[str, Callable[..., Any]]] = [
    ("canonical", _probe_canonical),
    ("subsumption", _probe_subsumption),
    ("ar-posting", _probe_arabic_posting),
    ("ar-cv", _probe_arabic_cv),
    ("ar-hud", _probe_arabic_assistant),
]


def _run_model(model: str, runs: int, config: Config,
               reasoning_effort: str | None = None) -> dict[str, Any]:
    from langchain_core.callbacks import UsageMetadataCallbackHandler

    from shared.llm import build_llm, structured

    overrides: dict[str, Any] = {"model": model}
    if reasoning_effort:
        overrides["reasoning_effort"] = reasoning_effort
    llm = build_llm(config, **overrides)

    # TOKENS, NOT LATENCY, IS THE COST QUESTION FOR A REASONING MODEL.
    #
    # `config.py` records why: gpt-5-nano was priced BELOW gpt-4o-mini and cost
    # 9.4x more per call, because it emitted 4,480 reasoning tokens on two trivial
    # calls and they bill at the output rate. A rate card cannot answer that and
    # neither can a stopwatch — only usage can, so it is read from the response
    # rather than estimated.
    usage = UsageMetadataCallbackHandler()

    def chain_for(prompt: Any, schema: Any) -> Any:
        return (prompt | structured(llm, schema)).with_config({"callbacks": [usage]})

    row: dict[str, Any] = {"model": model, "binds": "yes", "latencies": [],
                           "effort": reasoning_effort or "default", "usage": usage}
    for name, probe in PROBES:
        passes = 0
        detail = ""
        for _ in range(runs):
            started = time.monotonic()
            try:
                ok, detail = probe(chain_for)
            except Exception as exc:  # noqa: BLE001 - a failure IS the result
                row["binds"] = "no"
                # 400 chars, not 70. The first truncation cut a 402 off at "This
                # request " and hid the actionable half -- that the reservation
                # was 65,536 tokens, not that the account was empty. A diagnostic
                # that truncates the diagnosis is not one.
                detail = f"{type(exc).__name__}: {exc}"[:400]
                ok = False
            row["latencies"].append(time.monotonic() - started)
            passes += 1 if ok else 0
        row[name] = f"{passes}/{runs}"
        row[f"{name}_detail"] = detail
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python main.py bakeoff",
        description="Run this system's real prompts against candidate models.")
    parser.add_argument("--models", default="",
                        help="comma-separated; defaults to the current model plus "
                             "the candidates for this round")
    parser.add_argument("--runs", type=int, default=3,
                        help="repeats per probe (default 3; the recorded round used 5)")
    parser.add_argument("--effort", default="",
                        help="comma-separated reasoning efforts to try (e.g. none,low,medium). "
                             "Empty uses the model's default, which for the GPT-5.6 family "
                             "is medium — the setting the token column exists to check.")
    args = parser.parse_args(argv)

    config = Config()
    models = [m.strip() for m in args.models.split(",") if m.strip()] or [
        config.model,
        "google/gemini-3.7-flash",
        "openai/gpt-5.6-luna",
    ]

    efforts: list[str | None] = [e.strip() for e in args.effort.split(",") if e.strip()] or [None]

    print(f"base_url: {config.api_base or 'https://api.openai.com/v1 (direct)'}")
    print(f"runs per probe: {args.runs}\n")
    header = (f"{'model':<24} {'effort':<8} {'binds':<6} "
              + " ".join(f"{n:<12}" for n, _ in PROBES)
              + f" {'median':<8} {'in/call':<8} {'out':<7} reasoning")
    print(header)
    print("-" * len(header))

    rows = []
    for model in models:
        for effort in efforts:
            try:
                row = _run_model(model, args.runs, config, effort)
            except Exception as exc:  # noqa: BLE001 - report and carry on to the next
                print(f"{model:<24} {(effort or 'default'):<8} FAILED  "
                      f"{type(exc).__name__}: {exc}"[:120])
                continue
            rows.append(row)
            median = statistics.median(row["latencies"]) if row["latencies"] else 0.0
            calls = max(1, len(row["latencies"]))
            totals: dict[str, int] = {}
            for per_model in (row["usage"].usage_metadata or {}).values():
                totals["input"] = totals.get("input", 0) + per_model.get("input_tokens", 0)
                totals["output"] = totals.get("output", 0) + per_model.get("output_tokens", 0)
                details = per_model.get("output_token_details") or {}
                totals["reasoning"] = totals.get("reasoning", 0) + details.get("reasoning", 0)
            cells = " ".join(f"{row.get(n, '-'):<12}" for n, _ in PROBES)
            print(f"{row['model']:<24} {row['effort']:<8} {row['binds']:<6} {cells} "
                  f"{median:>6.2f}s  {totals.get('input', 0) // calls:<8} "
                  f"{totals.get('output', 0) // calls:<7} {totals.get('reasoning', 0) // calls}")

    # Always printed, not only on failure. `subsumption` passes whenever nothing
    # UNSAFE was said, and how much it actually resolved lives only here — a
    # model that declines to answer everything scores a clean pass above.
    print("\ndetail (the last run of each probe):")
    for row in rows:
        for name, _ in PROBES:
            print(f"  {row['model']:<28} {name:<12} {row.get(f'{name}_detail', '')}")

    print("\nRead the Arabic rows yourself before deciding. A fabricated Arabic "
          "skill passes every check in this file.")
    return 0
