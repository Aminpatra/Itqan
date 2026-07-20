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

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# PaddleX otherwise probes its model host on *every* import, which makes the CLI
# feel hung on a slow connection. Our models are already cached locally.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

# This env emits a RequestsDependencyWarning on every invocation (urllib3/chardet
# version skew). Harmless, but it corrupts the gap-collection prompts.
warnings.filterwarnings("ignore", message=".*urllib3.*chardet.*")
warnings.filterwarnings("ignore", category=UserWarning, module="paddle.*")

load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class Config:
    """Tunables.

    Defaults suit English-language documents. ``ocr_lang`` is the one to change
    for CVs in another script — PaddleOCR's English models will misread Arabic,
    Chinese or Devanagari rather than fail loudly. Exposed as ``--ocr-lang``.
    """

    model: str = field(default_factory=lambda: os.getenv("ITQAN_MODEL", "gpt-4o-mini"))
    temperature: float = 0.0

    # --- grounding thresholds (see agents/agent_a_cv_extraction/grounding.py) ---
    # >= grounded_threshold          -> accepted outright
    # adjudicate_threshold .. below  -> escalated to the LLM adjudicator
    # < adjudicate_threshold         -> dropped as ungrounded
    grounded_threshold: float = 0.92
    adjudicate_threshold: float = 0.75

    # An OCR block below this confidence makes any field it supports a gap.
    low_ocr_confidence: float = 0.60

    # Bounded HITL loop — after this many rounds, unfilled gaps finalize as null.
    max_review_rounds: int = 2

    ocr_lang: str = "en"
    pdf_raster_dpi: int = 200

    # A PDF needs at least this much extractable text to count as having a real
    # text layer; below it we treat the file as scanned and route to OCR.
    pdf_text_min_chars: int = 200
    pdf_text_min_chars_per_page: int = 50

    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "output")

    def require_api_key(self) -> str:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        return key
