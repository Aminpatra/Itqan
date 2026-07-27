"""PaddleOCR wrapper.

Three things here are version-specific to PaddleOCR 3.x and will look wrong if
compared against any 2.x tutorial:

- ``predict()``, not ``ocr()`` — the latter is deprecated in 3.x.
- ``use_textline_orientation``, not ``use_angle_cls`` — the old kwarg no longer
  exists and would be silently swallowed.
- Results come back as an ``OCRResult`` dict with ``rec_texts`` / ``rec_scores`` /
  ``rec_polys``, where the polys are numpy arrays. ``json.dump`` raises on those
  under numpy 2.x, so everything goes through ``shared.artifacts.to_jsonable``.

The engine is a module-level singleton because initialisation takes seconds and
the graph is invoked repeatedly across the human-in-the-loop resume cycle.

Debug entry point (run this before wiring OCR into the graph):

    python -m agents.agent_a_cv_extraction.ingestion.ocr some_image.png
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from shared import config as _config  # noqa: F401  — sets env vars before paddle import
from shared.text_norm import is_rtl

# Keyed by language: a run may legitimately process an Arabic CV alongside an
# English transcript, and a single cached engine would silently return the first
# language's model for both.
_ENGINES: dict[str, Any] = {}


def get_engine(lang: str = "en") -> Any:
    """Lazily build a PaddleOCR engine per language, reusing it thereafter."""
    if lang not in _ENGINES:
        from paddleocr import PaddleOCR

        _ENGINES[lang] = PaddleOCR(
            lang=lang,
            use_textline_orientation=True,
            # Both off by default: they add seconds per page and only help with
            # photographed/skewed documents. Turn on if you feed phone snaps.
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
    return _ENGINES[lang]


def _poly_bounds(poly: Any) -> tuple[float, float, float]:
    """Return ``(top_y, left_x, height)`` for a 4-point polygon."""
    points = [(float(p[0]), float(p[1])) for p in poly]
    ys = [p[1] for p in points]
    xs = [p[0] for p in points]
    return min(ys), min(xs), max(ys) - min(ys)


def order_blocks(blocks: list[dict[str, Any]], *, rtl: bool = False) -> list[dict[str, Any]]:
    """Sort blocks into human reading order: top-to-bottom, then along the line.

    Paddle's native order is usually fine for single-column documents, but CVs
    love two-column layouts and those come back interleaved — contact details
    from the sidebar spliced into the middle of a job description. Bucketing by
    vertical band before sorting horizontally fixes that.

    ``rtl`` reverses the within-line direction. Reading order is a property of the
    script, not of the page: sorting an Arabic line by ascending x puts its last
    word first, so every line came out reversed and the extractor was handed word
    salad. Blocks are still bucketed into lines top-to-bottom either way — only
    the direction along each line flips.
    """
    if not blocks:
        return blocks

    heights = [b["_height"] for b in blocks if b.get("_height")]
    tolerance = (statistics.median(heights) * 0.6) if heights else 10.0

    ordered = sorted(blocks, key=lambda b: (b["_top"], b["_left"]))

    lines: list[list[dict[str, Any]]] = []
    for block in ordered:
        if lines and abs(block["_top"] - lines[-1][0]["_top"]) <= tolerance:
            lines[-1].append(block)
        else:
            lines.append([block])

    result: list[dict[str, Any]] = []
    for line in lines:
        result.extend(sorted(line, key=lambda b: b["_left"], reverse=rtl))
    return result


def run_ocr(image_paths: list[Path], lang: str = "en") -> dict[str, Any]:
    """OCR one or more images. Returns normalized pages, blocks and text.

    Every value in the returned structure is plain Python — no numpy leaks through.
    """
    engine = get_engine(lang)
    pages: list[dict[str, Any]] = []
    all_scores: list[float] = []
    page_texts: list[str] = []

    for page_index, image_path in enumerate(image_paths):
        results = engine.predict(str(image_path))

        blocks: list[dict[str, Any]] = []
        for result in results:
            texts = result.get("rec_texts", []) or []
            scores = result.get("rec_scores", []) or []
            polys = result.get("rec_polys", []) or []

            for text, score, poly in zip(texts, scores, polys):
                if not str(text).strip():
                    continue
                top, left, height = _poly_bounds(poly)
                blocks.append(
                    {
                        "text": str(text),
                        "confidence": float(score),
                        "bbox": [[float(p[0]), float(p[1])] for p in poly],
                        "_top": top,
                        "_left": left,
                        "_height": height,
                    }
                )

        blocks = order_blocks(blocks, rtl=is_rtl(lang))
        for block in blocks:  # sort keys were internal only
            block.pop("_top", None)
            block.pop("_left", None)
            block.pop("_height", None)

        scores_here = [b["confidence"] for b in blocks]
        all_scores.extend(scores_here)
        text_here = "\n".join(b["text"] for b in blocks)
        page_texts.append(text_here)

        pages.append(
            {
                "page_index": page_index,
                "source_image": str(image_path),
                "blocks": blocks,
                "mean_confidence": (
                    round(sum(scores_here) / len(scores_here), 4) if scores_here else None
                ),
                "text": text_here,
            }
        )

    return {
        "engine": "PaddleOCR",
        "lang": lang,
        "pages": pages,
        "mean_confidence": (
            round(sum(all_scores) / len(all_scores), 4) if all_scores else None
        ),
        "text": "\n\n".join(page_texts).strip(),
        "blocks": [b for page in pages for b in page["blocks"]],
    }


def _debug_main() -> int:
    """Isolated OCR check — verify Paddle works before trusting it inside the graph."""
    import json
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m agents.agent_a_cv_extraction.ingestion.ocr <image>")
        return 2

    target = Path(sys.argv[1])
    if not target.exists():
        print(f"no such file: {target}")
        return 2

    result = run_ocr([target])
    print(f"blocks: {len(result['blocks'])}   mean confidence: {result['mean_confidence']}")
    print("-" * 60)
    print(result["text"][:2000])
    print("-" * 60)
    if result["blocks"]:
        sample = result["blocks"][0]
        print("first block (check bbox is nested lists, not numpy repr):")
        print(json.dumps(sample, indent=2)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(_debug_main())
