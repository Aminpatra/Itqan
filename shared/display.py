"""Console-display helpers. Currently one job: making Arabic readable in
terminals that cannot shape it.

Terminal renderers like VS Code's integrated terminal (xterm.js) and the legacy
Windows console do no contextual shaping and no bidi layout — they draw each
Arabic letter in its ISOLATED form, left to right, which reads as disconnected
gibberish. The data was never wrong; the renderer is.

``arabize`` compensates AT PRINT TIME ONLY: it converts logical-order Arabic to
pre-shaped presentation forms in visual order, which dumb terminals then display
connected and right-to-left. Three rules keep it safe:

* **Display layer only.** Nothing that gets stored, hashed, embedded, matched
  or written to a file may pass through this — presentation forms would corrupt
  every downstream comparison. Files stay pure logical Arabic (they already
  render correctly in any real viewer).
* **No-op without Arabic.** A cheap regex gate, so the common all-Latin line
  costs one failed search.
* **Degrades to identity.** If the two display libraries are missing, the
  original text is returned — a rendering nicety must never be able to crash a
  pipeline.
"""

from __future__ import annotations

import re

# Arabic block + supplements + Arabic Extended-A.
_ARABIC = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ]")


def arabize(text: str) -> str:
    """Return ``text`` shaped for a non-shaping terminal. Print-only; never store."""
    if not text or not _ARABIC.search(text):
        return text
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
    except ImportError:
        return text
    try:
        return get_display(arabic_reshaper.reshape(text))
    except Exception:
        # Whatever went wrong, unreadable-but-correct beats a crashed CLI.
        return text
