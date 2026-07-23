"""The console Arabic shaper — display layer only, and provably harmless."""

from __future__ import annotations

import builtins

from shared.display import arabize


def test_latin_text_passes_through_untouched():
    assert arabize("Software Engineer at Example Co") == "Software Engineer at Example Co"
    assert arabize("") == ""


def test_arabic_is_reshaped_into_presentation_forms():
    """Logical-order letters become pre-shaped glyphs a dumb terminal can draw
    connected. The exact glyphs are the libraries' business; what we assert is
    the transformation happened: presentation-form codepoints in, logical
    codepoints out of, the visible string."""
    logical = "وظيفة"
    shaped = arabize(logical)

    assert shaped != logical
    assert any("ﭐ" <= ch <= "﻿" for ch in shaped), "no presentation forms produced"
    # Same letters, same count — shaping is not allowed to add or drop content.
    assert len(shaped) == len(logical)


def test_mixed_arabic_latin_line_keeps_the_latin_readable():
    shaped = arabize("وظيفة مهندس – Example Co")
    assert "Example Co" in shaped


def test_missing_libraries_degrade_to_identity(monkeypatch):
    """A rendering nicety must never crash a pipeline: without the libs, the
    original (correct, just unshaped) text comes back."""
    real_import = builtins.__import__

    def no_display_libs(name, *args, **kwargs):
        if name in ("arabic_reshaper", "bidi.algorithm", "bidi"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_display_libs)
    assert arabize("وظيفة") == "وظيفة"
