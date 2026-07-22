"""A deterministic offline stand-in for OpenAIEmbeddings.

Near-duplicate detection is the one part of Agent B whose correctness is a
property of *vectors*, so its tests need real, comparable vectors — but calling
a paid embedding API from the suite would make it slow, non-deterministic, and
impossible to run without a key. This derives a stable 1536-d unit vector from
the sha256 of the text.

Two properties are guaranteed and relied on by the tests:

  * identical text → identical vector (so an unchanged posting embeds the same
    every cycle, which is what "second run does nothing" depends on)
  * the vectors are normalised, so a dot product IS cosine similarity

It does NOT attempt semantic similarity — two different strings get unrelated
vectors. Tests that need a controlled similarity build the vectors directly
rather than hoping the hash cooperates.
"""

from __future__ import annotations

import hashlib
import math

DIMS = 1536


def _vector_for(text: str) -> list[float]:
    # Expand sha256 into DIMS floats by hashing counter-suffixed copies. Cheap,
    # deterministic, and spread across the space rather than clustered.
    out: list[float] = []
    counter = 0
    while len(out) < DIMS:
        digest = hashlib.sha256(f"{counter}\x1f{text}".encode("utf-8")).digest()
        for i in range(0, len(digest), 2):
            if len(out) >= DIMS:
                break
            # Two bytes → a value in [-1, 1).
            raw = (digest[i] << 8) | digest[i + 1]
            out.append(raw / 32768.0 - 1.0)
        counter += 1

    norm = math.sqrt(sum(x * x for x in out)) or 1.0
    return [x / norm for x in out]


class FakeEmbedder:
    """Mirrors the surface Agent B uses from an embeddings client."""

    def __init__(self) -> None:
        self.embed_calls = 0
        self.embedded_texts: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls += 1
        self.embedded_texts.extend(texts)
        return [_vector_for(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.embed_calls += 1
        self.embedded_texts.append(text)
        return _vector_for(text)
