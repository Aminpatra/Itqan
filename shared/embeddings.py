"""Embedding factory shared by every agent.

Mirrors ``shared/llm.py`` deliberately: same config object, same lazy-import
posture, same "fail loudly on a missing key" contract. Agent B embeds job
postings; Agent C will embed candidate profiles to query them. Both must land in
the *same vector space* or the similarity numbers are meaningless — which is the
whole reason this lives in ``shared/`` rather than inside Agent B.

Changing ``embedding_model`` or ``embedding_dims`` after any vector has been
written is a breaking change: the stored vectors are not comparable to new ones
and the pgvector column has a fixed dimension. Re-embedding the corpus is the
only remedy, so treat these as schema, not as tunables.
"""

from __future__ import annotations

from typing import Any, Sequence

from .config import Config


def build_embedder(config: Config | None = None, **overrides: Any) -> Any:
    """Build the shared OpenAI embeddings client.

    Imported lazily so that importing this module — which
    ``shared/__init__`` and the CLIs do eagerly — never requires an API key or
    the network. Only actually building an embedder does.
    """
    from langchain_openai import OpenAIEmbeddings

    config = config or Config()
    config.require_api_key()

    params: dict[str, Any] = {
        "model": config.embedding_model,
        # Pinned explicitly rather than left to the model default. The pgvector
        # column is declared vector(1536); a silent default change upstream
        # would fail at INSERT time, deep inside a scheduled cycle.
        "dimensions": config.embedding_dims,
    }
    params.update(overrides)
    return OpenAIEmbeddings(**params)


def embed_texts(embedder: Any, texts: Sequence[str]) -> list[list[float]]:
    """Embed a batch, returning one vector per input in the same order.

    Batching is the caller's job — this exists so that call sites and the
    offline fake share one signature, and so the order guarantee is stated
    somewhere rather than assumed.
    """
    if not texts:
        return []
    vectors = embedder.embed_documents(list(texts))
    if len(vectors) != len(texts):
        raise RuntimeError(
            f"embedder returned {len(vectors)} vectors for {len(texts)} inputs; "
            "refusing to guess the alignment"
        )
    return vectors
