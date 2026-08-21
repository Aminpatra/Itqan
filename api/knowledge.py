"""What Itqan is, in Itqan's own words — ingested, retrieved, and quotable.

Agent S could only ever answer questions about the person asking. Asked what
Itqan *is*, it declined, and its prompt said so outright. That was the right
answer while the fact sheet was the only thing it could draw on, and the wrong
answer to a reasonable question from someone using the product.

**This is a second source, not a loosening of the fence.** The rule Agent S has
always followed is that the model may state only what it was actually shown; what
changes here is the size of that set, not the rule. Retrieved passages are
verbatim from documents in `docs/knowledge/` — written by us, reviewed as source,
version-controlled beside the code they describe. They are not scraped, not
user-supplied, and not the model's own recollection of career websites.

**Nothing here is per-user.** The table has no `user_id` and no foreign key to
`app_users`, so the isolation question this project asks of every read surface
does not arise: this one cannot leak one person's data to another because it
never held any.

**The one rule for the content itself: no figure that rots.** Anything in these
documents becomes quotable by Hud as present-tense fact, because `verify_answer`
accepts it. So corpus sizes and test counts stay out; the live numbers keep
coming from the fact sheet, which is measured per user per request.
"""

from __future__ import annotations

import hashlib
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

from shared.config import Config

# A chunk is one `##` section. Long sections are split again at a paragraph
# boundary rather than mid-sentence: a passage cut in half mid-claim is worse
# than a slightly longer one, because the model is asked to quote from it.
MAX_CHUNK_CHARS = 1_200

# `<slug>.<locale>.md` — the locale is part of the filename rather than
# front-matter so the pairing is visible in a directory listing and a missing
# translation is obvious at a glance.
_NAME = re.compile(r"^(?P<slug>[a-z0-9-]+)\.(?P<locale>en|ar)\.(?P<ext>md|pdf)$")


@dataclass(frozen=True)
class Document:
    slug: str
    locale: str
    title: str
    text: str


@dataclass(frozen=True)
class Passage:
    """One retrieved chunk, with what it took to retrieve it."""

    doc_slug: str
    title: str
    locale: str
    text: str
    similarity: float


# ---------------------------------------------------------------------------
# reading and chunking
# ---------------------------------------------------------------------------
def read_documents(directory: Path) -> list[Document]:
    """Every knowledge document in a directory, Markdown or PDF.

    A file whose name does not carry a locale is SKIPPED rather than guessed at.
    Guessing would put an Arabic answer's source in the English pool, where it
    would be retrieved for the wrong questions and never for the right ones.
    """
    out: list[Document] = []
    for path in sorted(directory.glob("*")):
        match = _NAME.match(path.name)
        if not match:
            continue
        if match.group("ext") == "pdf":
            from agents.agent_a_cv_extraction.ingestion.pdf_text import extract_text

            text, _pages = extract_text(path)
        else:
            text = path.read_text(encoding="utf-8")
        text = text.strip()
        if not text:
            continue
        out.append(Document(slug=match.group("slug"), locale=match.group("locale"),
                            title=_title_of(text, fallback=match.group("slug")),
                            text=text))
    return out


def _title_of(text: str, *, fallback: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback.replace("-", " ")


def chunk(document: Document) -> list[str]:
    """One passage per `##` section, each carrying the document's title.

    The title is PREPENDED to every chunk, and that is not padding. A passage is
    retrieved alone and shown alone: "Deleting removes the file itself, not just
    the entry in the list" is ambiguous until "What Itqan stores, and who can see
    it" sits above it. It also gives the embedding the document's subject, which
    is most of what makes a short passage findable at all.
    """
    body = document.text
    # Drop the H1: it becomes the prefix on every chunk instead.
    body = re.sub(r"\A#\s+.*?(?:\n|$)", "", body, count=1)

    sections = re.split(r"\n(?=##\s)", body)
    chunks: list[str] = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        for part in _split_long(section):
            chunks.append(f"{document.title}\n\n{part}")
    return chunks


def _split_long(section: str) -> list[str]:
    """Split at blank lines, never mid-sentence, and only when over the cap."""
    if len(section) <= MAX_CHUNK_CHARS:
        return [section]
    out: list[str] = []
    current = ""
    for para in section.split("\n\n"):
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) > MAX_CHUNK_CHARS and current:
            out.append(current)
            current = para
        else:
            current = candidate
    if current:
        out.append(current)
    return out


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# connection plumbing, mirroring shared/job_market.py
# ---------------------------------------------------------------------------
@contextmanager
def _connection(dsn: str) -> Iterator[Any]:
    import psycopg
    from psycopg.rows import dict_row

    # AUTOCOMMIT, and this is not a preference — it is the fix for a bug this
    # project has now hit three times.
    #
    # psycopg opens an implicit transaction on a connection's first statement. In
    # `ingest` the first statement is the SELECT that reads existing hashes, so
    # without this every later `with conn.transaction()` nests inside it as a
    # SAVEPOINT — and releasing a savepoint commits nothing. The connection then
    # closes without a commit and the whole ingest is silently rolled back, while
    # the summary happily reports the rows it thought it wrote.
    #
    # It cost Agent B a silent census, then the destination prune ("deleted 432
    # postings" with 432 rows still there), which is why `JobStore.connect` is
    # autocommit today. There is no quieter way to lose data: the log says the
    # work was done. Every writer here still wraps its work in `transaction()`,
    # so batch atomicity is unchanged.
    conn = psycopg.connect(dsn, row_factory=dict_row, autocommit=True)
    try:
        try:
            from pgvector.psycopg import register_vector

            register_vector(conn)
        except Exception:
            # Tolerated exactly as JobStore and job_market tolerate it: a database
            # without the extension should fail on the query, not the import.
            pass
        yield conn
    finally:
        conn.close()


def _rows(conn: Any, sql: str, params: Any) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------
def ingest(*, dsn: str, directory: Path, embedder: Any,
           config: Optional[Config] = None) -> dict[str, int]:
    """Load `docs/knowledge/` into the database. Idempotent, and cheap when nothing changed.

    **Unchanged chunks are not re-embedded.** The content-hash gate is the same
    one Agents B and D use, for the same reason: running the ingest after every
    documentation edit has to be cheap enough that people actually do it, and
    re-embedding a whole corpus to fix one typo is how that habit dies.

    A document that has LOST chunks has its extras deleted, so shortening a file
    does not leave orphaned passages that nothing links to but retrieval can
    still find.
    """
    config = config or Config()
    documents = read_documents(directory)
    summary = {"documents": len(documents), "chunks": 0, "embedded": 0, "deleted": 0}

    with _connection(dsn) as conn:
        for doc in documents:
            pieces = chunk(doc)
            summary["chunks"] += len(pieces)

            existing = {
                r["ordinal"]: r["content_hash"] for r in _rows(
                    conn,
                    "SELECT ordinal, content_hash FROM app_knowledge_chunks "
                    "WHERE doc_slug = %s AND locale = %s",
                    (doc.slug, doc.locale))
            }

            fresh: list[tuple[int, str, str]] = []
            for ordinal, text in enumerate(pieces):
                digest = _hash(text)
                if existing.get(ordinal) == digest:
                    continue
                fresh.append((ordinal, text, digest))

            vectors: list[list[float]] = []
            if fresh:
                from shared.embeddings import embed_texts

                vectors = embed_texts(embedder, [t for _, t, _ in fresh])
                summary["embedded"] += len(vectors)

            with conn.transaction():
                with conn.cursor() as cur:
                    for (ordinal, text, digest), vector in zip(fresh, vectors):
                        cur.execute(
                            """
                            INSERT INTO app_knowledge_chunks
                                (chunk_id, doc_slug, title, locale, ordinal, text,
                                 content_hash, embedding, ingested_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, now())
                            ON CONFLICT (doc_slug, locale, ordinal) DO UPDATE SET
                                title = EXCLUDED.title,
                                text = EXCLUDED.text,
                                content_hash = EXCLUDED.content_hash,
                                embedding = EXCLUDED.embedding,
                                ingested_at = now()
                            """,
                            (f"kb_{doc.slug}_{doc.locale}_{ordinal}", doc.slug,
                             doc.title, doc.locale, ordinal, text, digest, vector))
                    cur.execute(
                        "DELETE FROM app_knowledge_chunks "
                        " WHERE doc_slug = %s AND locale = %s AND ordinal >= %s",
                        (doc.slug, doc.locale, len(pieces)))
                    summary["deleted"] += cur.rowcount
    return summary


# ---------------------------------------------------------------------------
# retrieval
# ---------------------------------------------------------------------------
def search(question: str, *, dsn: str, embedder: Any, locale: str = "en",
           config: Optional[Config] = None) -> list[Passage]:
    """Passages that might answer this question, or nothing at all.

    **The floor drops turns with no question in them. It does NOT separate
    product questions from results questions, and it cannot.** Measured over the
    real documents on 2026-08-21, the two populations overlap and the overlap is
    structural rather than noise: the documentation explains the results
    features, so "what are my gaps?" matches the passage defining what a gap is
    (0.561) more strongly than "do employers see my CV?" matches the privacy page
    (0.462). A threshold high enough to try to split them would cut off real
    product questions, Arabic ones first — they score lower throughout.

    So this drops greetings ("Hi" 0.294, "thanks" 0.234) and passes everything
    else, and the separation is the prompt's job: it is told to answer from the
    FACTS when the question is about the person's own results, with the ABOUT
    block as background. See `shared/config.py` for the measurements.

    **The question's language wins, but is not a hard filter.** A translation may
    be missing, and answering from the other language beats declining — so the
    matching locale is preferred in the ordering, and the rest can still surface.
    """
    config = config or Config()
    if not (question or "").strip():
        return []

    vector = embedder.embed_query(question)
    with _connection(dsn) as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                # Same three rules as every other vector read in this system:
                # ef_search raised so a filtered HNSW scan cannot quietly return
                # fewer rows than asked, similarity is 1 - cosine DISTANCE, and
                # the filter implies the partial index predicate.
                cur.execute("SET LOCAL hnsw.ef_search = 100")
                cur.execute(
                    """
                    SELECT doc_slug, title, locale, text,
                           1 - (embedding <=> %(emb)s::vector) AS similarity
                      FROM app_knowledge_chunks
                     WHERE embedding IS NOT NULL
                     ORDER BY (CASE WHEN locale = %(locale)s THEN 0 ELSE 1 END),
                              embedding <=> %(emb)s::vector
                     LIMIT %(k)s
                    """,
                    {"emb": vector, "locale": locale,
                     "k": config.assistant_knowledge_top_k})
                rows = [dict(r) for r in cur.fetchall()]

    floor = config.assistant_knowledge_min_similarity
    return [Passage(doc_slug=r["doc_slug"], title=r["title"], locale=r["locale"],
                    text=r["text"], similarity=float(r["similarity"]))
            for r in rows if float(r["similarity"]) >= floor]


def knowledge_block(passages: list[Passage]) -> str:
    """The passages as the model sees them, or "" when nothing cleared the floor.

    An empty string means the prompt carries no ABOUT block at all, which is the
    difference between "the documentation does not cover this" and "here are
    three irrelevant passages, do your best".
    """
    if not passages:
        return ""
    return "\n\n".join(p.text for p in passages)
