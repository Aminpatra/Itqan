"""Load `docs/knowledge/` into the assistant's knowledge base.

    python main.py knowledge --ingest
    python main.py knowledge --dry-run     # what would be sent, and what it costs

Run it after editing anything under `docs/knowledge/`. Unchanged passages are not
re-embedded, so running it when nothing has changed is free and takes a second —
which is the point: an ingest that costs real money every time is one people
avoid running, and documentation the assistant has not been given is worse than
documentation that does not exist, because nobody can tell from the outside.

The database table is created by `api/db.apply_migrations`, which runs when the
API boots. On a machine that has never started the API, start it once first.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from shared.config import Config

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "docs" / "knowledge"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python main.py knowledge",
        description="Load Itqan's own documentation into the assistant's knowledge base.")
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR,
                        help=f"where the documents live (default: {DEFAULT_DIR})")
    parser.add_argument("--ingest", action="store_true",
                        help="read, chunk, embed and store")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be ingested without embedding or writing")
    args = parser.parse_args(argv)

    if not args.ingest and not args.dry_run:
        parser.error("nothing to do: pass --ingest, or --dry-run to see what it would do")

    from . import knowledge as knowledge_module

    config = Config()
    directory: Path = args.dir
    if not directory.is_dir():
        print(f"error: {directory} is not a directory")
        return 2

    documents = knowledge_module.read_documents(directory)
    if not documents:
        # Loud, because the filename pattern is the easy thing to get wrong and a
        # silent zero here looks exactly like a successful run.
        print(f"error: no documents matched in {directory}.\n"
              f"       Files must be named <slug>.<en|ar>.<md|pdf>, "
              f"e.g. what-is-itqan.ar.md")
        return 1

    if args.dry_run:
        total = 0
        for doc in documents:
            pieces = knowledge_module.chunk(doc)
            total += len(pieces)
            print(f"  {doc.slug:<26} {doc.locale}  {len(pieces):>2} passage(s)  {doc.title}")
        print(f"\n{len(documents)} document(s), {total} passage(s). "
              f"Nothing embedded and nothing written.")
        return 0

    from shared.embeddings import build_embedder

    summary = knowledge_module.ingest(
        dsn=config.require_database_url(), directory=directory,
        embedder=build_embedder(config), config=config)

    # `embedded` is the number that matters on a re-run: it should be 0 when
    # nothing changed, and a non-zero value when nothing was edited means the
    # content hash is not doing its job.
    print(f"documents {summary['documents']}  passages {summary['chunks']}  "
          f"embedded {summary['embedded']}  removed {summary['deleted']}")
    if summary["embedded"] == 0:
        print("nothing changed since the last run")
    return 0
