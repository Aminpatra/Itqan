"""Copy the local corpus into the deployed database.

    python scripts/seed_remote_db.py --target "postgresql://itqan:pw@vps-host:5432/itqan"

Measured 2026-07-31, the local database is **1,706 MB**, and `esco_labels` is
**1,625 MB of it** — 100,350 label rows each carrying a `vector(1536)`, plus a
798 MB HNSW index. On the VPS that is 1.7 GB of a 40 GB disk, so the default here
copies **everything**: every ESCO embedding, and with it the embedding tier that
maps skills the exact and alt-label tiers miss. That tier produced 851 of 1,306
course-skill mappings and 193 of 286 job-skill mappings — about two thirds of
everything that maps — and it is what lets the ingestion agents keep mapping newly
seen skills once cron starts running them.

`--trim` exists for a small database (a 0.5 GB free plan cannot take the full
thing; Neon suspends the project rather than billing). It keeps every label ROW,
so the lexical tiers are untouched, and drops only the embeddings of the 86,411
ALTERNATE labels — the 13,939 preferred ones keep theirs. Measured: 323 MB, and
Agent E returned 8 of 8 identical recommendations against the full original. Use
it only if the target cannot hold 1.7 GB; on this VPS it is not needed.

Nothing here writes to the source.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_ESCO_LABELS = "esco_labels"


def _pg_dump(source: str, out: Path, *, exclude_labels: bool) -> None:
    cmd = [
        "pg_dump", source,
        "--no-owner", "--no-privileges", "--no-acl",
        # Clean + if-exists so a re-run replaces rather than colliding with
        # everything that already exists.
        "--clean", "--if-exists",
        "-f", str(out),
    ]
    if exclude_labels:
        cmd.insert(2, f"--exclude-table-data={_ESCO_LABELS}")
    print(f"  pg_dump -> {out.name}" + ("  (esco_labels data excluded)" if exclude_labels else ""))
    subprocess.run(cmd, check=True)


def _psql(target: str, script: Path) -> None:
    print(f"  restoring {script.name}")
    subprocess.run(["psql", target, "-v", "ON_ERROR_STOP=1", "-q", "-f", str(script)],
                   check=True)


def _copy_labels_trimmed(source: str, target: str, *, batch: int = 5_000) -> None:
    """Stream esco_labels across, dropping ALTERNATE-label embeddings."""
    import psycopg

    # pgvector does not index NULLs, so the HNSW index on the target covers only
    # the preferred labels and shrinks with them — that index is 798 MB locally
    # and is most of what does not fit a small plan.
    keep = "CASE WHEN is_preferred THEN embedding END"

    with psycopg.connect(source) as src, psycopg.connect(target) as dst:
        total = src.execute(f"SELECT count(*) FROM {_ESCO_LABELS}").fetchone()[0]
        print(f"  esco_labels: {total:,} rows (embeddings kept for preferred labels only)")

        copied = 0
        with src.cursor(name="labels") as read:      # server-side cursor: 100k rows
            read.itersize = batch
            read.execute(
                f"SELECT label_key, label, esco_uri, is_preferred, {keep} FROM {_ESCO_LABELS}")
            with dst.cursor().copy(
                f"COPY {_ESCO_LABELS} (label_key, label, esco_uri, is_preferred, embedding) "
                "FROM STDIN"
            ) as write:
                for row in read:
                    write.write_row(row)
                    copied += 1
                    if copied % 20_000 == 0:
                        print(f"    {copied:,}/{total:,}")
        dst.commit()
        print(f"    {copied:,}/{total:,} done")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="seed_remote_db",
        description="Copy the local corpus into the deployed database.")
    ap.add_argument("--target", required=True, help="Destination connection string")
    ap.add_argument("--source", default=os.getenv("ITQAN_DATABASE_URL"),
                    help="Source database (default: $ITQAN_DATABASE_URL)")
    ap.add_argument("--trim", action="store_true",
                    help="Drop ALTERNATE-label embeddings: 1,706 MB -> 323 MB. Only for a "
                         "target that cannot hold the full corpus; costs the embedding "
                         "tier's reach when mapping newly-seen skills.")
    ap.add_argument("--keep-dump", action="store_true",
                    help="Leave the intermediate .sql file for inspection")
    args = ap.parse_args(argv)

    if not args.source:
        print("  --source or ITQAN_DATABASE_URL is required", file=sys.stderr)
        return 2

    # Checked up front rather than discovered halfway through a 1.7 GB dump. On a
    # machine where Postgres only ever ran in Docker these are genuinely absent,
    # which is the normal case here.
    from shutil import which

    if missing := [t for t in ("pg_dump", "psql") if which(t) is None]:
        print(f"\n  {' and '.join(missing)} not found on PATH.\n"
              "  Install the Postgres client tools, or run them from the container:\n"
              "    docker exec itqan-pg pg_dump -U postgres -d itqan ...\n",
              file=sys.stderr)
        return 2

    dump = PROJECT_ROOT / "output" / "seed.sql"
    dump.parent.mkdir(parents=True, exist_ok=True)

    print("\n  Itqan | seeding the deployed database")
    print(f"  mode: {'TRIMMED (~323 MB)' if args.trim else 'FULL (~1.7 GB)'}\n")
    try:
        _pg_dump(args.source, dump, exclude_labels=args.trim)
        _psql(args.target, dump)
        if args.trim:
            _copy_labels_trimmed(args.source, args.target)
    except subprocess.CalledProcessError as exc:
        print(f"\n  failed: {exc}\n", file=sys.stderr)
        return 1
    finally:
        if dump.exists() and not args.keep_dump:
            dump.unlink()

    print("\n  seeded. Confirm it landed:")
    print('    psql "$TARGET" -c "SELECT pg_size_pretty(pg_database_size(current_database()))"')
    print('    psql "$TARGET" -c "SELECT count(*) FROM courses"      # expect 2,099')
    print('    psql "$TARGET" -c "SELECT count(embedding) FROM esco_labels"  # expect 100,350\n')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
