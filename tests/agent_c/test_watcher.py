"""The watch-mode selection logic — pure filesystem reasoning, fully offline.

What must never happen: re-processing a run (the gap file beside the profile is
the marker), half-reading a file still being written (the stability window), or
a permanently-broken profile being retried every poll forever.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from agents.agent_c_gap_analysis.watcher import (
    GAP_NAME,
    PROFILE_NAME,
    Watcher,
    find_unprocessed_profiles,
)


def make_run(root: Path, name: str, *, with_gap=False, age=60.0) -> Path:
    run = root / name
    run.mkdir(parents=True)
    profile = run / PROFILE_NAME
    profile.write_text("{}", encoding="utf-8")
    old = time.time() - age
    os.utime(profile, (old, old))
    if with_gap:
        (run / GAP_NAME).write_text("{}", encoding="utf-8")
    return profile


def test_only_profiles_without_gap_output_are_pending(tmp_path):
    fresh = make_run(tmp_path, "fresh")
    make_run(tmp_path, "done", with_gap=True)

    assert list(find_unprocessed_profiles(tmp_path)) == [fresh]


def test_a_file_inside_the_stability_window_waits_for_the_next_poll(tmp_path):
    make_run(tmp_path, "being_written", age=0.0)

    assert list(find_unprocessed_profiles(tmp_path)) == []
    # ... and is picked up once it has settled:
    assert list(find_unprocessed_profiles(tmp_path, now=time.time() + 10)) != []


def test_missing_root_is_a_clean_noop(tmp_path):
    assert list(find_unprocessed_profiles(tmp_path / "never_made")) == []


def test_baseline_skips_preexisting_profiles_unless_backfill(tmp_path):
    make_run(tmp_path, "old_run")
    processed: list[Path] = []

    watcher = Watcher(root=tmp_path, process=processed.append, log=lambda _: None)
    assert watcher.baseline() == 1
    watcher.poll_once()
    assert processed == [], "a pre-existing profile was billed without --backfill"

    # A NEW arrival after baseline IS picked up.
    new = make_run(tmp_path, "new_run")
    watcher.poll_once()
    assert processed == [new]

    # --backfill opts the old ones in. (new_run is still pending too — the
    # no-op fake above never wrote its gap file, so a fresh watcher rightly
    # sees both.)
    processed.clear()
    backfiller = Watcher(root=tmp_path, process=processed.append,
                         backfill=True, log=lambda _: None)
    assert backfiller.baseline() == 0
    backfiller.poll_once()
    assert {p.parent.name for p in processed} == {"old_run", "new_run"}


def test_a_failing_profile_is_set_aside_not_retried_every_poll(tmp_path):
    make_run(tmp_path, "poison")
    calls: list[Path] = []

    def explode(profile: Path) -> None:
        calls.append(profile)
        raise RuntimeError("boom")

    watcher = Watcher(root=tmp_path, process=explode, log=lambda _: None)
    watcher.baseline()
    # baseline skipped it (pre-existing); simulate it as a new arrival instead:
    watcher._skip.clear()

    watcher.poll_once()
    watcher.poll_once()
    watcher.poll_once()

    assert len(calls) == 1, "a permanently-failing profile was retried in-session"


def test_processing_marks_done_via_the_gap_file_itself(tmp_path):
    """The marker IS the output: a process() that writes skill_gap.json makes
    the run invisible to every later poll, with no registry to corrupt."""
    make_run(tmp_path, "run1")
    seen: list[str] = []

    def process(profile: Path) -> None:
        seen.append(profile.parent.name)
        (profile.parent / GAP_NAME).write_text("{}", encoding="utf-8")

    watcher = Watcher(root=tmp_path, process=process, backfill=True, log=lambda _: None)
    watcher.baseline()
    watcher.poll_once()
    watcher.poll_once()

    assert seen == ["run1"]
