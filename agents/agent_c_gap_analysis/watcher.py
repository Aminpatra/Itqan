"""Watch mode: run the gap analysis automatically when Agent A produces a profile.

The agents stay decoupled — Agent A does not call Agent C, and neither imports
the other. The handshake is the FILESYSTEM: Agent A's deliverable is
``output/<run_id>/candidate_profile.json``, and this watcher polls for run
directories that have a profile but no ``skill_gap.json`` yet. The gap output is
written INTO THE SAME run directory, so its presence is the "processed" marker —
no registry file to corrupt, and re-running the watcher never re-chews work.

Three deliberate behaviours:

* **New arrivals only, by default.** Profiles that already exist when the
  watcher starts are skipped (a fresh watcher on a directory of 35 old runs
  should not silently bill 35 gap analyses); ``--backfill`` opts in to them.
* **A stability window.** A profile is only picked up once its mtime is a couple
  of seconds old, so a file still being written is never half-read.
* **A failure is not retried in-session.** A profile that errors is logged and
  set aside; retrying it every poll would loop on a permanent error forever.
  Restarting the watcher retries it once more.

Polling (stdlib) rather than filesystem events: no new dependency, identical
behaviour across Windows/Unix/network drives, and a few seconds of latency is
irrelevant against a human-driven Agent A run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

PROFILE_NAME = "candidate_profile.json"
GAP_NAME = "skill_gap.json"

# A profile younger than this may still be mid-write; wait a poll.
STABILITY_SECONDS = 2.0


def find_unprocessed_profiles(
    root: Path,
    *,
    skip: set[Path] = frozenset(),
    now: float | None = None,
) -> Iterator[Path]:
    """Run directories holding a profile but no gap output yet.

    ``skip`` carries the watcher's baseline (pre-existing profiles) and its
    failed set; ``now`` is injectable for the stability-window tests.
    """
    now = time.time() if now is None else now
    if not root.exists():
        return
    for profile in sorted(root.glob(f"*/{PROFILE_NAME}")):
        if profile in skip:
            continue
        if (profile.parent / GAP_NAME).exists():
            continue  # already processed — the output beside it is the marker
        try:
            if now - profile.stat().st_mtime < STABILITY_SECONDS:
                continue  # possibly still being written; next poll will see it
        except OSError:
            continue  # vanished between glob and stat
        yield profile


@dataclass
class Watcher:
    root: Path
    process: Callable[[Path], None]     # invoked once per ready profile
    poll_seconds: float = 5.0
    backfill: bool = False
    log: Callable[[str], None] = print

    _skip: set[Path] = field(default_factory=set)

    def baseline(self) -> int:
        """Record what already exists so default mode only reacts to NEW work."""
        if self.backfill:
            return 0
        existing = list(find_unprocessed_profiles(self.root, now=float("inf")))
        self._skip.update(existing)
        return len(existing)

    def poll_once(self) -> int:
        """One scan; returns how many profiles were processed."""
        handled = 0
        for profile in find_unprocessed_profiles(self.root, skip=self._skip):
            self.log(f"  new profile: {profile.parent.name}")
            try:
                self.process(profile)
                handled += 1
            except Exception as exc:  # noqa: BLE001 - one bad profile must not kill the watch
                self.log(f"  ! {profile.parent.name} failed: {exc} — set aside "
                         "(restart the watcher to retry)")
                self._skip.add(profile)
        return handled

    def run_forever(self) -> None:
        skipped = self.baseline()
        if skipped:
            self.log(f"  ignoring {skipped} pre-existing unprocessed profile(s); "
                     "use --backfill to include them")
        self.log(f"  watching {self.root} every {self.poll_seconds:.0f}s "
                 "(Ctrl-C to stop)\n")
        try:
            while True:
                self.poll_once()
                time.sleep(self.poll_seconds)
        except KeyboardInterrupt:
            self.log("\n  watcher stopped.")
