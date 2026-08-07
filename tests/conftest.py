import os
import sys
from pathlib import Path

# Tests import `shared` and `agents` as top-level packages.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# A user agent for the suite, set unconditionally.
#
# `Config.user_agent` defaults to a string containing "contact-not-configured",
# and `require_identified_user_agent()` refuses to make live requests with it —
# a deliberate guard, because a scraper that does not identify itself leaves a
# site operator no option but to block it.
#
# The tests never make a real request (every client is a fake), but the guard
# fires before the fake is reached, so six of them need a configured value. On a
# developer machine `.env` supplies one and they pass; on a fresh clone or in CI
# there is no `.env` and they fail. That is a test defect, not a CI defect: a
# suite whose result depends on an untracked file on one person's laptop is not
# telling you what you think it is. It went unnoticed exactly that long.
#
# Set here rather than in a fixture because several test modules construct a
# `Config()` at import time, which happens during collection — before any fixture
# runs. And set unconditionally rather than with `setdefault`, so the suite
# behaves identically everywhere; a developer's real contact address making the
# tests pass differently from CI is the bug being fixed.
#
# `load_dotenv` does not override existing environment variables, so this also
# wins over a local `.env`.
os.environ["ITQAN_USER_AGENT"] = "ItqanTestBot/0.0 (+tests@itqan.invalid)"
