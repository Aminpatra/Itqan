"""Fetch a posting's ROOT job page, so its skills come from the source of truth.

An aggregator (el7far) often carries only a thin summary — worst for an index
roundup that lists role titles with no requirements. When such a post links to
the employer's own single job page, the real skills are there. This module
fetches that page, BOUNDED and robots-respecting:

  * only a link that looks like a SINGLE job-detail page is ever followed —
    ``/job/<id>``, ``/JobDetail/<id>``, ``/viewjob``, ``/vacancy/<slug>`` — never a
    bare careers HUB (``/careers``, ``/jobs``) and never social/messaging links;
  * every host's robots.txt is consulted through the same fail-closed
    ``RobotsPolicy`` the adapters use; a disallow (or any error) yields None and
    the caller keeps the aggregator's skills. Enrichment never deletes anything.

It does NOT deep-crawl: it fetches the one linked page and stops. Whether that
page turns out to be a single job (enrich) or a listing (skip) is decided by the
caller re-extracting it — this module only fetches text.
"""

from __future__ import annotations

from urllib.parse import urlsplit

from selectolax.parser import HTMLParser

from shared.config import Config
from shared.scraping.http import Blocked, PoliteClient, SourcePolicy
from shared.scraping.robots import RobotsPolicy

# A fetched job page is capped like the adapters cap descriptions: enough for the
# requirements, bounded so one huge page cannot blow up the extraction prompt.
MAX_PAGE_CHARS = 20_000

# Domains that are never a job page: sharing widgets and messaging. Matched as
# whole domains (exact or as a parent domain), NEVER as substrings of the URL.
# Substring matching silently discarded real employer job pages — measured:
# `careers.recruitment.medgulf.com` contains "t.me" inside "recruitmen(t.me)dgulf"
# and `hr.omanunix.com` contains "x.com" inside "omanuni(x.com)". Both were dropped
# as "social", with no counter to notice it.
_SOCIAL_DOMAINS = frozenset({
    "facebook.com", "twitter.com", "x.com", "t.me", "telegram.me", "telegram.org",
    "wa.me", "whatsapp.com", "api.whatsapp.com", "chat.whatsapp.com",
    "linkedin.com", "instagram.com", "youtube.com", "pinterest.com",
    "blogger.com", "google.com",
})


def _is_social(host: str) -> bool:
    """Exact domain or a subdomain of one — `careers.medgulf.com` is not social
    just because some social domain is a substring of it."""
    host = (host or "").lower().removeprefix("www.")
    return any(host == d or host.endswith("." + d) for d in _SOCIAL_DOMAINS)

# Path segments that name a job listing. A job DETAIL page has one of these
# followed by an id/slug (or is a viewjob/jobdetail page); a bare hub does not.
_JOB_KEYWORDS = {
    "job", "jobs", "vacancy", "vacancies", "opening", "openings",
    "position", "positions", "externaljobs",
}


def _looks_like_job_detail(url: str) -> bool:
    parts = urlsplit(url)
    low = parts.path.lower()
    if "viewjob" in low or "jobdetail" in low:
        return True
    segs = [s for s in parts.path.split("/") if s]
    for i, seg in enumerate(segs):
        if seg.lower() in _JOB_KEYWORDS and i < len(segs) - 1 and segs[i + 1].strip():
            return True   # /job/<id>, /jobs/<slug>, /…/job/<id>
    return False


def candidate_job_link(outbound_links, self_host: str) -> str | None:
    """The first outbound link that is a single EXTERNAL job-detail page, or None.

    External (a different host from the post itself), not social/messaging, and
    shaped like a job detail rather than a careers hub. First match wins — the
    adapters already put the posting's primary link first."""
    self_host = (self_host or "").lower()
    for url in outbound_links or ():
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https"):
            continue
        host = parts.netloc.lower()
        if _is_social(host):
            continue
        if not host or host == self_host:
            continue  # same-site links here are el7far nav/search, not a root page
        if _looks_like_job_detail(url):
            return url
    return None


def _readable_text(html: str) -> str:
    if not html.strip():
        return ""
    tree = HTMLParser(html)
    for node in tree.css("script, style, noscript, nav, header, footer, svg"):
        node.decompose()
    text = tree.text(separator="\n", strip=True) or ""
    text = "\n".join(line for line in (ln.strip() for ln in text.splitlines()) if line)
    return text[:MAX_PAGE_CHARS]


class RootFetcher:
    """Fetches root job pages, one polite + robots-checked client PER HOST.

    Best-effort by contract: ``fetch`` returns the page text or None, and never
    raises — a blocked or broken root page must leave the aggregator's skills in
    place, not sink the posting."""

    def __init__(self, config: Config | None = None, *, interval_s: float = 1.0) -> None:
        self.config = config or Config()
        self.interval_s = interval_s
        self._hosts: dict[str, tuple[PoliteClient, RobotsPolicy]] = {}
        # Hosts that have refused us this cycle. `Blocked` means 403/429/robots —
        # a decision by the operator, not a transient hiccup — and the project's
        # own HTTP contract says retrying it harder is exactly what we must not
        # do. Without this the refusal was swallowed per URL and every subsequent
        # posting linking there cost another request: measured, 7 requests to a
        # host that refused on the first.
        self._refused: set[str] = set()
        self.blocked = 0

    def _for_host(self, url: str) -> tuple[PoliteClient, RobotsPolicy]:
        host = urlsplit(url).netloc.lower()
        if host not in self._hosts:
            client = PoliteClient(
                source=f"root:{host}",
                policy=SourcePolicy(
                    min_interval_s=self.interval_s, max_bytes=self.config.max_response_bytes
                ),
                config=self.config,
            )
            robots = RobotsPolicy(client, user_agent=self.config.user_agent)
            self._hosts[host] = (client, robots)
        return self._hosts[host]

    def fetch(self, url: str) -> str | None:
        host = urlsplit(url).netloc.lower()
        if host in self._refused:
            return None               # already said no; do not ask again
        try:
            client, robots = self._for_host(url)
            if not robots.can_fetch(url):
                # robots is a per-path answer, so this does not condemn the host —
                # but it is still a refusal, and we keep the aggregator's skills.
                return None
            html = client.get_text(url)
        except Blocked:
            # An explicit refusal. Remember it for the rest of the cycle.
            self._refused.add(host)
            self.blocked += 1
            return None
        except Exception:              # noqa: BLE001 - best-effort; never sink a posting
            return None
        return _readable_text(html) or None

    def close(self) -> None:
        for client, _ in self._hosts.values():
            try:
                client.close()
            except Exception:          # noqa: BLE001
                pass
