"""Coursera Catalog API adapter (source_type='api').

``api.coursera.org/api/courses.v1`` is public and unauthenticated (verified
live: 23,101 courses, cursor pagination via ``paging.next``). ``includes=
partnerIds`` returns partner names in ``linked.partners.v1``, so ``provider`` is
the teaching institution, not just "Coursera".

Consent model: this is an API governed by its terms, gated by a human
``terms_reviewed`` flag (see sources/config.py) — NOT by robots.txt, which
disallows /api/ for crawlers and does not govern a documented API client. So
this adapter does not perform a robots check; the config gate is the consent.

Scope: the English catalog only. Non-English descriptions extract poorly against
an English prompt, and the whole catalog is far more than one cycle needs; a
language filter is an honest source-configuration choice, not the adapter
editorialising about content.
"""

from __future__ import annotations

import json

from shared.config import Config
from shared.scraping.http import Blocked, PoliteClient, ResponseTooLarge, SourcePolicy

from .base import AdapterResult, BaseAdapter, RawCourse

API_PATH = "/api/courses.v1"
COURSE_FIELDS = "name,description,slug,partnerIds,primaryLanguages,workload"
COURSE_URL = "https://www.coursera.org/learn/{slug}"


class CourseraAdapter(BaseAdapter):
    source_type = "api"

    def __init__(
        self,
        *,
        name: str = "coursera",
        source_group: str = "coursera",
        base_url: str = "https://api.coursera.org",
        client: PoliteClient | None = None,
        config: Config | None = None,
        is_known_unchanged=None,
    ) -> None:
        super().__init__(name=name, source_group=source_group)
        self.config = config or Config()
        self.base_url = base_url.rstrip("/")
        self._client = client
        self._is_known_unchanged = is_known_unchanged

    @property
    def api_url(self) -> str:
        return f"{self.base_url}{API_PATH}"

    def _ensure_client(self) -> PoliteClient:
        if self._client is None:
            self._client = PoliteClient(
                source=self.name,
                policy=SourcePolicy(min_interval_s=2.0, max_bytes=self.config.max_response_bytes),
                config=self.config,
            )
        return self._client

    def _fetch(self, result: AdapterResult, *, limit: int | None) -> None:
        client = self._ensure_client()
        start = "0"
        seen: set[str] = set()

        for _ in range(self.config.coursera_max_pages):
            params = {
                "start": start,
                "limit": str(self.config.coursera_page_size),
                "fields": COURSE_FIELDS,
                "includes": "partnerIds",
            }
            try:
                body = client.get_text(self.api_url, params=params)
            except (Blocked, ResponseTooLarge) as exc:
                result.fail(str(exc))
                return
            except Exception as exc:  # noqa: BLE001
                result.fail(f"{type(exc).__name__}: {exc}")
                return

            result.pages_fetched += 1
            result.bytes_fetched = client.bytes_fetched

            try:
                data = json.loads(body)
            except json.JSONDecodeError as exc:
                result.fail(f"catalog response was not JSON: {exc}")
                return

            partners = _partner_names(data)
            elements = data.get("elements") or []
            if not elements:
                return

            for el in elements:
                course = self._parse(el, partners)
                if course is None:
                    result.skipped += 1
                    continue
                if course.source_url in seen:
                    continue
                seen.add(course.source_url)
                if self._is_known_unchanged and self._is_known_unchanged(course):
                    continue
                result.courses.append(course)
                if limit is not None and len(result.courses) >= limit:
                    return

            nxt = (data.get("paging") or {}).get("next")
            if not nxt:
                return
            start = str(nxt)

    def _parse(self, el: dict, partners: dict[str, str]) -> RawCourse | None:
        slug = (el.get("slug") or "").strip()
        name = (el.get("name") or "").strip()
        description = (el.get("description") or "").strip()
        if not slug or (not name and not description):
            return None

        langs = el.get("primaryLanguages") or []
        primary = _iso2(langs[0]) if langs else None
        # English catalog only — see module docstring.
        if primary != "en":
            return None

        partner_ids = el.get("partnerIds") or []
        provider = next(
            (partners[p] for p in partner_ids if p in partners),
            "Coursera",
        )

        return RawCourse(
            source=self.name,
            source_group=self.source_group,
            source_type=self.source_type,
            source_url=COURSE_URL.format(slug=slug),
            name=name,
            raw_description=description,
            provider=provider,
            level=None,               # not exposed by this API; extracted or unknown
            primary_language=primary,
            attribution=f"{provider} via Coursera",
            license=None,             # proprietary; we catalog facts + a link only
        )


def _partner_names(data: dict) -> dict[str, str]:
    linked = (data.get("linked") or {}).get("partners.v1") or []
    return {str(p.get("id")): (p.get("name") or "").strip() for p in linked if p.get("id")}


def _iso2(code: str) -> str | None:
    code = (code or "").strip().lower()
    return code[:2] if len(code) >= 2 else None
