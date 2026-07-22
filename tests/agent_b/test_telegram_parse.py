"""The Telegram adapter, offline against a fixture.

The view-count test is the load-bearing one. Its failure mode is not a wrong
value in a column — it is an ever-growing, permanent bill.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from agents.agent_b_job_ingest.sources.telegram import TelegramAdapter
from shared.config import Config
from tests.agent_b.fake_source_client import AllowAllRobots, FakeClient, fixture


def build(page: str | list[str] | None = None, **kwargs) -> TelegramAdapter:
    client = FakeClient({"t.me/s/": page or fixture("telegram_channel.html")})
    return TelegramAdapter(
        name="tg_example",
        handle="@ExampleChannel",
        source_group="test_group",
        client=client,
        robots=AllowAllRobots(),
        config=Config(),
        **kwargs,
    )


# ---------------------------------------------------------------------------
def test_view_counts_never_reach_the_raw_description():
    """View counts change on every fetch.

    If one entered ``raw_description``, the content hash of every posting in the
    channel would change every cycle, so every posting would be re-extracted and
    re-embedded forever — unbounded LLM and embedding spend producing no new
    information, and the "second run does nothing" property that phase 4's gate
    depends on would never hold. Structurally prevented by scoping the text to
    the message-text div; asserted here so a widened selector fails loudly.
    """
    postings = build().fetch().postings
    assert postings

    for posting in postings:
        assert "57" not in posting.raw_description
        assert "402" not in posting.raw_description
        assert "views" not in posting.raw_description.casefold()

    fixture_html = fixture("telegram_channel.html")
    counts = set(re.findall(r'class="tgme_widget_message_views">([^<]+)<', fixture_html))
    assert counts, "fixture no longer contains view spans; this test would pass vacuously"
    for posting in postings:
        for count in counts:
            assert count not in posting.raw_description


def test_a_photo_only_post_is_skipped_and_never_ocrd():
    """No text div means no text. Counted as skipped rather than dropped
    silently, so a sudden rise is visible; never routed to OCR, which would be a
    different and far more invasive kind of scraping than was authorised."""
    result = build().fetch()

    assert result.skipped == 1
    assert all("31804" not in p.source_url for p in result.postings)


def test_the_permalink_is_canonical_not_the_preview_path():
    """``/s/`` is the logged-out preview surface. Emitting it would mean a link
    from elsewhere to the same post canonicalizes differently, and the
    deterministic dedup path would miss the pair it exists to catch."""
    posting = build().fetch().postings[0]

    assert posting.source_url == "https://t.me/examplechannel/31803"
    assert "/s/" not in posting.source_url


def test_outbound_links_keep_document_order():
    """Live posts carry two blog links: the job the post is about, then an
    unrelated next job. Order is what lets phase 4 prefer the first rather than
    merging on any link and collapsing two real vacancies into one."""
    posting = build().fetch().postings[0]

    assert posting.outbound_links == (
        "https://example.test/2026/07/control-systems-engineer.html",
        "https://example.test/2026/07/office-administrator.html",
    )


def test_tracking_parameters_are_stripped_so_links_match_the_blog_permalink():
    """The whole deterministic dedup path rests on this: the Telegram link
    carries ``?utm_source=telegram`` and the blog's own URL does not."""
    posting = build().fetch().postings[0]

    assert all("utm_source" not in link for link in posting.outbound_links)


def test_dates_come_from_the_time_element():
    posting = build().fetch().postings[0]

    assert posting.posted_date == datetime(2026, 7, 19, 11, 1, 31, tzinfo=timezone.utc)
    assert posting.posted_date_text is None


def test_the_channel_is_not_treated_as_the_employer():
    """Same rule as the Atom adapter: the publisher is not the employer."""
    for posting in build().fetch().postings:
        assert posting.company is None


def test_title_is_selected_from_the_text_not_invented():
    posting = build().fetch().postings[0]

    assert posting.title == "Example Engineering Co - Control Systems Engineer"
    assert posting.title in posting.raw_description


def test_a_leading_reference_code_is_not_used_as_the_title():
    """Live posts open with an internal reference code. Using the first line
    blindly would fill the title column with codes — unreadable in the run log
    and useless to anything downstream matching on titles."""
    from agents.agent_b_job_ingest.sources.telegram import _select_title

    text = "03#1807-R1\nExample Engineering Co - Control Systems Engineer\nDetails below"
    assert _select_title(text) == "Example Engineering Co - Control Systems Engineer"


def test_title_selection_falls_back_rather_than_returning_nothing():
    """When every line is short there is no substantive line to prefer.
    Returning something the source actually wrote beats returning an empty
    title — and beats composing one."""
    from agents.agent_b_job_ingest.sources.telegram import _select_title

    assert _select_title("Short\nAlso short") == "Short"
    assert _select_title("") == ""


def test_handle_is_normalized_from_any_written_form():
    assert build().handle == "examplechannel"
