"""The Blogger Atom adapter, offline against a fixture.

The first test is the one that matters most in this file. Everything else here
guards against a broken parse, which is loud; that one guards against a parse
that succeeds and is wrong, which is silent and would corrupt the scam filter.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest

from agents.agent_b_job_ingest.sources.el7far import El7farAdapter, canonical_url
from shared.config import Config
from tests.agent_b.fake_source_client import _STAMP, recent, AllowAllRobots, FakeClient, fixture




def build(feed: str | list[str] | None = None, **kwargs) -> El7farAdapter:
    client = FakeClient({"/feeds/posts/default": feed or recent(fixture("el7far_feed.xml"))})
    return El7farAdapter(
        base_url="https://example.test",
        client=client,
        robots=AllowAllRobots(),
        config=Config(),
        **kwargs,
    )


# ---------------------------------------------------------------------------
def test_the_feed_author_never_becomes_the_company():
    """The single most consequential assertion in the adapter.

    ``atom:author/name`` is the blogger — on the live feed, a personal name.
    Mapping it to ``company`` would invent an employer that does not exist, and
    because ``no_employer_named`` is a legitimacy signal it would simultaneously
    switch off the check that would have caught the invention. A fabrication
    that disables its own detector is the worst failure available here, so it is
    asserted directly rather than trusted to a comment.
    """
    result = build().fetch()
    assert result.postings

    for posting in result.postings:
        assert posting.company is None, "the blog author was mapped to company"

    blob = " ".join(f"{p.company}" for p in result.postings)
    assert "Mahmoud" not in blob


def test_alternate_link_is_the_posting_not_the_comments_page():
    """Blogger emits several text/html links per entry. ``rel="replies"`` points
    at the comment form for the same post; taking the first html link would use
    it as the posting's identity, and the ``#comment-form`` fragment would make
    one post look like two."""
    posting = build().fetch().postings[0]

    assert posting.source_url == "https://example.test/2026/07/control-systems-engineer.html"
    assert "comment" not in posting.source_url


def test_dates_are_real_timestamps_normalized_to_utc():
    """A real ``published`` value means there is no relative-date string to
    re-resolve on each fetch — the trap that re-hashes every posting per cycle."""
    feed = recent(fixture("el7far_feed.xml"))
    # Computed from the feed rather than hard-coded, so this asserts the PARSE —
    # a real timestamp, converted from the feed's +03:00 to UTC — instead of a
    # fixed instant that stops being reachable once the fixture ages out of the
    # lookback window. See `recent`.
    expected = max(datetime.fromisoformat(x)
                   for x in re.findall(_STAMP, feed)).astimezone(timezone.utc)

    posting = build(feed).fetch().postings[0]

    assert posting.posted_date == expected
    assert posting.posted_date.tzinfo == timezone.utc
    assert posting.posted_date_text is None


def test_labels_are_captured_but_left_as_telemetry():
    """Labels are collected and deliberately not used for sector or skills yet.

    A model shown the publisher's label echoes it back, after which "the label
    disagrees with the extraction" can never fire. Agreement has to be measured
    against an unprimed baseline first.
    """
    posting = build().fetch().postings[0]
    assert posting.labels == ("Engineering Jobs", "Muscat")


def test_script_and_style_text_is_not_treated_as_content():
    posting = build().fetch().postings[0]

    assert "var tracking" not in posting.raw_description
    assert "color:red" not in posting.raw_description
    assert "Control Systems Engineer" in posting.raw_description


def test_outbound_links_are_absolute_and_ordered():
    """Relative hrefs resolve against the post URL, and order is preserved
    because the dedup path treats the first link differently from the rest."""
    posting = build().fetch().postings[0]

    assert posting.outbound_links == (
        "https://careers.example.test/roles/4821",
        "https://example.test/2026/07/another-unrelated-role.html",
    )


def test_a_posting_never_lists_itself_as_an_outbound_link():
    """Live posts link to themselves. Phase 4's rule — the linker is the
    duplicate, the target is canonical — would make such a posting its own
    duplicate, and the schema's CHECK (duplicate_of IS DISTINCT FROM
    posting_id) would then abort the transaction mid-cycle.
    """
    feed = """<?xml version='1.0' encoding='UTF-8'?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry><title type="text">Role</title>
        <content type="html">&lt;a href="https://example.test/p.html"&gt;Read more&lt;/a&gt;
          &lt;a href="https://careers.example.test/apply"&gt;Apply&lt;/a&gt;</content>
        <link rel="alternate" type="text/html" href="https://example.test/p.html"/>
      </entry>
    </feed>"""
    posting = build(feed).fetch().postings[0]

    assert posting.source_url not in posting.outbound_links
    assert posting.outbound_links == ("https://careers.example.test/apply",)


def test_image_anchors_are_not_outbound_links():
    """Blogger wraps every inline image in an anchor to the full-size file, so
    on live posts the first anchor is the post's own photo. Left in, a CDN path
    would sit at the head of outbound_links and phase 4's "prefer the first
    link" rule would be reading an image instead of a posting.
    """
    feed = """<?xml version='1.0' encoding='UTF-8'?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry><title type="text">Role</title>
        <content type="html">&lt;a href="https://cdn.example.test/img/a/AVvXsEi.jpg"&gt;&lt;img/&gt;&lt;/a&gt;
          &lt;a href="https://careers.example.test/apply"&gt;Apply&lt;/a&gt;</content>
        <link rel="alternate" type="text/html" href="https://example.test/p.html"/>
      </entry>
    </feed>"""
    posting = build(feed).fetch().postings[0]

    assert posting.outbound_links == ("https://careers.example.test/apply",)


def test_tracking_parameters_are_stripped_from_the_posting_url():
    """One post reachable at two query strings would be counted as two jobs."""
    postings = build().fetch().postings
    admin = [p for p in postings if "office-administrator" in p.source_url][0]

    assert admin.source_url == "https://example.test/2026/07/office-administrator.html"
    assert "utm_source" not in admin.source_url


# ---------------------------------------------------------------------------
# failure modes
# ---------------------------------------------------------------------------
def test_malformed_xml_returns_an_errored_result_rather_than_raising():
    """A feed that changed shape must not half-parse into postings with quietly
    missing fields, and must not take down the branch it runs in — the other
    sources in the cycle are unaffected and their work must survive."""
    result = build("<feed><entry>truncated").fetch()

    assert result.postings == []
    assert result.error is not None
    assert "did not parse" in result.error
    assert not result.ok


def test_an_entry_with_no_alternate_link_is_skipped_not_invented():
    feed = """<?xml version='1.0' encoding='UTF-8'?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry><title type="text">No link here</title>
        <content type="html">&lt;p&gt;body&lt;/p&gt;</content></entry>
    </feed>"""
    result = build(feed).fetch()

    assert result.postings == []
    assert result.skipped == 1


def test_limit_is_honoured():
    assert len(build().fetch(limit=1).postings) == 1


def test_known_unchanged_entries_are_not_re_emitted():
    """The incremental path: an entry the store already has, unchanged, is not
    re-parsed into the batch. Injected as a callable — nothing under sources/
    may import the database."""
    seen = {"https://example.test/2026/07/control-systems-engineer.html"}
    adapter = build(is_known_unchanged=lambda p: p.source_url in seen)

    urls = [p.source_url for p in adapter.fetch().postings]
    assert "https://example.test/2026/07/control-systems-engineer.html" not in urls
    assert "https://example.test/2026/07/office-administrator.html" in urls


# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("https://Example.test/a/b/", "https://example.test/a/b"),
        ("http://example.test/a?utm=1#frag", "https://example.test/a"),
        ("https://example.test", "https://example.test/"),
    ],
)
def test_canonical_url_collapses_the_variants_that_are_one_page(raw, expected):
    assert canonical_url(raw) == expected
