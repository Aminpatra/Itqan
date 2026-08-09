"""The verifier that stops a posting being described by a model rather than read.

Written because of a measurement, not a hunch. First live cycle extracting these
fields: **19 of 19 postings came back `work_arrangement = 'onsite'` and 0 of the
19 contained any arrangement phrase at all**, in Arabic or English. The
extraction prompt forbids that inference in as many words. It happened anyway.

Bilingual throughout, and that is not decoration: this corpus is Omani, and the
last time a filter in this repo was written in English and assumed the Arabic
followed, it scored an identical posting 0.560 in Arabic and 0.700 in English.
"""

from __future__ import annotations

import pytest

from agents.agent_b_job_ingest.stated_facts import verify_stated_facts


class Job:
    """The extraction's fields, minus the schema — the verifier reads attributes."""

    def __init__(self, **kw):
        self.title = kw.pop("title", "")
        for field in ("work_arrangement", "employment_type", "salary_min",
                      "salary_max", "salary_currency", "salary_period"):
            setattr(self, field, kw.pop(field, None))
        assert not kw, kw


# ---------------------------------------------------------------------------
# work arrangement — the field the measurement was about
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "Data Analyst. Muscat, Oman. Head office: Al Khuwair.",   # THE live case
    "Data Analyst. Our offices are in Muscat.",
    "محلل بيانات. مسقط، عمان.",
])
def test_a_location_is_not_a_statement_about_where_you_work(text):
    """An address says the employer exists somewhere. It says nothing about
    whether the ROLE is performed there, and 19 of 19 live postings were labelled
    'onsite' on exactly this non-evidence."""
    assert verify_stated_facts(Job(work_arrangement="onsite"), text).work_arrangement is None


@pytest.mark.parametrize("text,want", [
    ("This role is fully remote.", "remote"),
    ("Work from home, flexible hours.", "remote"),
    ("هذه الوظيفة عن بعد.", "remote"),
    ("Hybrid role: 3 days in the office.", "hybrid"),
    ("This is an on-site position in Sohar.", "onsite"),
    ("العمل حضوري في مقر الشركة.", "onsite"),
])
def test_a_stated_arrangement_survives(text, want):
    assert verify_stated_facts(Job(work_arrangement=want), text).work_arrangement == want


def test_a_page_naming_two_arrangements_settles_nothing():
    """"Remote or hybrid may be considered" is a page declining to answer. Taking
    the model's pick would be the same guess with better manners."""
    got = verify_stated_facts(Job(work_arrangement="remote"),
                              "Remote or hybrid arrangements considered.")
    assert got.work_arrangement is None


def test_the_verifier_does_not_correct_the_model_it_only_silences_it():
    """A page saying 'remote' while the model said 'onsite' is a disagreement,
    and a disagreement is not evidence for either reading. Being wrong here is
    worse than being silent — the whole point of the field is that someone acts
    on it."""
    got = verify_stated_facts(Job(work_arrangement="onsite"), "Fully remote role.")
    assert got.work_arrangement is None      # not 'remote'
    assert "work_arrangement" in got.dropped


# ---------------------------------------------------------------------------
# employment type
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text,value,kept", [
    ("Full-time permanent position.", "full_time", True),
    ("وظيفة بدوام كامل", "full_time", True),          # the live Arabic phrasing
    ("Summer internship programme.", "internship", True),
    ("برنامج تدريب صيفي", "internship", True),
    ("2-year contract role.", "contract", True),
    # Nothing said. The commonest case by far, and the one that must stay silent.
    ("Accountant needed in Muscat. Send your CV.", "full_time", False),
    ("محاسب مطلوب في مسقط. أرسل سيرتك الذاتية.", "full_time", False),
])
def test_employment_type_must_be_written_down_somewhere(text, value, kept):
    got = verify_stated_facts(Job(employment_type=value), text)
    assert (got.employment_type == value) is kept


# ---------------------------------------------------------------------------
# pay — the most damaging field to invent
# ---------------------------------------------------------------------------
def test_every_figure_must_appear_in_the_text():
    got = verify_stated_facts(
        Job(salary_min=800, salary_max=1200, salary_currency="OMR",
            salary_period="month"),
        "Salary: 800 - 1,200 OMR per month.")
    assert (got.salary_min, got.salary_max) == (800, 1200)
    assert (got.salary_currency, got.salary_period) == ("OMR", "month")


@pytest.mark.parametrize("text", [
    "Competitive salary and benefits.",
    "Attractive package, negotiable.",
    "راتب مجزي",
])
def test_competitive_is_not_a_number(text):
    """The exact phrase the extraction prompt already warns about. A model asked
    for a range will supply a plausible market rate, and a plausible market rate
    is a fabrication with good manners."""
    got = verify_stated_facts(Job(salary_min=700, salary_max=900,
                                  salary_currency="OMR"), text)
    assert got.salary_min is None and got.salary_max is None
    assert got.salary_currency is None       # a currency for no amount says nothing


def test_arabic_indic_digits_are_the_same_number():
    """٨٠٠ and 800 are one figure written two ways. Without folding, a genuine
    Omani posting's salary would be discarded as fabricated — the failure mode
    that penalised Arabic postings 0.560 vs 0.700 the last time."""
    got = verify_stated_facts(Job(salary_min=800, salary_period="month"),
                              "الراتب ٨٠٠ ريال شهريا")
    assert got.salary_min == 800
    assert got.salary_period == "month"


def test_a_thousands_separator_is_not_a_different_number():
    got = verify_stated_facts(Job(salary_max=1200), "Up to 1,200 OMR.")
    assert got.salary_max == 1200


def test_a_stated_zero_is_kept():
    """An unpaid internship saying so is an answer, and `0 or None` would erase
    it. Third time this project has had to defend a real zero from falsiness."""
    got = verify_stated_facts(Job(salary_min=0, salary_max=0),
                              "Unpaid internship: 0 OMR.")
    assert got.salary_min == 0 and got.salary_max == 0


def test_silence_is_reported_as_silence_not_as_a_failure():
    """A posting stating nothing must produce no drops — otherwise the counter
    that tells an operator the verifier is working would just count postings."""
    got = verify_stated_facts(Job(), "Accountant needed. Send your CV.")
    assert got.dropped == ()
    assert got.work_arrangement is None and got.salary_min is None
