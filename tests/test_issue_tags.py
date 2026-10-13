"""Tests for the issue tagger: each tag fires on typical phrases and respects negation."""

import pytest

from src.issue_tags import tag_issues, tag_list


@pytest.mark.parametrize("text, tag", [
    ("The box arrived crushed and the corner was torn", "damaged_packaging"),
    ("Package was damaged when it got here", "damaged_packaging"),
    ("It arrived broken, glass everywhere", "damaged_packaging"),
    ("The jar leaked all over the other items, damaged in shipping", "damaged_packaging"),
    ("They sent me the wrong item", "wrong_item"),
    ("Wrong size, this is not what I ordered", "wrong_item"),
    ("Two of the pieces were missing", "missing_item"),
    ("My order never arrived", "missing_item"),
    ("I received an empty box", "missing_item"),
    ("Stopped working after a week", "defective_product"),
    ("The charger doesn't work at all", "defective_product"),
    ("Defective unit, dead on arrival", "defective_product"),
    ("The bread was moldy and the chips were stale", "spoiled_food"),
    ("Expired two months ago", "spoiled_food"),
    ("Found bugs in the bag of rice", "spoiled_food"),
    ("Delivery was late by a week", "late_delivery"),
    ("Took 3 weeks to arrive", "late_delivery"),
    ("I want a refund", "refund_request"),
    ("I returned it the same day", "refund_request"),
    ("Item is not returnable, so I'm stuck with it", "refund_request"),
])
def test_tag_fires(text, tag):
    assert tag in tag_list(text), (text, tag_issues(text))


@pytest.mark.parametrize("text, tag", [
    ("The box was not damaged at all", "damaged_packaging"),
    ("No damage to the package, well wrapped", "damaged_packaging"),
    ("Arrived without any damage", "damaged_packaging"),
    ("Nothing was missing", "missing_item"),
    ("It never stopped working in 5 years", "defective_product"),
    ("Not expired, fresh dates", "spoiled_food"),
    ("Delivery was not late", "late_delivery"),
])
def test_negation_blocks_tag(text, tag):
    assert tag not in tag_list(text), (text, tag_issues(text))


@pytest.mark.parametrize("text, tag", [
    # False hits found on positive test reviews; these senses are not complaints.
    ("These are the perfect replacement for my wired headset", "refund_request"),
    ("My husband was missing home at Christmas, so I bought these", "missing_item"),
])
def test_other_word_senses_are_not_tagged(text, tag):
    assert tag not in tag_list(text), (text, tag_issues(text))


@pytest.mark.parametrize("text, tag", [
    ("Two screws were missing from the kit", "missing_item"),
    ("I asked for a replacement and never heard back", "refund_request"),
])
def test_tightened_patterns_still_fire(text, tag):
    assert tag in tag_list(text), (text, tag_issues(text))


def test_negation_does_not_cross_clauses():
    # "not" belongs to the first clause; the second clause is a real damage complaint.
    assert "damaged_packaging" in tag_list("It was not cheap, but the box arrived crushed")


def test_happy_review_has_no_tags():
    assert tag_issues("Great coffee, fast shipping, will buy again!") == {}


def test_tagger_returns_the_trigger_phrase():
    tags = tag_issues("The box arrived crushed and I want a refund")
    assert tags["refund_request"] == "refund"
    assert "crushed" in tags["damaged_packaging"]
