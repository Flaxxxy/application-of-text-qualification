"""Tests for the fusion rules (src/fusion.py). They use fixed inputs, no models."""

import pytest

from src.fusion import fuse

SETTINGS = {"weights": {"text": 0.5, "image": 0.35, "tag": 0.15}, "thresholds": {"review": 0.4, "urgent": 0.7}}
NEG = {"negative": 0.9, "neutral": 0.08, "positive": 0.02}
NEU = {"negative": 0.2, "neutral": 0.7, "positive": 0.1}
POS = {"negative": 0.02, "neutral": 0.08, "positive": 0.9}


def img(p_bad, domain="packages", threshold=0.4):
    return {"domain": domain, "p_bad": p_bad, "is_bad": p_bad >= threshold, "threshold": threshold}


def codes(d):
    return [r["code"] for r in d.reasons]


def test_negative_text_and_bad_photo_is_urgent():
    d = fuse(NEG, [], "electronics", img(0.9), SETTINGS)
    assert d.status == "URGENT" and codes(d)[0] == "BOTH_PROBLEM"


def test_positive_text_and_bad_photo_is_review():
    d = fuse(POS, [], "electronics", img(0.9), SETTINGS)
    assert d.status == "REVIEW" and codes(d)[0] == "PHOTO_CONTRADICTS_REVIEW"


def test_damage_claim_with_clean_photo_is_review_possible_false_claim():
    d = fuse(NEG, ["damaged_packaging", "refund_request"], "electronics", img(0.05), SETTINGS)
    assert d.status == "REVIEW" and codes(d)[0] == "CLAIM_NOT_VISIBLE"


def test_spoiled_food_claim_is_checked_by_the_produce_model():
    d = fuse(NEG, ["spoiled_food"], "grocery", img(0.05, "produce"), SETTINGS)
    assert codes(d)[0] == "CLAIM_NOT_VISIBLE" and "fresh" in d.reasons[0]["text"]


def test_claim_the_photo_cannot_show_is_not_a_false_claim():
    # "stopped working" can't be seen on a package photo, so it's a non-visual issue, not a false claim.
    d = fuse(NEG, ["defective_product"], "electronics", img(0.05), SETTINGS)
    assert d.status == "REVIEW" and codes(d)[0] == "ISSUE_NOT_VISUAL"


def test_negative_text_clean_photo_no_tag_is_review():
    d = fuse(NEG, [], "electronics", img(0.05), SETTINGS)
    assert d.status == "REVIEW" and codes(d)[0] == "ISSUE_NOT_VISUAL"


def test_neutral_text_and_bad_photo_is_never_ok():
    d = fuse(NEU, [], "grocery", img(0.5, "produce", 0.34), SETTINGS)
    assert d.status in ("REVIEW", "URGENT") and "PHOTO_PROBLEM" in codes(d)


def test_happy_order_is_ok():
    d = fuse(POS, [], "electronics", img(0.05), SETTINGS)
    assert d.status == "OK"


def test_no_photo_is_text_only_and_says_so():
    d = fuse(NEG, ["refund_request"], "other", None, SETTINGS)
    assert not d.used_photo and "NO_PHOTO" in codes(d)
    assert d.status == "URGENT"  # (0.5*0.9 + 0.15*1) / 0.65 = 0.92


@pytest.mark.parametrize("p_neg, expected", [(0.1, "OK"), (0.6, "REVIEW"), (0.95, "URGENT")])
def test_text_only_risk_thresholds(p_neg, expected):
    probs = {"negative": p_neg, "neutral": 1 - p_neg, "positive": 0.0}
    assert fuse(probs, [], "other", None, SETTINGS).status == expected
