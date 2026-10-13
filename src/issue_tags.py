"""Issue tagger: find concrete problems mentioned in a review with a keyword/regex lexicon.

Sentiment says *how* unhappy a customer is; tags say *what* went wrong. The fusion step
uses both: e.g. a "damaged_packaging" claim can be checked against the customer's photo.

Why rules and not a model: we have no labelled data for these tags, the phrases are very
regular ("arrived damaged", "wrong item", "want a refund"), and rules are fully explainable:
the app shows the exact phrase that triggered each tag.

Negation: "the box was not damaged" or "no damage at all" must NOT be tagged. A match is
rejected if a negation word appears inside the matched phrase, or in the last few words
before it within the same clause.
Some patterns are complaints *because of* a negation ("never arrived", "doesn't work",
"not returnable"). They start with "!" in the lexicon and skip the negation check.
"""

from __future__ import annotations

import re

from src.preprocess_text import basic_clean

# Each tag: a list of regex patterns (matched on lowercase text). \b = word boundary.
_DAMAGE = r"(damaged|crushed|smashed|dented|torn|ripped|broken|busted|cracked|squashed|leak(ed|ing)?|soaked|wet)"
_PACKAGE = r"(box|boxes|package|packaging|parcel|carton|packet|envelope|wrapping|seal)"

LEXICON: dict[str, list[str]] = {
    "damaged_packaging": [
        rf"\b{_PACKAGE}\b(\s+\w+){{0,4}}?\s+(was|were|is|came|arrived)?\s*{_DAMAGE}\b",
        rf"\b{_DAMAGE}\s+{_PACKAGE}\b",
        rf"\b(arrived|came|delivered|showed up|received it)\s+(\w+\s+)?{_DAMAGE}\b",
        r"\bdamaged (in|during) (shipping|transit|delivery)\b",
        r"\b(shattered|smashed) (glass|jar|bottle)s?\b",
    ],
    "wrong_item": [
        r"\b(wrong|incorrect) (item|product|size|color|colour|model|one|thing|version|flavou?r|order)\b",
        r"\bnot what i (ordered|paid for|asked for)\b",
        r"\b(sent|received|got|shipped) (me )?(a|the) (wrong|different)\b",
        r"\b(different|another) (item|product|model|brand) than (what )?(i )?(ordered|pictured|advertised)\b",
        r"\bnot the (item|product|one|model) (i )?(ordered|pictured|shown)\b",
    ],
    "missing_item": [
        # "missing" alone is too broad ("was missing home"), so it needs a thing or a place.
        r"\b(item|items|piece|pieces|part|parts|screws?|lid|cap|cable|charger|accessor(y|ies)|half|some|one|two|three)"
        r" (was|were|is|are) missing\b",
        r"\bmissing (item|items|piece|pieces|part|parts|screws?|lid|cap|cable|charger|accessor(y|ies)|instructions)\b",
        r"\bmissing (from|in) (the |my )?(box|package|order|bag|parcel|delivery)\b",
        r"!\bnever (arrived|came|received|got|showed up|delivered)\b",
        r"!\b(didn't|did not|have not|haven't|never) (receive|get|got)\b",
        r"\bempty (box|package|bag|parcel)\b",
        r"\bonly (got|received|came with) (\d+|one|two|three|half)\b",
    ],
    "defective_product": [
        r"\b(defective|faulty|malfunction(ing|ed)?|dead on arrival|doa)\b",
        r"\b(stopped|quit|ceased) working\b",
        r"!\b(doesn't|does not|didn't|did not|won't|will not|wouldn't|would not) (work|turn on|charge|power on|connect|pair|start)\b",
        r"!\b(not|isn't|is not|wasn't|was not) working\b",
        r"\bbroke (after|within|in) (a|one|two|three|\d+)\b",
        r"\bfell apart\b",
    ],
    "spoiled_food": [
        r"\b(expired|expiration date (has |had )?passed|past (the |its )?(expiration|best before|use by))\b",
        r"\b(moldy|mouldy|mold|mould|rotten|rotting|rancid|spoiled|spoilt|stale)\b",
        r"\b(gone|went) (bad|off|rancid)\b",
        r"\b(bugs|insects|weevils|worms|maggots)\b",
        r"\b(smelled|smells|tasted|tastes) (bad|off|rotten|sour|spoiled)\b",
    ],
    "late_delivery": [
        r"\b(arrived|came|delivered|shipped)\s+(\w+\s+)?(late|after|weeks later|too late)\b",
        r"\b(late|delayed|slow) (delivery|shipping|shipment|arrival)\b",
        r"\b(delivery|shipping|shipment|order) (was |is )?(late|delayed|slow)\b",
        r"\btook (over |more than |almost )?(\d+|two|three|four|several|many) (days|weeks|months)\b",
        r"\btook (forever|ages|a month|weeks)\b",
        r"\bstill waiting\b",
    ],
    "refund_request": [
        r"\b(refund|refunded|refunding|money back|reimburse(ment)?)\b",
        r"\b(return(ed|ing)?|send(ing)? (it|them) back|sent (it|them) back)\b",
        # "replacement" alone also means "a replacement for my old headset", so it needs
        # a request or delivery verb around it.
        r"\b(send|sent|get|got|need|needed|want|wanted|request|requested|asked for|ordered|receive|received)"
        r" (me |us )?(a |another )?(replacement|exchange)\b",
        r"\breplacement (was|is|also|never|came|arrived|was sent)\b",
        r"!\b(not |non-?)?(eligible|returnable)\b",
    ],
}

# Tags the fusion step treats as "this should be visible in a photo".
VISUAL_TAGS = {"damaged_packaging", "defective_product", "spoiled_food", "wrong_item"}

_NEGATORS = re.compile(r"\b(not|no|never|without|nothing|none|neither|nor|wasn't|weren't|isn't|aren't|"
                       r"didn't|doesn't|don't|hasn't|haven't|hardly|zero)\b")
_CLAUSE_BREAK = re.compile(r"[.!?;,]|\bbut\b|\bhowever\b|\bthough\b")
# (pattern, negation_is_the_claim)
_COMPILED = {tag: [(re.compile(p.lstrip("!")), p.startswith("!")) for p in pats] for tag, pats in LEXICON.items()}


def _is_negated(text: str, start: int, end: int, window_words: int = 4) -> bool:
    """True if a negation word is inside the match or in the few words before it (same clause).

    "the box was not damaged"                  -> "not" inside the match: negated
    "arrived without any damage", "no damage"  -> negation just before: negated
    "it was not cheap, but the box arrived crushed" -> "but" starts a new clause: NOT negated
    """
    before = text[:start]
    breaks = list(_CLAUSE_BREAK.finditer(before))
    if breaks:
        before = before[breaks[-1].end():]
    last_words = " ".join(before.split()[-window_words:])
    return bool(_NEGATORS.search(last_words) or _NEGATORS.search(text[start:end]))


def tag_issues(text: str) -> dict[str, str]:
    """Return {tag: phrase that triggered it} for every issue found in the review.

    >>> tag_issues("The box arrived crushed and I want a refund")
    {'damaged_packaging': 'box arrived crushed', 'refund_request': 'refund'}
    """
    low = basic_clean(text).lower().replace("’", "'")  # curly apostrophe -> straight
    found: dict[str, str] = {}
    for tag, patterns in _COMPILED.items():
        for pat, negation_is_claim in patterns:
            for m in pat.finditer(low):
                if negation_is_claim or not _is_negated(low, m.start(), m.end()):
                    found[tag] = m.group(0).strip()
                    break
            if tag in found:
                break
    return found


def tag_list(text: str) -> list[str]:
    """Just the tag names, in lexicon order."""
    return list(tag_issues(text))


def tag_rates(df, text_col: str = "text"):
    """Share of reviews in each sentiment class that get each tag (a sanity check without labels).

    We have no ground-truth tags, but a good tagger should fire far more often on negative
    reviews than on positive ones. A tag that fires often on positive reviews is suspicious.
    """
    import pandas as pd

    tags = df[text_col].map(tag_list)
    rows = {}
    for label, idx in df.groupby("label").groups.items():
        t = tags.loc[idx]
        rows[label] = {tag: t.map(lambda x: tag in x).mean() for tag in LEXICON}
        rows[label]["any tag"] = t.map(bool).mean()
    return pd.DataFrame(rows)[["negative", "neutral", "positive"]].round(3)
