"""Text cleaning: one light cleaner for every model, plus a heavier one for classical models.

Two paths, because the two model families "read" text differently:

1. Classical models (Naive Bayes, TF-IDF + Logistic Regression) only see word counts.
   Words like "the" or "is" add noise to the counts, and "box" / "boxes" should count
   as one word. So we lowercase, drop punctuation and stopwords, and lemmatise.
   BUT we keep negations: removing "not" turns "not good" into "good" and flips the
   sentiment. NLTK's stopword list contains "not", "no", "don't", ... so we take them out
   of the list on purpose.

2. Transformer models (MiniLM, DistilBERT, RoBERTa) read whole sentences and were
   pre-trained on normal text with stopwords and punctuation. Removing words would take
   away context they rely on ("not", "but", "!"), so they get the light cleaning only.
"""

from __future__ import annotations

import html
import re
from functools import lru_cache

# ------------------------------------------------------------ light cleaning --

_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
_SPACE_RE = re.compile(r"\s+")


def basic_clean(text: str) -> str:
    """Light cleaning used for EVERY model and for the stored dataset.

    Amazon reviews contain HTML like "<br />" and "&amp;", and some contain links.
    These carry no sentiment, so we remove them and squeeze repeated spaces.
    """
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)          # "&amp;" -> "&", "&#39;" -> "'"
    text = _TAG_RE.sub(" ", text)        # "<br />" -> " "
    text = _URL_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def normalise_for_dedup(text: str) -> str:
    """Very aggressive form used ONLY to find duplicates.

    "Great product!!" and "great product" are the same review for our purposes,
    so we compare lowercase letters and digits only.
    """
    return _SPACE_RE.sub(" ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


def word_count(text: str) -> int:
    return len(text.split())


# -------------------------------------------------- classical-model cleaning --

# Words that flip or weaken sentiment. They must survive stopword removal.
NEGATIONS = {
    "not", "no", "nor", "never", "n't", "cannot", "without", "against",
    "don", "don't", "doesn", "doesn't", "didn", "didn't", "isn", "isn't",
    "wasn", "wasn't", "weren", "weren't", "aren", "aren't", "won", "won't",
    "wouldn", "wouldn't", "shouldn", "shouldn't", "couldn", "couldn't",
    "hasn", "hasn't", "haven", "haven't", "hadn", "hadn't", "mustn", "mustn't",
    "needn", "needn't", "ain", "but", "very", "too", "only",
}
# ("but", "very", "too", "only" are not negations, but they change how strong or
#  mixed an opinion is: "good but too small". They are cheap to keep.)


def ensure_nltk() -> None:
    """Download the small NLTK data files once (tokeniser, stopwords, WordNet)."""
    import nltk

    for pkg, probe in [("punkt_tab", "tokenizers/punkt_tab"), ("stopwords", "corpora/stopwords"),
                       ("wordnet", "corpora/wordnet"), ("omw-1.4", "corpora/omw-1.4")]:
        try:
            nltk.data.find(probe)
        except LookupError:
            nltk.download(pkg, quiet=True)


@lru_cache(maxsize=1)
def _nltk_tools():
    """Build the stopword set and lemmatiser once; creating them per review is slow."""
    ensure_nltk()
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    stop = set(stopwords.words("english")) - NEGATIONS
    return stop, WordNetLemmatizer()


_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


def clean_for_classical(text: str) -> str:
    """Lowercase, remove punctuation/URLs, drop stopwords (keeping negations), lemmatise.

    Example: "The boxes were NOT damaged!!" -> "box not damaged"
    """
    stop, lemmatizer = _nltk_tools()
    text = basic_clean(text).lower()
    # Map all "n't" endings to a separate "not" token, so "isn't", "wasn't" and
    # "not" become the same feature instead of three rare ones.
    text = re.sub(r"n't\b", " not", text)
    tokens = _TOKEN_RE.findall(text)  # letters only: drops punctuation and numbers
    out = []
    for tok in tokens:
        if tok in stop or (len(tok) < 2 and tok not in NEGATIONS):
            continue  # stopword, or a single letter left over from punctuation
        out.append(lemmatizer.lemmatize(tok))
    return " ".join(out)


def clean_for_transformer(text: str) -> str:
    """Transformers get the light cleaning only (see module docstring for why)."""
    return basic_clean(text)
