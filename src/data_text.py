"""Build the text dataset: load 5 sources -> one schema -> label -> clean -> dedupe -> balance -> split.

Run everything:   python -m src.data_text
Or step by step from notebooks/01_text_data.ipynb.

Unified schema: review_id, source, category, text, rating, label
plus `text_classical` (the heavier cleaning for Naive Bayes / TF-IDF, see preprocess_text.py).
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import load_config, path
from src.preprocess_text import basic_clean, clean_for_classical, normalise_for_dedup, word_count

COLUMNS = ["review_id", "source", "category", "text", "rating", "label"]

# Titles that Amazon fills in automatically from the star rating ("Five Stars").
# They give the answer away (a model would learn "five stars" = positive), so we drop
# the title and keep only the customer's own body text.
_STAR_TITLE_RE = re.compile(r"^\s*(one|two|three|four|five|[1-5])\s+stars?\s*\.?\s*$", re.IGNORECASE)


# ----------------------------------------------------------------- labelling --

def rating_to_label(rating: float | int | None) -> str | None:
    """1-2 stars -> negative, 3 -> neutral, 4-5 -> positive. Anything else -> None.

    One rule for every source with stars, so "negative" means the same thing everywhere.
    """
    mapping = load_config()["text"]["rating_to_label"]
    try:
        r = int(float(rating))
    except (TypeError, ValueError):
        return None
    if r != float(rating):  # 3.5 stars is not a valid rating in these datasets
        return None
    return mapping.get(r)


def combine_title_body(title: str | None, body: str | None) -> str:
    """Join title and body into one text, e.g. "Does not work. Tried several configurations".

    The title often holds the strongest opinion ("Waste of money"), so it is worth keeping,
    unless it is an automatic star title (see _STAR_TITLE_RE).
    """
    title = basic_clean(title) if isinstance(title, str) else ""
    body = basic_clean(body) if isinstance(body, str) else ""
    if _STAR_TITLE_RE.match(title):
        title = ""
    if not title:
        return body
    if not body or title.lower() in body.lower():
        return body or title
    sep = " " if title[-1] in ".!?" else ". "
    return f"{title}{sep}{body}"


# ------------------------------------------------------------------- loaders --
# Each loader returns a DataFrame with exactly COLUMNS. Labels are added later by
# `add_labels`, except for Customer Feedback which has its own labels and no stars.

def _raw(name: str, filename: str) -> Path:
    return path("data_raw", "text", name, filename)


def load_amazon_lecturer() -> pd.DataFrame:
    """Lecturer's Amazon set: 4,915 reviews of one memory card. `summary` = title."""
    src = load_config()["text"]["sources"]["amazon_lecturer"]
    df = pd.read_csv(_raw("amazon_lecturer", "amazon_review.csv"))
    return pd.DataFrame({
        "review_id": "amazon_lecturer_" + df.index.astype(str),
        "source": "amazon_lecturer",
        "category": src["category"],
        "text": [combine_title_body(t, b) for t, b in zip(df["summary"], df["reviewText"])],
        "rating": df["overall"],
        "label": None,
    })


def load_flipkart() -> pd.DataFrame:
    """Flipkart: use ONLY the `Summary` column (the customer's own words).

    Why not `Review`: it is not written by customers. Flipkart picks it from the star
    rating ("terrific" = 5 stars in 99% of rows, "delightful" = 4 stars). Training on it
    would teach the model to read the star rating, not the sentiment.
    Why not `Sentiment`: it was made by the VADER tool, not by people, so it isn't a real label.
    `Rate` has 3 broken rows that contain product names; to_numeric turns them into NaN.
    """
    src = load_config()["text"]["sources"]["flipkart"]
    df = pd.read_csv(_raw("flipkart", "Dataset-SA.csv"))
    return pd.DataFrame({
        "review_id": "flipkart_" + df.index.astype(str),
        "source": "flipkart",
        "category": src["category"],
        "text": [basic_clean(s) for s in df["Summary"]],
        "rating": pd.to_numeric(df["Rate"], errors="coerce"),
        "label": None,
    })


def load_amazon2023(name: str) -> pd.DataFrame:
    """Amazon Reviews 2023 sample (random chunks, see scripts/download_data.py)."""
    src = load_config()["text"]["sources"][name]
    df = pd.read_json(_raw(name, "sample.jsonl"), lines=True)
    return pd.DataFrame({
        "review_id": name + "_" + df.index.astype(str),
        "source": name,
        "category": src["category"],
        "text": [combine_title_body(t, b) for t, b in zip(df["title"], df["text"])],
        "rating": df["rating"],
        "label": None,
    })


def load_customer_feedback() -> pd.DataFrame:
    """Lecturer's Customer Feedback set: external test only, never trained on.

    The file is broken: every row is wrapped in quotes twice, so pandas reads it as a
    single column. We parse each line as CSV, then parse the inside again.
    It has no stars, so we keep its own Positive/Negative labels (it has no Neutral).
    """
    path_ = _raw("customer_feedback", "sentiment-analysis.csv")
    rows = []
    with open(path_, encoding="utf-8") as f:
        for outer in csv.reader(f):
            if outer:
                rows.append(next(csv.reader([outer[0]], skipinitialspace=True)))
    header = [h.strip() for h in rows[0]]
    df = pd.DataFrame([r[: len(header)] for r in rows[1:] if len(r) >= len(header)], columns=header)
    return pd.DataFrame({
        "review_id": "customer_feedback_" + df.index.astype(str),
        "source": "customer_feedback",
        "category": load_config()["text"]["sources"]["customer_feedback"]["category"],
        "text": [basic_clean(t) for t in df["Text"]],
        "rating": np.nan,
        "label": df["Sentiment"].str.strip().str.lower(),
    })


def load_all_sources() -> dict[str, pd.DataFrame]:
    """Load every text source listed in config.yaml into the unified schema."""
    loaders = {
        "amazon_lecturer": load_amazon_lecturer,
        "flipkart": load_flipkart,
        "customer_feedback": load_customer_feedback,
    }
    out = {}
    for name, src in load_config()["text"]["sources"].items():
        out[name] = load_amazon2023(name) if src["kind"] == "hf_sample" else loaders[name]()
    return out


# ------------------------------------------------------------ clean + dedupe --

def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the star rule to rows that have a rating; keep existing labels otherwise."""
    df = df.copy()
    has_rating = df["rating"].notna()
    df.loc[has_rating, "label"] = df.loc[has_rating, "rating"].map(rating_to_label)
    return df


def clean(df: pd.DataFrame, log: list[dict] | None = None) -> pd.DataFrame:
    """Drop rows without a valid label, empty/very short texts, and duplicates.

    Duplicates are found on the normalised text, across ALL sources, so the same
    review can never end up in both train and test. If copies of one text carry
    different labels ("good" rated 3 and 5 stars), the text is ambiguous and we drop
    every copy instead of guessing.
    `log` collects how many rows each step removed, for the report.
    """
    min_words = load_config()["text"]["min_words"]
    log = log if log is not None else []

    def step(name: str, before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
        log.append({"step": name, "removed": len(before) - len(after), "left": len(after)})
        return after

    df = step("no valid label", df, df[df["label"].isin(load_config()["text"]["labels"])])
    df = step(f"< {min_words} words", df, df[df["text"].map(word_count) >= min_words])

    df = df.assign(_norm=df["text"].map(normalise_for_dedup))
    n_labels = df.groupby("_norm")["label"].transform("nunique")
    df = step("same text, different labels", df, df[n_labels == 1])
    df = step("duplicate text", df, df.drop_duplicates("_norm", keep="first"))
    return df.drop(columns="_norm").reset_index(drop=True)


# ------------------------------------------------------------------- balance --

def balance(df: pd.DataFrame, per_class: int, seed: int, max_class_ratio: float = 3.0) -> pd.DataFrame:
    """Downsample so every class has about `per_class` reviews, spread fairly over sources.

    Why spread over sources: if 90% of negatives came from Flipkart, the model could learn
    "Flipkart style = negative". So for each class we give every source an equal share,
    and when a source has fewer reviews than its share (e.g. the small lecturer set),
    we take all of them and hand the leftover to the other sources.

    Why `max_class_ratio`: the same shortcut can happen inside one source. The lecturer set
    has 4,400 positives but only 142 neutrals, all about one memory card. Without a cap the
    model could learn "memory card text = positive". So within a source, no class may be
    more than `max_class_ratio` times its smallest class.
    """
    smallest = df.groupby(["source", "label"]).size().groupby("source").min()
    cap = (smallest * max_class_ratio).astype(int).to_dict()
    parts = []
    for label, group in df.groupby("label"):
        available = {s: min(n, cap[s]) for s, n in group["source"].value_counts().items()}
        quota = _fair_quota(available, min(per_class, sum(available.values())))
        for source, n in quota.items():
            parts.append(group[group["source"] == source].sample(n=n, random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def _fair_quota(available: dict[str, int], total: int) -> dict[str, int]:
    """Split `total` as evenly as possible over sources, capped by what each has."""
    quota = {s: 0 for s in available}
    left = dict(available)
    remaining = total
    while remaining > 0 and left:
        share = max(remaining // len(left), 1)
        for s in sorted(left, key=left.get):  # smallest sources first
            take = min(share, left[s], remaining)
            quota[s] += take
            left[s] -= take
            remaining -= take
        left = {s: n for s, n in left.items() if n > 0}
    return quota


# --------------------------------------------------------------------- split --

def split(df: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    """70/15/15 split, stratified on label AND source together.

    Stratifying on both keeps the same class mix inside every source in every split,
    so per-source test scores are meaningful.
    """
    frac = load_config()["text"]["split"]
    strata = df["label"] + "|" + df["source"]
    train, rest = train_test_split(df, train_size=frac["train"], stratify=strata, random_state=seed)
    val_share = frac["val"] / (frac["val"] + frac["test"])
    val, test = train_test_split(rest, train_size=val_share,
                                 stratify=strata.loc[rest.index], random_state=seed)
    return {"train": train.reset_index(drop=True), "val": val.reset_index(drop=True),
            "test": test.reset_index(drop=True)}


# ----------------------------------------------------------------- reporting --

def describe_raw_sources() -> pd.DataFrame:
    """One row per raw source: size, original columns and star distribution."""
    rows = []
    for name, src in load_config()["text"]["sources"].items():
        if src["kind"] == "hf_sample":
            raw = pd.read_json(_raw(name, "sample.jsonl"), lines=True)
            stars = raw["rating"]
        elif name == "customer_feedback":
            raw = load_customer_feedback()
            stars = pd.Series(dtype=float)
        else:
            file = {"amazon_lecturer": "amazon_review.csv", "flipkart": "Dataset-SA.csv"}[name]
            raw = pd.read_csv(_raw(name, file))
            stars = pd.to_numeric(raw["overall" if name == "amazon_lecturer" else "Rate"], errors="coerce")
        dist = stars.value_counts(normalize=True).sort_index()
        rows.append({
            "source": name, "role": src["role"], "category": src["category"], "rows": len(raw),
            "columns": ", ".join(map(str, raw.columns)),
            "stars 1/2/3/4/5 (%)": " / ".join(f"{dist.get(float(s), 0):.0%}" for s in range(1, 6)) if len(dist) else "no stars",
        })
    return pd.DataFrame(rows).set_index("source")


def flipkart_title_leak(top: int = 12) -> pd.DataFrame:
    """Show why Flipkart's `Review` column is dropped: its most common values vs the star rating."""
    df = pd.read_csv(_raw("flipkart", "Dataset-SA.csv"))
    common = df["Review"].value_counts().head(top).index
    sub = df[df["Review"].isin(common)]
    return pd.crosstab(sub["Review"], sub["Rate"])[["1", "2", "3", "4", "5"]]


def counts_table(df: pd.DataFrame) -> pd.DataFrame:
    """Reviews per source x label, with totals: used for the before/after report."""
    t = pd.crosstab(df["source"], df["label"], margins=True, margins_name="total")
    labels = [lab for lab in load_config()["text"]["labels"] if lab in t.columns]
    return t[labels + ["total"]]


# ---------------------------------------------------------------------- main --

def build_text_dataset(save: bool = True, verbose: bool = True) -> dict:
    """Run the full pipeline and save CSVs to data/processed/.

    Returns a dict with the splits, the external test set and the count tables,
    so the notebook can show every intermediate result.
    """
    cfg = load_config()
    seed = cfg["seed"]
    sources = load_all_sources()
    roles = {name: src["role"] for name, src in cfg["text"]["sources"].items()}

    raw = pd.concat([add_labels(d) for d in sources.values()], ignore_index=True)
    raw_counts = pd.crosstab(raw["source"], raw["label"].fillna("no label"), margins=True,
                             margins_name="total")

    # The external set is cleaned on its own: its duplicates must not remove training rows.
    internal = raw[raw["source"].map(roles) == "train"]
    external = raw[raw["source"].map(roles) == "external_test"]
    log_internal: list[dict] = []
    internal = clean(internal, log_internal)
    external = clean(external)
    clean_counts = counts_table(internal)

    balanced = balance(internal, cfg["text"]["per_class_target"], seed, cfg["text"]["max_class_ratio"])
    splits = split(balanced, seed)

    # The heavier cleaning for classical models is stored next to the text, so every
    # notebook uses exactly the same preprocessed input.
    for df in [*splits.values(), external]:
        df["text_classical"] = df["text"].map(clean_for_classical)

    result = {
        **splits,
        "external_test": external.reset_index(drop=True),
        "raw_counts": raw_counts,
        "clean_log": pd.DataFrame(log_internal),
        "clean_counts": clean_counts,
        "balanced_counts": counts_table(balanced),
        "split_counts": pd.DataFrame({k: v["label"].value_counts() for k, v in splits.items()}),
    }

    if save:
        out = path("data_processed", create=True)
        for name in ["train", "val", "test", "external_test"]:
            result[name].to_csv(out / f"text_{name}.csv", index=False)
        reports = path("results", create=True)
        for name in ["raw_counts", "clean_counts", "balanced_counts", "split_counts", "clean_log"]:
            result[name].to_csv(reports / f"text_data_{name}.csv")
    if verbose:
        for name in ["raw_counts", "clean_log", "clean_counts", "balanced_counts", "split_counts"]:
            print(f"\n--- {name} ---\n{result[name]}")
    return result


if __name__ == "__main__":
    build_text_dataset()
