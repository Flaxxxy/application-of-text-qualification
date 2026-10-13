"""Exploratory charts for the text dataset (used in notebooks/01_text_data.ipynb)."""

from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_config
from src.plots import CLASS_COLORS, INK, INK_MUTED, SURFACE, apply_style, save

LABELS = ["negative", "neutral", "positive"]


def _class_mix(ax: plt.Axes, counts: pd.DataFrame, title: str) -> None:
    """100% stacked bars: the share of each class inside each source."""
    counts = counts.drop(index="total", errors="ignore")[LABELS]
    shares = counts.div(counts.sum(axis=1), axis=0)
    left = np.zeros(len(shares))
    for label in LABELS:
        ax.barh(shares.index, shares[label], left=left, color=CLASS_COLORS[label],
                edgecolor=SURFACE, linewidth=2, height=0.6, label=label)
        for y, (x0, w) in enumerate(zip(left, shares[label])):
            if w > 0.08:  # only label segments wide enough to hold text
                ax.text(x0 + w / 2, y, f"{w:.0%}", ha="center", va="center", color="white", fontsize=8)
        left += shares[label].to_numpy()
    for y, n in enumerate(counts.sum(axis=1)):
        ax.text(1.02, y, f"n={n:,}", va="center", color=INK_MUTED, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1], ["0%", "50%", "100%"])
    ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left")


def plot_class_balance(clean_counts: pd.DataFrame, balanced_counts: pd.DataFrame) -> plt.Figure:
    """Class mix per source before and after balancing."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
    _class_mix(axes[0], clean_counts, "Before balancing (after cleaning)")
    _class_mix(axes[1], balanced_counts, "After balancing (used for training)")
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(w_pad=6)
    save(fig, "text_class_balance")
    return fig


def plot_length_distribution(df: pd.DataFrame) -> plt.Figure:
    """Review length in words per class (cut at the 99th percentile so the tail doesn't hide the shape)."""
    apply_style()
    words = df["text"].str.split().str.len()
    cap = int(words.quantile(0.99))
    bins = np.linspace(0, cap, 50)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for label in LABELS:
        w = words[df["label"] == label].clip(upper=cap)
        ax.hist(w, bins=bins, histtype="step", linewidth=2, color=CLASS_COLORS[label],
                label=f"{label} (median {int(w.median())} words)")
    ax.set_xlabel(f"words per review (longer than {cap} shown at {cap})")
    ax.set_ylabel("reviews")
    ax.set_title("Review length by class", loc="left")
    ax.legend()
    save(fig, "text_length_distribution")
    return fig


def distinctive_words(df: pd.DataFrame, text_col: str = "text_classical", top: int = 15,
                      min_count: int = 30) -> dict[str, pd.Series]:
    """Words most typical of each class, by smoothed log-odds against the other classes.

    Plain "most frequent words" are almost the same for every class ("product", "use",
    "good"), which tells us nothing. Log-odds asks instead: how much more often does this
    word appear in negative reviews than in the rest? That surfaces words like "return"
    or "broke" for negative and "love" or "perfect" for positive.
    """
    # Each word is counted once per review (document frequency): otherwise one long review
    # that repeats "SP3" 72 times would push "sp" to the top of the positive list.
    counts = {lab: Counter(w for t in df.loc[df["label"] == lab, text_col].fillna("") for w in set(t.split()))
              for lab in LABELS}
    out = {}
    for lab in LABELS:
        mine = counts[lab]
        rest = sum((counts[o] for o in LABELS if o != lab), Counter())
        n_mine, n_rest = sum(mine.values()), sum(rest.values())
        scores = {w: np.log((c + 1) / (n_mine + 1)) - np.log((rest[w] + 1) / (n_rest + 1))
                  for w, c in mine.items() if c >= min_count}
        out[lab] = pd.Series(scores).sort_values(ascending=False).head(top)
    return out


def plot_distinctive_words(df: pd.DataFrame, top: int = 15) -> plt.Figure:
    """Three panels: the most typical words of negative, neutral and positive reviews."""
    apply_style()
    words = distinctive_words(df, top=top)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, lab in zip(axes, LABELS):
        s = words[lab][::-1]
        ax.barh(s.index, s.values, color=CLASS_COLORS[lab], height=0.65)
        ax.set_title(f"Most typical of {lab}", loc="left")
        ax.set_xlabel("log-odds vs other classes")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelcolor=INK)
    fig.tight_layout(w_pad=3)
    save(fig, "text_distinctive_words")
    return fig


def sample_reviews(df: pd.DataFrame, n: int = 2) -> pd.DataFrame:
    """A few example reviews per source and class, to sanity-check the labels by eye."""
    seed = load_config()["seed"]
    shuffled = df.sample(frac=1, random_state=seed)
    return (shuffled.groupby(["source", "label"]).head(n)
                    .sort_values(["source", "label"])[["source", "label", "rating", "text"]])
