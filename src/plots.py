"""Shared figure style, so every chart in the notebooks, README and app looks the same.

Sentiment is a scale from bad to good, so the three classes use a diverging pair:
red = negative, gray = neutral (the "nothing" middle), blue = positive.
The colors were checked with a colour-blindness validator; charts also always carry
labels or a legend, so colour is never the only way to tell classes apart.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files; works on servers and Colab without a screen
import matplotlib.pyplot as plt

from src.config import path

CLASS_COLORS = {"negative": "#e34948", "neutral": "#8a8984", "positive": "#2a78d6"}
INK = "#0b0b0b"         # titles, values
INK_MUTED = "#52514e"   # axis labels, ticks
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def apply_style() -> None:
    """Thin, quiet axes so the data stands out."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlecolor": INK,
        "axes.labelcolor": INK_MUTED, "axes.labelsize": 10,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 9,
        "font.family": "sans-serif", "font.size": 10,
    })


def save(fig: plt.Figure, name: str) -> Path:
    """Save a figure to results/figures/<name>.png and return the path."""
    out = path("figures", create=True) / f"{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    return out
