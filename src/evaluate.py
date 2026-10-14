"""Shared evaluation: metrics, confusion matrices and the results/metrics.json store.

Every model (ours, the internet benchmarks, the Colab fine-tunes) is scored by the same
functions here, so the comparison table is fair. Main metric: macro-F1, the average F1 of
the three classes, so the rarest/hardest class (neutral) counts as much as the others.
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from src.config import path
from src.plots import INK, apply_style, save

LABELS = ["negative", "neutral", "positive"]


def classification_metrics(y_true, y_pred, labels: list[str] = LABELS) -> dict[str, Any]:
    """Accuracy, macro-F1, per-class precision/recall/F1 and the confusion matrix."""
    y_true, y_pred = list(y_true), list(y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "n": len(y_true),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 4),
        "per_class": {lab: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
                            "f1": round(float(f[i]), 4), "support": int(s[i])}
                      for i, lab in enumerate(labels)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }


def evaluate_text_model(name: str, test: pd.DataFrame, pred: np.ndarray, external: pd.DataFrame | None = None,
                        external_pred: np.ndarray | None = None) -> dict[str, Any]:
    """Score one text model on the internal test set, per source, and on the external set.

    The external Customer Feedback set has no neutral class, so there we only ask:
    "did the model put the review on the right side, positive or negative?"
    `external_pred` must already be a positive/negative decision (see TextModel.predict_binary).
    """
    out = {"test": classification_metrics(test["label"], pred)}
    out["test_per_source"] = {
        src: {k: v for k, v in classification_metrics(g["label"], pred[g.index]).items()
              if k in ("n", "accuracy", "macro_f1")}
        for src, g in test.groupby("source")
    }
    if external is not None and external_pred is not None:
        out["external"] = classification_metrics(external["label"], external_pred, labels=["negative", "positive"])
    return out


# ------------------------------------------------------------- metrics store --

def _metrics_file():
    return path("results", create=True) / "metrics.json"


def load_metrics() -> dict[str, Any]:
    f = _metrics_file()
    return json.loads(f.read_text()) if f.exists() else {}


def save_metrics(section: str, name: str, metrics: dict[str, Any]) -> None:
    """Merge one model's results into results/metrics.json under metrics[section][name].

    Merging (instead of overwriting the file) lets local runs and Colab runs add their
    results to the same file independently.
    """
    data = load_metrics()
    data.setdefault(section, {})[name] = metrics
    _metrics_file().write_text(json.dumps(data, indent=2))


def import_colab_metrics() -> list[str]:
    """Merge results/colab/metrics_<section>_<name>.json files (made on Colab) into metrics.json.

    Colab writes one small file per model instead of editing metrics.json, so the local file
    and the Colab files never overwrite each other. Returns the names that were imported.
    """
    imported = []
    for f in sorted(path("results", "colab", create=True).glob("metrics_*.json")):
        stem = f.stem.removeprefix("metrics_")
        # New files: metrics_<section>__<name>.json (section may contain "_", e.g. image_packages).
        # Older text files: metrics_text_<name>.json.
        section, name = stem.split("__", 1) if "__" in stem else stem.split("_", 1)
        save_metrics(section, name, json.loads(f.read_text()))
        imported.append(f"{section}/{name}")
    return imported


def comparison_table(section: str = "text") -> pd.DataFrame:
    """One row per model: the numbers for the slides and the app's Models page."""
    rows = []
    for name, m in load_metrics().get(section, {}).items():
        test = m.get("test", {})
        row = {"model": name, "type": m.get("type", ""),
               "test macro-F1": test.get("macro_f1"), "test accuracy": test.get("accuracy")}
        for lab in LABELS:
            row[f"F1 {lab}"] = test.get("per_class", {}).get(lab, {}).get("f1")
        row["external F1 (pos/neg)"] = m.get("external", {}).get("macro_f1")
        row["ms / review"] = m.get("ms_per_review")
        row["size MB"] = m.get("size_mb")
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("test macro-F1", ascending=False).set_index("model")


# ------------------------------------------------------------ explainability --

def misclassified_examples(test: pd.DataFrame, model_name: str, n: int = 8, seed: int = 42) -> pd.DataFrame:
    """The model's most *confident* mistakes: reviews where it was sure and wrong.

    Confident mistakes are the most instructive ones to discuss: they show what the model
    misunderstands (sarcasm, mixed opinions, labels that disagree with the text...).
    """
    pred = pd.read_csv(path("results", "predictions") / f"text_{model_name}_test.csv")
    df = test[["review_id", "source", "text", "rating"]].merge(pred, on=["review_id", "source"])
    wrong = df[df["label"] != df["pred"]].copy()
    wrong["confidence"] = wrong[["p_negative", "p_neutral", "p_positive"]].max(axis=1)
    return (wrong.sort_values("confidence", ascending=False).head(n)
                 [["source", "rating", "label", "pred", "confidence", "text"]].reset_index(drop=True))


# -------------------------------------------------------------------- plots --

GROUP_COLORS = {"ours": "#2a78d6", "ready-made": "#eb6834", "fine-tuned": "#1baf7a"}  # validated slots 1-3


def model_group(name: str) -> str:
    if name.startswith("bench_"):
        return "ready-made"
    if name in ("mobilenetv2",):
        return "fine-tuned"  # pre-trained on ImageNet, adapted to our photos
    if name.endswith("_finetuned"):
        return "fine-tuned"
    return "ours"


def plot_model_comparison(table: pd.DataFrame, metric: str = "test macro-F1",
                          filename: str = "text_model_comparison") -> plt.Figure:
    """Horizontal bars, best model on top, coloured by where the model comes from."""
    apply_style()
    t = table.dropna(subset=[metric]).sort_values(metric)
    groups = [model_group(m) for m in t.index]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(t) + 1.2))
    ax.barh(t.index, t[metric], color=[GROUP_COLORS[g] for g in groups], height=0.62)
    for y, v in enumerate(t[metric]):
        ax.text(v + 0.005, y, f"{v:.3f}", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, min(1.0, t[metric].max() + 0.1))
    ax.set_xlabel(metric)
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for g, c in GROUP_COLORS.items() if g in groups]
    ax.legend(handles, [g for g in GROUP_COLORS if g in groups], loc="lower right")
    ax.set_title(f"Text sentiment models: {metric}", loc="left")
    save(fig, filename)
    return fig


def plot_top_words(top: dict[str, pd.Series], filename: str = "text_tfidf_top_words") -> plt.Figure:
    """The words/phrases with the largest Logistic Regression weight for each class."""
    from src.plots import CLASS_COLORS

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, lab in zip(axes, LABELS):
        s = top[lab][::-1]
        ax.barh(s.index, s.values, color=CLASS_COLORS[lab], height=0.65)
        ax.set_title(f"Pushes towards {lab}", loc="left")
        ax.set_xlabel("LR coefficient")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelcolor=INK)
    fig.tight_layout(w_pad=3)
    save(fig, filename)
    return fig


def plot_confusion_matrix(cm: list[list[int]], labels: list[str], title: str, filename: str | None = None) -> plt.Figure:
    """Confusion matrix as a heatmap with row percentages (how each true class was predicted).

    Row percentages make classes of different sizes comparable; the raw count is shown too.
    """
    apply_style()
    cm = np.asarray(cm)
    share = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)  # one hue, light -> dark = more
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{share[i, j]:.0%}\n({cm[i, j]:,})", ha="center", va="center", fontsize=9,
                    color="white" if share[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.grid(False)
    ax.set_title(title, loc="left", fontsize=11)
    if filename:
        save(fig, filename)
    return fig


def image_comparison_table(domain: str) -> pd.DataFrame:
    """One row per image model of a domain (test set, tuned threshold): for slides and the app."""
    section = load_metrics().get(f"image_{domain}", {})
    rows = []
    for name, m in section.items():
        bad = m["labels"][1]
        t, d = m["test"], m["test_default"]
        rows.append({"model": name, "type": m.get("type", ""), "test macro-F1": t["macro_f1"], "accuracy": t["accuracy"],
                     f"recall {bad}": t["per_class"][bad]["recall"], f"precision {bad}": t["per_class"][bad]["precision"],
                     "threshold": m["threshold"], "macro-F1 at 0.5": d["macro_f1"], "size MB": m.get("size_mb")})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("test macro-F1", ascending=False).set_index("model")
