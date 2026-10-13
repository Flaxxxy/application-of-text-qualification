"""Train and score the local text models; save models, metrics and test predictions.

Usage (from the repo root):
    python scripts/train_text_models.py                      # all local models
    python scripts/train_text_models.py textblob tfidf_logreg
    python scripts/train_text_models.py bench                # the 3 ready-made Hugging Face models
    python scripts/train_text_models.py bench_nlptown_stars  # just one of them

Outputs:
    models/train/text_<name>.joblib          trained models (git-ignored)
    results/metrics.json                     metrics, merged per model
    results/predictions/text_<name>_test.csv test predictions (for error analysis + fusion)
    results/figures/cm_text_<name>.png       confusion matrices
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config, path, set_seed  # noqa: E402
from src.evaluate import plot_confusion_matrix, save_metrics  # noqa: E402
from src.text_models import (HFBenchmarkModel, MiniLMLogRegModel, NaiveBayesModel, TextBlobModel,  # noqa: E402
                             TfidfLogRegModel, run_and_score, save_predictions)

LOCAL = {"textblob": TextBlobModel, "naive_bayes": NaiveBayesModel,
         "tfidf_logreg": TfidfLogRegModel, "minilm_logreg": MiniLMLogRegModel}


def load_splits() -> dict[str, pd.DataFrame]:
    d = path("data_processed")
    return {s: pd.read_csv(d / f"text_{s}.csv") for s in ["train", "val", "test", "external_test"]}


def finish(model, metrics, proba, test, pred=None) -> None:
    save_metrics("text", model.name, metrics)
    save_predictions(model.name, test, proba, pred)
    plot_confusion_matrix(metrics["test"]["confusion_matrix"], metrics["test"]["labels"],
                          f"{model.name}  (macro-F1 {metrics['test']['macro_f1']:.3f})", f"cm_text_{model.name}")
    print(f"[done] {model.name}: test macro-F1 {metrics['test']['macro_f1']:.4f}, "
          f"accuracy {metrics['test']['accuracy']:.4f}, external F1 {metrics['external']['macro_f1']:.4f}, "
          f"{metrics['ms_per_review']} ms/review, chosen {metrics['chosen_on_val']}")


def main(names: list[str]) -> None:
    set_seed()
    data = load_splits()
    train, val, test, ext = data["train"], data["val"], data["test"], data["external_test"]
    models_dir = path("models_train", create=True)

    for name in names:
        if name.startswith("bench"):
            # "bench" = all ready-made models; "bench_<key>" = just one of them.
            keys = list(load_config()["text"]["benchmarks"]) if name == "bench" else [name.removeprefix("bench_")]
            for key in keys:
                model = HFBenchmarkModel(key)
                print(f"\n[run] {model.name} ({model.model_name}) on {len(test):,} test reviews, CPU ...")
                metrics, proba = run_and_score(model, test, ext)
                finish(model, metrics, proba, test)
            continue

        model = LOCAL[name]()
        print(f"\n[train] {name}")
        t0 = time.perf_counter()
        model.fit(train, val)
        train_s = time.perf_counter() - t0
        if hasattr(model, "save"):
            model.save(models_dir / f"text_{name}.joblib")
        if name == "textblob":
            import joblib

            joblib.dump({"low": model.low, "high": model.high}, models_dir / "text_textblob.joblib")
        metrics, proba = run_and_score(model, test, ext)
        metrics["train_seconds"] = round(train_s, 1)
        finish(model, metrics, proba, test, model.predict(test) if name == "textblob" else None)


if __name__ == "__main__":
    main(sys.argv[1:] or list(LOCAL))
