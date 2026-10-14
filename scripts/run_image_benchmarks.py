"""Zero-shot CLIP baseline for both image domains (no training, runs on a laptop).

Usage:  python scripts/run_image_benchmarks.py [packages] [produce]

CLIP compares each photo with two sentences ("a photo of an intact package" vs "a photo of a
damaged package") and says which fits better. Its decision threshold is tuned on the validation
set exactly like our own models, so the comparison is fair. Results go to results/metrics.json
under image_<domain> / bench_clip_zero_shot.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config, set_seed  # noqa: E402
from src.evaluate import plot_confusion_matrix, save_metrics  # noqa: E402
from src.image_models import (CLIP_PROMPTS, clip_zero_shot, evaluate_image_model, load_split,  # noqa: E402
                              save_image_predictions, tune_threshold)

NAME = "bench_clip_zero_shot"


def main(domains: list[str]) -> None:
    set_seed()
    for domain in domains:
        Xv, yv, mv = load_split(domain, "val")
        Xt, yt, mt = load_split(domain, "test")
        p_val = clip_zero_shot(domain, Xv)
        save_image_predictions(domain, NAME, mv, yv, p_val, split="val")
        tuned = tune_threshold(yv, p_val, load_config()["images"]["target_recall_bad"])
        p_test = clip_zero_shot(domain, Xt)
        metrics = evaluate_image_model(domain, p_test, yt, mt, tuned["threshold"])
        metrics.update({"type": "ready-made (zero-shot CLIP, no training)", "threshold_tuning": tuned,
                        "prompts": CLIP_PROMPTS[domain], "model": load_config()["images"]["clip_model"]})
        save_metrics(f"image_{domain}", NAME, metrics)
        save_image_predictions(domain, NAME, mt, yt, p_test)
        plot_confusion_matrix(metrics["test"]["confusion_matrix"], metrics["test"]["labels"],
                              f"{domain} · CLIP zero-shot (threshold {tuned['threshold']:.2f})",
                              f"cm_image_{domain}_{NAME}")
        t = metrics["test"]
        print(f"[done] {domain}: accuracy {t['accuracy']:.3f} macro-F1 {t['macro_f1']:.3f} "
              f"(default 0.5: {metrics['test_default']['macro_f1']:.3f}) threshold {tuned['threshold']}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["packages", "produce"])
