"""Image models for the two photo checks: packages (damaged vs intact) and produce (rotten vs fresh).

Two Keras models per domain, as in the course:
    small_cnn     a small CNN trained from scratch (Conv2D, MaxPooling, Dropout, Flatten, Dense softmax)
    mobilenetv2   transfer learning: MobileNetV2 pre-trained on ImageNet, first with its base frozen,
                  then with the top layers fine-tuned                                <- used on the website
plus a ready-made baseline without training: zero-shot CLIP (clip_zero_shot).

Each trainable model is built as two parts:
    core      = preprocessing + network + softmax head. Takes raw RGB pixels 0-255. This is what we
                save and export to ONNX, so the website only has to resize the photo.
    augmenter = random flips/rotations/zoom/brightness, used only while training.
Keeping augmentation out of `core` matters for the ONNX export (src/export_onnx.py): random layers are
training-only and some converters can't handle them.

Outputs are 2 probabilities [P(good), P(bad)]; "bad" = damaged / rotten.
Heavy training runs on Colab (notebooks/colab_05_*, colab_06_*).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import pandas as pd
from PIL import Image

from src.config import load_config, path
from src.data_images import CLASSES


# ---------------------------------------------------------------------- data --

def class_names(domain: str) -> list[str]:
    """[good, bad] for a domain, e.g. ["intact", "damaged"]. Index 1 is always the bad class."""
    bad, good = CLASSES[domain]
    return [good, bad]


def load_split(domain: str, split: str, limit: int | None = None) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Images of one split as a uint8 array (n, 224, 224, 3), labels (1 = bad), and their manifest rows.

    The datasets are small (~1.5k training images), so loading everything into memory is
    simpler and faster than a streaming loader, and it works the same on a Mac and on Colab.
    """
    root = path("data_processed", "images")
    m = pd.read_csv(root / f"{domain}_manifest.csv")
    m = m[m["split"] == split].sort_values("path").reset_index(drop=True)
    if limit:
        m = m.groupby("label", group_keys=False).head(limit // 2).reset_index(drop=True)
    X = np.stack([np.asarray(Image.open(root / p).convert("RGB")) for p in m["path"]])
    y = (m["label"] == class_names(domain)[1]).astype(int).to_numpy()
    return X, y, m


# -------------------------------------------------------------------- models --

def build_augmenter(seed: int):
    """Random changes applied to training images only (flip, rotation, zoom, brightness, contrast).

    Same idea as the course's ImageDataGenerator (which Keras 3 no longer has): the model sees a
    slightly different version of each photo every epoch, so it learns the damage/rot itself
    instead of memorising exact pictures. Customer photos are taken at any angle, hence rotation.
    """
    import keras
    from keras import layers

    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical", seed=seed),
        layers.RandomRotation(0.15, seed=seed),
        layers.RandomZoom(0.15, seed=seed),
        layers.RandomBrightness(0.15, value_range=(0, 255), seed=seed),
        layers.RandomContrast(0.15, seed=seed),
    ], name="augmenter")


def build_small_cnn(size: int) -> Any:
    """Baseline CNN trained from scratch (course architecture): 4 conv blocks + Dense softmax."""
    import keras
    from keras import layers

    inputs = keras.Input((size, size, 3), name="image")          # raw pixels 0-255
    x = layers.Rescaling(1 / 255)(inputs)                         # course: rescale=1./255
    for filters in (32, 64, 128, 128):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(2, activation="softmax", name="probs")(x)
    return keras.Model(inputs, outputs, name="small_cnn")


def build_mobilenetv2(size: int) -> tuple[Any, Any]:
    """MobileNetV2 (ImageNet weights) + a small softmax head. Returns (core model, backbone).

    Why MobileNetV2: it learned general visual features (edges, textures, shapes) from 1.4M
    ImageNet photos, so it needs few examples of our classes, and it is small (~9 MB) and fast
    enough for the website.
    """
    import keras
    from keras import layers
    from keras.applications import MobileNetV2

    backbone = MobileNetV2(input_shape=(size, size, 3), include_top=False, weights="imagenet", pooling="avg")
    backbone.trainable = False
    inputs = keras.Input((size, size, 3), name="image")          # raw pixels 0-255
    x = layers.Rescaling(1 / 127.5, offset=-1)(inputs)            # MobileNetV2 expects -1..1
    x = backbone(x, training=False)  # training=False keeps BatchNorm statistics fixed, also when fine-tuning
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(2, activation="softmax", name="probs")(x)
    return keras.Model(inputs, outputs, name="mobilenetv2"), backbone


def _training_model(core, seed: int):
    import keras

    inputs = keras.Input(core.input_shape[1:], name="image")
    return keras.Model(inputs, core(build_augmenter(seed)(inputs)), name=f"{core.name}_train")


def _class_weights(y: np.ndarray) -> dict[int, float]:
    """Give the smaller class more weight so the model can't win by always guessing the big one."""
    counts = np.bincount(y, minlength=2)
    return {i: len(y) / (2 * c) for i, c in enumerate(counts)}


def _fit(model, X, y, Xv, yv, epochs: int, lr: float, patience: int, batch: int):
    import keras

    model.compile(optimizer=keras.optimizers.Adam(lr), loss="categorical_crossentropy", metrics=["accuracy"])
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    return model.fit(X, keras.utils.to_categorical(y, 2), validation_data=(Xv, keras.utils.to_categorical(yv, 2)),
                     epochs=epochs, batch_size=batch, class_weight=_class_weights(y), callbacks=[stop], verbose=2)


def train_model(kind: str, X, y, Xv, yv, quick: bool = False) -> tuple[Any, dict[str, list[float]]]:
    """Train `small_cnn` or `mobilenetv2`; return (core model, training history).

    Early stopping watches the validation loss and restores the best epoch, so we don't pick
    an over-trained model. MobileNetV2 trains in two stages:
      1. base frozen: only the new head learns (fast, safe);
      2. fine-tuning: the top layers of the base are unfrozen with a 100x smaller learning rate,
         so they adapt to boxes/fruit without forgetting what ImageNet taught them.
    `quick=True` runs 1 epoch per stage (smoke test only).
    """
    import keras

    cfg = load_config()["images"]["train"]
    seed, size = load_config()["seed"], load_config()["images"]["size"]
    keras.utils.set_random_seed(seed)
    ep = (lambda n: 1) if quick else (lambda n: n)
    history: dict[str, list[float]] = {}

    def add(h, stage):
        for k, v in h.history.items():
            history.setdefault(k, []).extend(v)
        history.setdefault("stage", []).extend([stage] * len(h.history["loss"]))

    if kind == "small_cnn":
        core = build_small_cnn(size)
        add(_fit(_training_model(core, seed), X, y, Xv, yv, ep(cfg["epochs_cnn"]), cfg["lr_cnn"],
                 cfg["patience"], cfg["batch_size"]), "scratch")
    elif kind == "mobilenetv2":
        core, backbone = build_mobilenetv2(size)
        trainer = _training_model(core, seed)
        h = _fit(trainer, X, y, Xv, yv, ep(cfg["epochs_head"]), cfg["lr_head"], cfg["patience"], cfg["batch_size"])
        add(h, "head")
        head_best, head_weights = min(h.history["val_loss"]), core.get_weights()
        backbone.trainable = True
        for layer in backbone.layers[:-cfg["finetune_layers"]]:
            layer.trainable = False
        h = _fit(trainer, X, y, Xv, yv, ep(cfg["epochs_finetune"]), cfg["lr_finetune"], cfg["patience"],
                 cfg["batch_size"])
        add(h, "fine-tune")
        # Fine-tuning helps when the photos differ a lot from ImageNet, but on small mixed datasets
        # it can overfit (we saw this for packages). Keep it only if it improved validation loss.
        if min(h.history["val_loss"]) >= head_best:
            core.set_weights(head_weights)
            history["kept_stage"] = ["head (fine-tuning did not improve val loss)"]
        else:
            history["kept_stage"] = ["fine-tune"]
    else:
        raise ValueError(kind)
    return core, history


def predict_bad(model, X: np.ndarray, batch: int = 64) -> np.ndarray:
    """P(bad) for every image."""
    return model.predict(X.astype("float32"), batch_size=batch, verbose=0)[:, 1]


# ------------------------------------------------------------ threshold + eval --

def tune_threshold(y_val: np.ndarray, p_val: np.ndarray, target_recall: float) -> dict[str, float]:
    """Pick the decision threshold on the VALIDATION set.

    Business view: missing a damaged/rotten order (false negative) costs more than an extra
    manual check (false positive), so the threshold must catch at least `target_recall` of the
    bad class on validation. Among the thresholds that do, we take the one with the best F1 for
    the bad class (ties -> the lower, safer threshold).
    Why not simply "the highest threshold that still reaches the target": that value sits right
    at the edge, so on new photos recall drops below the target (we saw 0.90 on validation become
    0.87 on test for produce, while 0.5 would have given 0.95).
    If no threshold reaches the target, we take the one with the best F1 for the bad class.
    """
    from sklearn.metrics import f1_score, recall_score

    candidates = np.unique(np.concatenate([np.round(p_val, 4), [0.5]]))
    f1 = {t: f1_score(y_val, p_val >= t, zero_division=0) for t in candidates}
    ok = [t for t in candidates if recall_score(y_val, p_val >= t, zero_division=0) >= target_recall]
    pool = ok or list(candidates)
    best = max(f1[t] for t in pool)
    t = float(min(c for c in pool if f1[c] == best))
    rule = (f"best val F1(bad) among thresholds with val recall(bad) >= {target_recall}" if ok
            else "best val F1(bad) (target recall not reachable)")
    return {"threshold": round(t, 4), "rule": rule, "val_recall_bad": round(float(recall_score(y_val, p_val >= t)), 4),
            "val_f1_bad": round(float(f1[t]), 4)}


def evaluate_image_model(domain: str, p_test: np.ndarray, y_test: np.ndarray, meta: pd.DataFrame,
                         threshold: float) -> dict[str, Any]:
    """Metrics at the default 0.5 threshold and at the tuned threshold, overall and per group.

    Groups: per source for packages (Kaggle renders vs real photos behave differently) and per
    fruit for produce.
    """
    from src.evaluate import classification_metrics

    names = class_names(domain)

    def as_labels(pred):
        return np.where(pred, names[1], names[0])

    truth = as_labels(y_test.astype(bool))
    out: dict[str, Any] = {"labels": names, "threshold": threshold}
    for key, t in [("test_default", 0.5), ("test", threshold)]:
        out[key] = classification_metrics(truth, as_labels(p_test >= t), labels=names)
    group_col = "source" if domain == "packages" else "subtype"
    out["test_per_group"] = {}
    for g, idx in meta.groupby(group_col).groups.items():
        m = classification_metrics(truth[idx], as_labels(p_test[idx] >= threshold), labels=names)
        out["test_per_group"][g] = {"n": m["n"], "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                                    "recall_bad": m["per_class"][names[1]]["recall"]}
    return out


def save_image_predictions(domain: str, name: str, meta: pd.DataFrame, y: np.ndarray, p_bad: np.ndarray,
                           split: str = "test") -> Path:
    """P(bad) per image, saved so thresholds can be re-tuned later without the model."""
    out = path("results", "predictions", create=True) / f"image_{domain}_{name}_{split}.csv"
    pd.DataFrame({"path": meta["path"], "source": meta["source"], "subtype": meta["subtype"],
                  "label": meta["label"], "is_bad": y, "p_bad": np.round(p_bad, 4)}).to_csv(out, index=False)
    return out


# ------------------------------------------------------------------ Grad-CAM --

def grad_cam(core, images: np.ndarray) -> np.ndarray:
    """Grad-CAM heatmaps (n, H, W) in 0..1: which parts of each photo pushed P(bad) up.

    How it works: take the last convolutional feature map (a 7x7 grid of "what is where"),
    measure how much each feature channel changes P(bad) (its gradient), weight the channels by
    that, and keep the positive part. Bright = this region made the model say "damaged/rotten".
    """
    import keras
    import tensorflow as tf
    from keras import layers

    x = tf.convert_to_tensor(images, tf.float32)
    backbone = next((l for l in core.layers if isinstance(l, keras.Model)), None)
    if backbone is not None:
        # MobileNetV2 sits inside our model as one layer, so we rebuild the path by hand:
        # rescale -> backbone up to its last conv block ("out_relu") -> average pool -> softmax head.
        rescale = next(l for l in core.layers if isinstance(l, layers.Rescaling))
        features = keras.Model(backbone.inputs, backbone.get_layer("out_relu").output)
        head = core.get_layer("probs")
        with tf.GradientTape() as tape:
            fmap = features(rescale(x), training=False)
            score = head(tf.reduce_mean(fmap, axis=(1, 2)))[:, 1]
    else:
        last_conv = [l for l in core.layers if isinstance(l, layers.Conv2D)][-1]
        grad_model = keras.Model(core.inputs, [last_conv.output, core.output])
        with tf.GradientTape() as tape:
            fmap, probs = grad_model(x, training=False)
            score = probs[:, 1]
    grads = tape.gradient(score, fmap)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * fmap, axis=-1)).numpy()
    cam = cam / (cam.max(axis=(1, 2), keepdims=True) + 1e-8)
    size = images.shape[1]
    return np.stack([np.asarray(Image.fromarray((c * 255).astype("uint8")).resize((size, size), Image.BILINEAR))
                     / 255.0 for c in cam])


def plot_grad_cam(core, X: np.ndarray, y: np.ndarray, p_bad: np.ndarray, threshold: float, domain: str,
                  name: str, n: int = 4) -> Any:
    """Grad-CAM on a few correct and a few wrong test predictions (overlaid on the photo)."""
    import matplotlib.pyplot as plt

    from src.plots import apply_style, save

    apply_style()
    pred = p_bad >= threshold
    rng = np.random.default_rng(load_config()["seed"])
    correct = rng.permutation(np.where((pred == y.astype(bool)) & y.astype(bool))[0])[:n]
    wrong = rng.permutation(np.where(pred != y.astype(bool))[0])[:n]
    names = class_names(domain)
    rows = [("correct", correct), ("wrong", wrong)]
    fig, axes = plt.subplots(2, n, figsize=(n * 2.4, 6.4))
    for r, (kind, idx) in enumerate(rows):
        cams = grad_cam(core, X[idx]) if len(idx) else []
        for c in range(n):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(idx):
                i = idx[c]
                ax.imshow(X[i])
                ax.imshow(cams[c], cmap="jet", alpha=0.4)
                ax.set_title(f"{kind}: true {names[y[i]]}\nP({names[1]}) = {p_bad[i]:.2f}", loc="left", fontsize=8)
    fig.suptitle(f"Grad-CAM · {domain} · {name}: where the model looks for '{names[1]}'", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(h_pad=2.5)
    save(fig, f"gradcam_{domain}_{name}")
    return fig


def plot_history(history: dict[str, list[float]], domain: str, name: str) -> Any:
    """Training curves (loss and accuracy, train vs validation)."""
    import matplotlib.pyplot as plt

    from src.plots import apply_style, save

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    epochs = np.arange(1, len(history["loss"]) + 1)
    for ax, metric in zip(axes, ["loss", "accuracy"]):
        ax.plot(epochs, history[metric], color="#2a78d6", linewidth=2, label="train")
        ax.plot(epochs, history[f"val_{metric}"], color="#eb6834", linewidth=2, label="validation")
        if "fine-tune" in history.get("stage", []):  # mark where stage 2 begins
            start = history["stage"].index("fine-tune") + 1
            ax.axvline(start - 0.5, color="#8a8984", linewidth=1, linestyle="--")
            ax.text(start - 0.4, ax.get_ylim()[1], " fine-tuning starts", va="top", fontsize=8, color="#52514e")
        ax.set_xlabel("epoch")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_title(metric, loc="left")
        ax.legend()
    kept = history.get("kept_stage", [""])[0]
    note = f"  (kept: {kept})" if kept else ""
    fig.suptitle(f"{domain} · {name}: training curves{note}", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, f"history_{domain}_{name}")
    return fig


# ------------------------------------------------------------ zero-shot CLIP --

CLIP_PROMPTS = {
    "packages": ["a photo of an intact, undamaged package or box",
                 "a photo of a damaged, crushed, torn or dented package or box"],
    "produce": ["a photo of a fresh fruit", "a photo of a rotten, moldy or spoiled fruit"],
}


def clip_zero_shot(domain: str, X: np.ndarray, batch: int = 64) -> np.ndarray:
    """P(bad) from CLIP with no training at all: it compares each photo to two text descriptions.

    This is the "ready-made model from the internet" baseline for images.
    """
    import torch
    from transformers import CLIPModel, CLIPProcessor

    name = load_config()["images"]["clip_model"]
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = CLIPModel.from_pretrained(name).to(device).eval()
    proc = CLIPProcessor.from_pretrained(name)
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            enc = proc(text=CLIP_PROMPTS[domain], images=list(X[i:i + batch]), return_tensors="pt", padding=True)
            logits = model(**{k: v.to(device) for k, v in enc.items()}).logits_per_image
            out.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------- full run --

def run_experiment(domain: str, kind: str, quick: bool = False, limit: int | None = None) -> dict[str, Any]:
    """Train one model for one domain and save everything the comparison needs.

    Saves: models/train/image_<domain>_<kind>.keras, results/colab/metrics_image_<domain>__<kind>.json,
    test predictions, confusion matrix, training curves and Grad-CAM figures.
    `quick`/`limit` are only for a smoke test (1 epoch, few images).
    """
    import json

    from src.evaluate import plot_confusion_matrix

    X, y, _ = load_split(domain, "train", limit)
    Xv, yv, mv = load_split(domain, "val", limit)
    Xt, yt, mt = load_split(domain, "test", limit)
    core, history = train_model(kind, X, y, Xv, yv, quick=quick)

    p_val = predict_bad(core, Xv)
    save_image_predictions(domain, kind, mv, yv, p_val, split="val")
    tuned = tune_threshold(yv, p_val, load_config()["images"]["target_recall_bad"])
    p_test = predict_bad(core, Xt)
    metrics = evaluate_image_model(domain, p_test, yt, mt, tuned["threshold"])
    metrics.update({"type": {"small_cnn": "CNN from scratch", "mobilenetv2": "MobileNetV2 transfer learning"}[kind],
                    "threshold_tuning": tuned, "epochs_run": len(history["loss"]),
                    "kept_stage": history.get("kept_stage", ["scratch"])[0],
                    "n_train": int(len(y)), "params": int(core.count_params())})

    model_file = path("models_train", create=True) / f"image_{domain}_{kind}.keras"
    core.save(model_file)
    metrics["size_mb"] = round(model_file.stat().st_size / 1e6, 1)
    path("results", "colab", create=True).joinpath(f"metrics_image_{domain}__{kind}.json").write_text(
        json.dumps(metrics, indent=2))
    save_image_predictions(domain, kind, mt, yt, p_test)
    test = metrics["test"]
    plot_confusion_matrix(test["confusion_matrix"], test["labels"],
                          f"{domain} · {kind}  (threshold {tuned['threshold']:.2f})", f"cm_image_{domain}_{kind}")
    plot_history(history, domain, kind)
    plot_grad_cam(core, Xt, yt, p_test, tuned["threshold"], domain, kind)
    bad = test["per_class"][class_names(domain)[1]]
    print(f"[done] {domain}/{kind}: accuracy {test['accuracy']:.3f}, macro-F1 {test['macro_f1']:.3f}, "
          f"recall(bad) {bad['recall']:.3f}, precision(bad) {bad['precision']:.3f}, threshold {tuned['threshold']}")
    return metrics


def reevaluate_saved(domain: str, kind: str) -> dict[str, Any]:
    """Re-score a saved model (models/train/image_<domain>_<kind>.keras) without retraining.

    Used after changing the threshold rule: predicts val + test again, re-tunes the threshold,
    and rewrites the metrics file, keeping the training facts (epochs, size) from the original run.
    """
    import json

    import keras

    from src.evaluate import plot_confusion_matrix

    core = keras.models.load_model(path("models_train") / f"image_{domain}_{kind}.keras")
    Xv, yv, mv = load_split(domain, "val")
    Xt, yt, mt = load_split(domain, "test")
    p_val, p_test = predict_bad(core, Xv), predict_bad(core, Xt)
    save_image_predictions(domain, kind, mv, yv, p_val, split="val")
    save_image_predictions(domain, kind, mt, yt, p_test)
    tuned = tune_threshold(yv, p_val, load_config()["images"]["target_recall_bad"])
    f = path("results", "colab") / f"metrics_image_{domain}__{kind}.json"
    old = json.loads(f.read_text())
    metrics = evaluate_image_model(domain, p_test, yt, mt, tuned["threshold"])
    metrics.update({k: old[k] for k in ("type", "epochs_run", "n_train", "params", "size_mb") if k in old})
    metrics["threshold_tuning"] = tuned
    f.write_text(json.dumps(metrics, indent=2))
    plot_confusion_matrix(metrics["test"]["confusion_matrix"], metrics["test"]["labels"],
                          f"{domain} · {kind}  (threshold {tuned['threshold']:.2f})", f"cm_image_{domain}_{kind}")
    plot_grad_cam(core, Xt, yt, p_test, tuned["threshold"], domain, kind)
    return metrics
