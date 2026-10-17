"""Export the deployed models to ONNX for the website.

Run:  python -m src.export_onnx

Why ONNX: the website runs on a 512 MB server. torch / tensorflow / transformers would take most
of that memory just by being imported. ONNX Runtime runs the same networks with a ~15 MB library,
so the app only needs onnxruntime, tokenizers, numpy and Pillow (see src/inference.py).

Writes to models/deploy/ (committed to git, must stay < 100 MB):
    text_distilbert.int8.onnx     fine-tuned DistilBERT, weights quantised to 8-bit integers
    tokenizer.json                the exact tokenizer used in training
    image_packages.onnx           MobileNetV2 damaged-vs-intact
    image_produce.onnx            MobileNetV2 rotten-vs-fresh
    settings.json                 labels, max length, image size, thresholds, routing
and tests/fixtures/parity_reference.npz: original-model outputs on 50 samples per model
(for tests/test_inference_parity.py; kept out of models/deploy because the website doesn't need it).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ROOT, load_config, path

N_PARITY = 50


def deploy_dir() -> Path:
    return path("models_deploy", create=True)


# ---------------------------------------------------------------------- text --

def export_text(max_length: int) -> dict:
    """DistilBERT -> ONNX (float32) -> ONNX with 8-bit weights.

    Dynamic int8 quantisation stores every weight matrix as 8-bit integers plus a scale, which
    makes the file ~4x smaller and CPU inference faster, at the cost of a tiny change in the
    outputs (measured by the parity test and by re-scoring the whole test set).
    """
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    src = path("models_train") / "text_distilbert_finetuned"
    out = deploy_dir()
    # "eager" = the plain attention code. The default fast version (SDPA) contains Python-level
    # length checks that the ONNX tracer would freeze at the example's length.
    model = AutoModelForSequenceClassification.from_pretrained(src, attn_implementation="eager").eval()
    tok = AutoTokenizer.from_pretrained(src)

    fp32 = path("models_train") / "text_distilbert.fp32.onnx"
    dummy = tok(["a short example review"], return_tensors="pt", truncation=True, max_length=max_length)
    torch.onnx.export(
        model, (dummy["input_ids"], dummy["attention_mask"]), str(fp32),
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "tokens"}, "attention_mask": {0: "batch", 1: "tokens"},
                      "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    int8 = out / "text_distilbert.int8.onnx"
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    shutil.copy(src / "tokenizer.json", out / "tokenizer.json")

    # Reference outputs of the ORIGINAL PyTorch model on 50 test reviews.
    test = pd.read_csv(path("data_processed") / "text_test.csv").sample(N_PARITY, random_state=load_config()["seed"])
    with torch.no_grad():
        enc = tok(test["text"].tolist(), truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        ref = torch.softmax(model(**enc).logits, -1).numpy()
    return {"fp32_mb": fp32.stat().st_size / 1e6, "int8_mb": int8.stat().st_size / 1e6,
            "texts": test["text"].tolist(), "reference": ref}


# -------------------------------------------------------------------- images --

def export_image(domain: str) -> dict:
    """MobileNetV2 (.keras) -> ONNX, kept in float32 (only ~9 MB, and 8-bit convolutions can cost accuracy).

    The model already contains its preprocessing (rescaling), so the ONNX input is simply the
    photo resized to 224x224 as float pixels 0-255.
    """
    import keras

    from src.image_models import load_split

    core = keras.models.load_model(path("models_train") / f"image_{domain}_mobilenetv2.keras")
    out = deploy_dir() / f"image_{domain}.onnx"
    X, _, meta = load_split(domain, "test")
    core.predict(X[:2].astype("float32"), verbose=0)  # Keras only exports a model that has been run once
    core.export(str(out), format="onnx")
    idx = np.random.default_rng(load_config()["seed"]).choice(len(X), N_PARITY, replace=False)
    ref = core.predict(X[idx].astype("float32"), verbose=0)
    return {"mb": out.stat().st_size / 1e6, "images": X[idx], "reference": ref, "paths": meta["path"].iloc[idx].tolist()}


# ------------------------------------------------------------------ settings --

def write_settings(max_length: int) -> dict:
    """Everything the website needs besides the network weights, in one small JSON file."""
    cfg = load_config()
    metrics = json.loads((path("results") / "metrics.json").read_text())
    settings = {
        "text": {"model": "text_distilbert.int8.onnx", "tokenizer": "tokenizer.json",
                 "labels": cfg["text"]["labels"], "max_length": max_length},
        "images": {
            domain: {"model": f"image_{domain}.onnx", "labels": metrics[f"image_{domain}"]["mobilenetv2"]["labels"],
                     "threshold": metrics[f"image_{domain}"]["mobilenetv2"]["threshold"],
                     "size": cfg["images"]["size"], "input": "RGB, resized (stretch) to size x size, float32 0-255"}
            for domain in ("packages", "produce")
        },
        "routing": cfg["routing"],
    }
    (deploy_dir() / "settings.json").write_text(json.dumps(settings, indent=2))
    return settings


def main() -> None:
    max_length = load_config()["text"]["finetune"]["max_length"]
    text = export_text(max_length)
    print(f"[text] DistilBERT ONNX: float32 {text['fp32_mb']:.1f} MB -> int8 {text['int8_mb']:.1f} MB")
    images = {}
    for domain in ("packages", "produce"):
        images[domain] = export_image(domain)
        print(f"[image] {domain}: {images[domain]['mb']:.1f} MB")
    write_settings(max_length)
    fixtures = ROOT / "tests" / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        fixtures / "parity_reference.npz",
        text_texts=np.array(text["texts"]), text_reference=text["reference"],
        **{f"{d}_images": v["images"] for d, v in images.items()},
        **{f"{d}_reference": v["reference"] for d, v in images.items()},
    )
    total = sum(f.stat().st_size for f in deploy_dir().iterdir()) / 1e6
    print(f"[deploy] models/deploy/ total {total:.1f} MB")


if __name__ == "__main__":
    main()
