"""Deploy-safe inference: runs the exported ONNX models with no torch / tensorflow / transformers.

Only onnxruntime, tokenizers, numpy and Pillow are used, so the website fits in 512 MB of RAM.
Models are loaded once (when `Predictor()` is created) and reused for every request.

    from src.inference import Predictor
    p = Predictor()
    p.text("The box arrived crushed and the mug inside was broken")
    p.image(Image.open("photo.jpg"), category="electronics")

The preprocessing here must match training exactly, or the predictions silently drift:
    text  : basic_clean (HTML/URLs removed) -> the training tokenizer (lowercase, WordPiece), max 256 tokens
    image : RGB, EXIF rotation applied, stretched to 224x224 with bilinear resizing, float pixels 0-255
            (the rescaling to the network's range is inside the ONNX model)
tests/test_inference_parity.py checks this against the original models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from src.config import path
from src.issue_tags import tag_issues
from src.preprocess_text import basic_clean


def _session(file: Path, threads: int):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads  # the free Render server has a fraction of one CPU
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(file), opts, providers=["CPUExecutionProvider"])


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class TextSentiment:
    """Fine-tuned DistilBERT (8-bit ONNX): review text -> P(negative), P(neutral), P(positive)."""

    def __init__(self, deploy: Path, settings: dict, threads: int = 1):
        from tokenizers import Tokenizer

        self.labels = settings["labels"]
        self.tok = Tokenizer.from_file(str(deploy / settings["tokenizer"]))
        self.tok.enable_truncation(max_length=settings["max_length"])
        pad_id = self.tok.token_to_id("[PAD]")
        self.tok.enable_padding(pad_id=pad_id, pad_token="[PAD]")
        self.session = _session(deploy / settings["model"], threads)

    def predict_proba(self, texts: list[str], batch: int = 16) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch):
            enc = self.tok.encode_batch([basic_clean(t) for t in texts[i:i + batch]])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            logits = self.session.run(["logits"], {"input_ids": ids, "attention_mask": mask})[0]
            out.append(_softmax(logits))
        return np.vstack(out)


class ImageChecker:
    """MobileNetV2 (ONNX) for one domain: photo -> P(bad), plus the tuned decision threshold."""

    def __init__(self, deploy: Path, settings: dict, threads: int = 1):
        self.labels = settings["labels"]          # [good, bad]
        self.threshold = settings["threshold"]
        self.size = settings["size"]
        self.session = _session(deploy / settings["model"], threads)
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, img: Image.Image) -> np.ndarray:
        img = ImageOps.exif_transpose(img).convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        return np.asarray(img, dtype=np.float32)[None]

    def predict_bad(self, images: list[Image.Image] | np.ndarray) -> np.ndarray:
        x = images.astype(np.float32) if isinstance(images, np.ndarray) else np.vstack([self.preprocess(i) for i in images])
        return self.session.run(None, {self.input_name: x})[0][:, 1]


class Predictor:
    """All deployed models together, loaded once. This is what the website and CLI use."""

    def __init__(self, deploy: Path | None = None, threads: int = 1):
        self.deploy = deploy or path("models_deploy")
        self.settings = json.loads((self.deploy / "settings.json").read_text())
        self.text_model = TextSentiment(self.deploy, self.settings["text"], threads)
        self.image_models = {d: ImageChecker(self.deploy, s, threads) for d, s in self.settings["images"].items()}

    def text(self, review: str) -> dict[str, Any]:
        """Sentiment probabilities, predicted label and issue tags (with the phrase that triggered each)."""
        p = self.text_model.predict_proba([review])[0]
        probs = {lab: round(float(v), 4) for lab, v in zip(self.text_model.labels, p)}
        return {"probs": probs, "label": max(probs, key=probs.get), "tags": tag_issues(review)}

    def image_domain(self, category: str) -> str:
        """Which photo model handles a product category (electronics -> packages, grocery -> produce)."""
        routing = self.settings["routing"]
        return routing.get(category, routing["default"])

    def image(self, img: Image.Image, category: str) -> dict[str, Any]:
        """P(bad) for one photo, using the model that matches the order's category."""
        domain = self.image_domain(category)
        model = self.image_models[domain]
        p_bad = float(model.predict_bad([img])[0])
        return {"domain": domain, "p_bad": round(p_bad, 4), "threshold": model.threshold,
                "is_bad": p_bad >= model.threshold, "label": model.labels[int(p_bad >= model.threshold)]}
