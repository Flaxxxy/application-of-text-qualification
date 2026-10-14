"""Parity tests: the ONNX models used by the website must agree with the original trained models.

tests/fixtures/parity_reference.npz holds the ORIGINAL models' outputs (PyTorch DistilBERT and
Keras MobileNetV2) on 50 test samples each, saved by `python -m src.export_onnx`. These tests run
the deployed ONNX models through src/inference.py, so they also check the app's preprocessing.
They need only the slim app requirements (no torch / tensorflow).
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "parity_reference.npz"
pytestmark = pytest.mark.skipif(not (ROOT / "models" / "deploy" / "settings.json").exists(),
                                reason="models/deploy not exported yet")


@pytest.fixture(scope="module")
def predictor():
    from src.inference import Predictor

    return Predictor()


@pytest.fixture(scope="module")
def ref():
    return np.load(FIXTURE)


def test_text_int8_matches_original(predictor, ref):
    # 8-bit weights change probabilities slightly (measured max ~0.15, mean ~0.02 on these 50),
    # but the predicted class must stay the same; full test set: macro-F1 0.7678 vs 0.7681.
    p = predictor.text_model.predict_proba(list(ref["text_texts"]))
    r = ref["text_reference"]
    assert (p.argmax(1) == r.argmax(1)).mean() >= 0.96
    assert np.abs(p - r).mean() < 0.05


@pytest.mark.parametrize("domain", ["packages", "produce"])
def test_image_onnx_matches_keras(predictor, ref, domain):
    # Same float32 weights, so outputs should be equal up to rounding noise.
    model = predictor.image_models[domain]
    p = model.predict_bad(ref[f"{domain}_images"])
    r = ref[f"{domain}_reference"][:, 1]
    assert np.abs(p - r).max() < 1e-4
    assert ((p >= model.threshold) == (r >= model.threshold)).all()


def test_image_preprocessing_matches_training(predictor):
    # The app resizes uploaded photos itself; a 224x224 training image must pass through unchanged.
    from PIL import Image

    img = Image.open(next((ROOT / "data" / "processed" / "images" / "produce" / "test").glob("*/*.jpg"))) \
        if (ROOT / "data" / "processed").exists() else Image.new("RGB", (224, 224), (120, 60, 30))
    x = predictor.image_models["produce"].preprocess(img)
    assert x.shape == (1, 224, 224, 3) and x.dtype == np.float32
    assert np.allclose(x[0], np.asarray(img.convert("RGB"), dtype=np.float32))


def test_inference_does_not_import_heavy_libraries():
    # The website must fit in 512 MB: importing and using src.inference must not pull in
    # torch, tensorflow, transformers or keras.
    code = ("import sys; from src.inference import Predictor; p = Predictor(); p.text('fine product'); "
            "print([m for m in ('torch', 'tensorflow', 'transformers', 'keras') if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "[]"
