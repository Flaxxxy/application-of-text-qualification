"""Train one image model locally (CPU) instead of on Colab. Slower, same code and seed.

Usage:  python scripts/retrain_local.py <domain> <kind>     e.g. produce small_cnn
The metrics file records that the model was trained locally.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import path, set_seed  # noqa: E402
from src.image_models import run_experiment  # noqa: E402

domain, kind = sys.argv[1], sys.argv[2]
set_seed()
run_experiment(domain, kind)
f = path("results", "colab") / f"metrics_image_{domain}__{kind}.json"
m = json.loads(f.read_text())
m["trained_on"] = "local CPU (Mac), same code and seed as Colab"
f.write_text(json.dumps(m, indent=2))
