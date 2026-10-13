"""Load config.yaml and give every module the same settings and paths.

Why: CLAUDE.md forbids hard-coded absolute paths. Every script and notebook calls
`load_config()` and `path(...)`, so the project runs the same on my Mac, on Colab
and on Render, no matter which folder it was cloned into.
"""

from __future__ import annotations

import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# The repo root is one level above src/. Resolving it from this file (not from the
# current working directory) means notebooks inside notebooks/ still find config.yaml.
ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """Read config.yaml once and cache it."""
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(key: str, *parts: str, create: bool = False) -> Path:
    """Absolute path for a key in `paths:` of config.yaml, plus optional sub-parts.

    Example: path("data_raw", "kaggle") -> <repo>/data/raw/kaggle
    """
    p = ROOT / load_config()["paths"][key]
    p = p.joinpath(*parts)
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def load_env() -> None:
    """Load secrets (API keys) from a .env file in the repo root, if it exists.

    Secrets never go into config.yaml or git; they live in .env or real env vars.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:  # the slim app requirements may skip python-dotenv
        return
    load_dotenv(ROOT / ".env")


def set_seed(seed: int | None = None) -> int:
    """Fix random seeds so results can be reproduced (seed 42 by default)."""
    seed = load_config()["seed"] if seed is None else seed
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    return seed
