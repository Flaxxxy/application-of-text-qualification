# Progress log

Running log of what was done, decisions made, and open issues. Newest phase at the bottom.

## Phase 0 — Setup (2026-10-13)

**Done**
- Repo structure, `config.yaml`, `src/config.py`, `requirements-train.txt`, `requirements-app.txt`, `.gitignore`, `.env.example`.
- `scripts/download_data.py`: Kaggle API, Roboflow API, Hugging Face random-chunk sampling, demo photos. `--check` mode tests keys and sizes without downloading.

**Verified sources (2026-10-13)**
- All 5 Kaggle slugs exist (page titles checked; a fake slug returns 404).
- `McAuley-Lab/Amazon-Reviews-2023` is public (not gated, no HF token needed). Electronics.jsonl = 22.6 GB, Grocery_and_Gourmet_Food.jsonl = 6.0 GB. Server supports byte-range requests.
- Roboflow `aadhavs-first-workspace/damaged-vs-good-packages`: classification, CC BY 4.0, version 1. Image count unknown until the API key is set.

**Decisions**
- Amazon 2023 sampling: the jsonl files are grouped by user, so "first 300k lines" would come from few reviewers. Instead: 30 random chunks × 10k lines per category (still ~300k lines, ~200 MB, never the full file). Configurable in `config.yaml → text.hf_sample`.
- Demo photos are saved to `data/raw/demo_photos/` (git-ignored), not `demo_data/`, because the Amazon dataset has no clear licence for redistributing customer photos. Decide at Phase 6/7 which few to commit.
- Local environment: conda env `rrf` with Python 3.11 (system Python is 3.15, too new for TensorFlow/onnx).

**Open issues**
- Need Kaggle + Roboflow keys from Denis.
- Fruits fresh/rotten size not checked yet (needs Kaggle key); may be > 2 GB → will ask before downloading.
- tf2onnx + Keras 3 compatibility to be checked in Phase 5 (known pain point); pin versions then.
