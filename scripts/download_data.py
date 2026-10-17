"""Download every dataset the project needs into data/raw/.

Usage (from the repo root):
    python scripts/download_data.py --check            # test keys + sizes, download nothing
    python scripts/download_data.py text               # all text sources
    python scripts/download_data.py images             # all image sources
    python scripts/download_data.py demo               # demo photos (needs the Amazon 2023 text sample first)
    python scripts/download_data.py text --only flipkart
    python scripts/download_data.py images --allow-large   # if a dataset is bigger than download.max_size_gb

Three kinds of sources (set by `kind:` in config.yaml):
    kaggle     -> Kaggle API, needs KAGGLE_API_TOKEN (or KAGGLE_USERNAME + KAGGLE_KEY) in .env
    roboflow   -> Roboflow API, needs ROBOFLOW_API_KEY in .env
    hf_sample  -> public Hugging Face file, no key; we read small random pieces of it

Already-downloaded sources are skipped (a `.done` marker file is written at the end),
so the script is safe to run again after a failure.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

# Make `src` importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config, load_env, path  # noqa: E402

GB = 1024**3


# ------------------------------------------------------------------ helpers --

def _done_marker(folder: Path) -> Path:
    return folder / ".done"


def _already_done(folder: Path, name: str) -> bool:
    if _done_marker(folder).exists():
        print(f"[skip] {name}: already downloaded in {folder}")
        return True
    return False


def _too_big(size_bytes: int | None, name: str, allow_large: bool) -> bool:
    """Stop before huge downloads (above download.max_size_gb in config.yaml) unless --allow-large is given."""
    limit_gb = load_config()["download"]["max_size_gb"]
    if size_bytes is None:
        print(f"[warn] {name}: size unknown, continuing")
        return False
    size_gb = size_bytes / GB
    print(f"[info] {name}: {size_gb:.2f} GB")
    if size_gb > limit_gb and not allow_large:
        print(f"[stop] {name} is larger than {limit_gb} GB. Re-run with --allow-large if that's OK.")
        return True
    return False


# ------------------------------------------------------------------- kaggle --

def _kaggle_api():
    """Authenticate with Kaggle. Import happens here because `import kaggle` itself
    tries to log in and crashes without credentials."""
    load_env()
    if not (os.getenv("KAGGLE_API_TOKEN") or (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))):
        if not (Path.home() / ".kaggle" / "kaggle.json").exists():
            raise SystemExit("Kaggle key missing: set KAGGLE_API_TOKEN (or KAGGLE_USERNAME + KAGGLE_KEY) in .env")
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def _kaggle_size(api, slug: str) -> int | None:
    """Total size of a Kaggle dataset in bytes, looked up through the dataset search."""
    owner, name = slug.split("/")
    try:
        for d in api.dataset_list(search=name, user=owner) or []:
            if str(getattr(d, "ref", "")).lower() == slug.lower():
                return getattr(d, "total_bytes", None) or getattr(d, "totalBytes", None)
    except Exception as e:  # the size is only a safety check, so never fail on it
        print(f"[warn] could not read size of {slug}: {e}")
    return None


def download_kaggle(name: str, src: dict[str, Any], out_root: Path, allow_large: bool, check: bool) -> None:
    api = _kaggle_api()
    out = out_root / name
    size = _kaggle_size(api, src["slug"])
    if check:
        print(f"[ok]   {name}: kaggle {src['slug']} reachable, "
              f"{'size unknown' if size is None else f'{size / GB:.2f} GB'}")
        return
    if _already_done(out, name) or _too_big(size, name, allow_large):
        return
    out.mkdir(parents=True, exist_ok=True)
    print(f"[get]  {name}: kaggle {src['slug']} -> {out}")
    api.dataset_download_files(src["slug"], path=str(out), unzip=True, quiet=False)
    _done_marker(out).touch()


# ----------------------------------------------------------------- roboflow --

def download_roboflow(name: str, src: dict[str, Any], out_root: Path, allow_large: bool, check: bool) -> None:
    load_env()
    key = os.getenv("ROBOFLOW_API_KEY")
    if not key:
        raise SystemExit("Roboflow key missing: set ROBOFLOW_API_KEY in .env")
    from roboflow import Roboflow

    out = out_root / name
    version = Roboflow(api_key=key).workspace(src["workspace"]).project(src["project"]).version(src["version"])
    if check:
        print(f"[ok]   {name}: roboflow {src['workspace']}/{src['project']} v{src['version']} reachable")
        return
    if _already_done(out, name):
        return
    print(f"[get]  {name}: roboflow -> {out}")
    version.download(src["format"], location=str(out), overwrite=True)
    _done_marker(out).touch()


# ------------------------------------------------------- hugging face sample --

def _hf_url(src: dict[str, Any]) -> str:
    return f"https://huggingface.co/datasets/{src['repo']}/resolve/main/{src['file']}"


def _remote_size(url: str) -> int:
    r = requests.head(url, allow_redirects=True, timeout=30)
    r.raise_for_status()
    return int(r.headers["Content-Length"])


def _read_chunk(url: str, offset: int, n_lines: int, retries: int = 4) -> tuple[list[str], int]:
    """Read n_lines complete JSON lines starting at a byte offset, retrying on network errors.

    Lines are collected in memory and returned only when the chunk is complete, so a
    failed attempt never leaves half a chunk (or duplicates) in the output file.
    """
    for attempt in range(1, retries + 1):
        try:
            lines_out, bad = [], 0
            with requests.get(url, headers={"Range": f"bytes={offset}-"}, stream=True, timeout=60) as r:
                r.raise_for_status()
                lines = r.iter_lines(decode_unicode=False)
                next(lines, None)  # the first line is cut in the middle by the random offset
                for _ in range(n_lines):
                    raw = next(lines, None)
                    if raw is None:
                        break
                    try:
                        json.loads(raw)  # keep only complete, valid JSON lines
                    except ValueError:
                        bad += 1
                        continue
                    lines_out.append(raw.decode("utf-8"))
            return lines_out, bad
        except requests.RequestException as e:
            if attempt == retries:
                raise
            print(f"[retry] offset {offset}: {type(e).__name__}, attempt {attempt + 1}/{retries}")
            time.sleep(5 * attempt)
    return [], 0  # not reached


def download_hf_sample(name: str, src: dict[str, Any], out_root: Path, allow_large: bool, check: bool) -> None:
    """Read `n_chunks` pieces of `lines_per_chunk` lines from random places in a huge jsonl.

    Why not just the first 300k lines: the Amazon 2023 files are grouped by user, so
    the top of the file is a handful of reviewers. HTTP Range requests let us jump to
    any byte offset and read from there, without downloading the 22 GB file.
    """
    cfg = load_config()
    n_chunks = cfg["text"]["hf_sample"]["n_chunks"]
    lines_per_chunk = cfg["text"]["hf_sample"]["lines_per_chunk"]
    url = _hf_url(src)
    size = _remote_size(url)
    if check:
        print(f"[ok]   {name}: {src['file']} reachable, {size / GB:.1f} GB on server "
              f"(we read ~{n_chunks * lines_per_chunk:,} lines only)")
        return

    out = out_root / name
    if _already_done(out, name):
        return
    out.mkdir(parents=True, exist_ok=True)

    # Split the file into n equal slices and pick one random start point inside each.
    # One point per slice spreads the sample over the whole file and avoids overlaps.
    rng = random.Random(cfg["seed"])
    stride = size // n_chunks
    offsets = [i * stride + rng.randint(0, max(stride // 2, 1)) for i in range(n_chunks)]

    out_file = out / "sample.jsonl"
    kept = bad = 0
    with open(out_file, "w", encoding="utf-8") as f:
        for offset in tqdm(offsets, desc=f"{name} chunks"):
            chunk, chunk_bad = _read_chunk(url, offset, lines_per_chunk)
            f.writelines(line + "\n" for line in chunk)
            kept += len(chunk)
            bad += chunk_bad
    print(f"[done] {name}: {kept:,} reviews saved to {out_file} ({bad} broken lines skipped)")
    _done_marker(out).touch()


# -------------------------------------------------------------- demo photos --

def _best_image_url(img: dict[str, Any]) -> str | None:
    for key in ("large_image_url", "medium_image_url", "small_image_url"):
        if img.get(key):
            return img[key]
    return None


def download_demo_photos(text_root: Path, out: Path, check: bool) -> None:
    """Pick a few customer photos from the Amazon 2023 samples, for the website demo.

    These are unlabelled and never used for training. They stay in data/ (git-ignored)
    because the dataset has no clear licence for redistributing customer photos.
    """
    cfg = load_config()
    max_images = cfg["images"]["demo_photos"]["max_images"]
    candidates = []
    for name, src in cfg["text"]["sources"].items():
        if src["kind"] != "hf_sample":
            continue
        sample = text_root / name / "sample.jsonl"
        if not sample.exists():
            print(f"[warn] {sample} missing: run `download_data.py text` first")
            continue
        with open(sample, encoding="utf-8") as f:
            for line in f:
                review = json.loads(line)
                for img in review.get("images") or []:
                    url = _best_image_url(img)
                    if url:
                        candidates.append((src["category"], review, url))
                        break  # one photo per review is enough
    print(f"[info] demo photos: {len(candidates):,} reviews with photos found")
    if check or not candidates:
        return

    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg["seed"])
    picked = rng.sample(candidates, min(max_images, len(candidates)))
    meta = []
    for i, (category, review, url) in enumerate(tqdm(picked, desc="demo photos")):
        fname = f"{category}_{i:02d}.jpg"
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"[warn] could not get {url}: {e}")
            continue
        (out / fname).write_bytes(r.content)
        meta.append({"file": fname, "category": category, "rating": review.get("rating"),
                     "title": review.get("title"), "text": review.get("text"), "url": url})
    with open(out / "photos.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[done] {len(meta)} demo photos saved to {out}")


# --------------------------------------------------------------------- main --

DOWNLOADERS = {"kaggle": download_kaggle, "roboflow": download_roboflow, "hf_sample": download_hf_sample}


def run_group(group: str, only: str | None, allow_large: bool, check: bool) -> None:
    """Download every source of one group ("text" or "images") from config.yaml."""
    cfg = load_config()
    out_root = path("data_raw", group, create=True)
    sources = cfg[group]["sources"]
    if only:
        if only not in sources:
            raise SystemExit(f"Unknown source '{only}'. Choose from: {', '.join(sources)}")
        sources = {only: sources[only]}
    failed = []
    for name, src in sources.items():
        try:
            DOWNLOADERS[src["kind"]](name, src, out_root, allow_large, check)
        except SystemExit as e:
            if not check:
                raise  # missing key: stop and tell the user
            print(f"[FAIL] {name}: {e}")  # in --check mode, list every problem at once
            failed.append(name)
        except Exception as e:  # one broken source should not hide the others
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
            failed.append(name)
    if failed and check:
        print(f"[check] problems with: {', '.join(failed)}")
    elif failed:
        raise SystemExit(f"Failed sources: {', '.join(failed)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("group", nargs="?", choices=["text", "images", "demo", "all"], default="all")
    parser.add_argument("--only", help="download just one source by its name in config.yaml")
    parser.add_argument("--allow-large", action="store_true", help="allow datasets above download.max_size_gb")
    parser.add_argument("--check", action="store_true", help="test keys and sizes, download nothing")
    args = parser.parse_args()

    groups = ["text", "images", "demo"] if args.group == "all" else [args.group]
    for group in groups:
        print(f"\n=== {group} ===")
        if group == "demo":
            download_demo_photos(path("data_raw", "text"), path("data_raw", "demo_photos"), args.check)
        else:
            run_group(group, args.only, args.allow_large, args.check)


if __name__ == "__main__":
    main()
