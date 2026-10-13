"""Build the image datasets: index raw photos -> check -> remove near-duplicates -> resize -> split.

Two domains (CLAUDE.md 5.2), one binary model each:
    packages: damaged vs intact   (Kaggle + Roboflow, merged)  -> misdeliveries / shipping damage
    produce:  rotten  vs fresh    (Kaggle fruits, fruit type kept as metadata) -> product defects

Run everything:   python -m src.data_images
Or step by step from notebooks/04_image_data.ipynb.

Output: data/processed/images/<domain>/{train,val,test}/<class>/<source>__<name>.jpg
plus a manifest CSV per domain with source, subtype (view or fruit), split and hash of every image.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from src.config import load_config, path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Class names per domain: the first one is the "bad" class the business cares about.
CLASSES = {"packages": ["damaged", "intact"], "produce": ["rotten", "fresh"]}


# ------------------------------------------------------------------ indexing --

def _images_under(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXT and not p.name.startswith("."))


def index_packages_kaggle() -> pd.DataFrame:
    """Kaggle packages: damaged|intact / side|top / *.png (a lab setup with two camera views)."""
    root = path("data_raw", "images", "packages_kaggle")
    rows = []
    for p in _images_under(root):
        label, view = p.relative_to(root).parts[:2]
        rows.append({"path": p, "label": label, "subtype": f"{view} view"})
    return pd.DataFrame(rows).assign(source="packages_kaggle", domain="packages")


def index_packages_roboflow() -> pd.DataFrame:
    """Roboflow packages: train|valid|test / damaged|intact / *.jpg.

    We ignore Roboflow's own split and re-split everything together with the Kaggle images,
    so both sources are spread over train/val/test in the same way.
    """
    root = path("data_raw", "images", "packages_roboflow")
    rows = []
    for p in _images_under(root):
        _, label = p.relative_to(root).parts[:2]
        rows.append({"path": p, "label": label, "subtype": "photo"})
    return pd.DataFrame(rows).assign(source="packages_roboflow", domain="packages")


_FRUIT_RE = re.compile(r"^(fresh|rotten)\s*([a-z]+?)s?$")


def index_produce_kaggle() -> pd.DataFrame:
    """Kaggle fruits: <split>/<fresh|rotten><fruit>/*.png, e.g. train/rottenbanana/xyz.png.

    Six classes (fresh/rotten x apple/banana/orange) are collapsed into fresh vs rotten;
    the fruit type is kept in `subtype` so we can report accuracy per fruit later.
    """
    # The Kaggle zip contains the whole dataset twice (dataset/ and dataset/dataset/), so we
    # read only the top copy.
    root = path("data_raw", "images", "produce_kaggle", "dataset")
    rows = []
    for p in _images_under(root):
        if p.relative_to(root).parts[0] == "dataset":
            continue
        m = _FRUIT_RE.match(p.parent.name.lower().replace("_", ""))
        if m:
            rows.append({"path": p, "label": m.group(1), "subtype": m.group(2)})
    return pd.DataFrame(rows).assign(source="produce_kaggle", domain="produce")


def index_domain(domain: str) -> pd.DataFrame:
    """All raw images of one domain in one table."""
    parts = {"packages": [index_packages_kaggle, index_packages_roboflow],
             "produce": [index_produce_kaggle]}[domain]
    df = pd.concat([f() for f in parts], ignore_index=True)
    return df[["domain", "source", "label", "subtype", "path"]]


# ------------------------------------------------------------- quality check --

def check_readable(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Open every image fully; return (good rows with size info, broken rows).

    `verify()` catches truncated/corrupt files cheaply, but some broken files only fail
    when the pixels are actually decoded, so we also call `load()` on a fresh handle.
    """
    good, bad = [], []
    for row in df.itertuples(index=False):
        try:
            with Image.open(row.path) as im:
                im.verify()
            with Image.open(row.path) as im:
                im.load()
                good.append({**row._asdict(), "width": im.width, "height": im.height, "mode": im.mode})
        except Exception as e:  # any decode problem means the file is unusable
            bad.append({**row._asdict(), "error": f"{type(e).__name__}: {e}"})
    return pd.DataFrame(good), pd.DataFrame(bad)


# ------------------------------------------------------------ near-duplicates --
# First attempt: perceptual hashes. Kept only for the notebook, to show why they were not
# good enough here; the pipeline itself uses SSCD (below).

_DIHEDRAL = [None, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]


def perceptual_hashes(p: Path) -> list[int]:
    """8 perceptual hashes of one image: 4 rotations x (normal, mirrored).

    A perceptual hash (64 bits) barely changes when a photo is re-compressed, re-coloured,
    slightly cropped or resized, so near-copies get hashes that differ in only a few bits.
    It does change completely when the photo is rotated or flipped, and rotating/flipping is
    the most common way datasets make "augmented copies". Hashing all 8 orientations lets us
    catch those copies too. Element 0 is the hash of the image as it is.
    """
    import imagehash

    with Image.open(p) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        hashes = []
        for mirrored in (im, ImageOps.mirror(im)):
            for rot in _DIHEDRAL:
                v = mirrored if rot is None else mirrored.transpose(rot)
                hashes.append(int(str(imagehash.phash(v)), 16))
    return hashes


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Number of differing bits between every hash in `a` and every hash in `b`."""
    x = np.bitwise_xor(a[:, None], b[None, :])
    return _POPCOUNT[x.view(np.uint8).reshape(*x.shape, 8)].sum(axis=-1)


def orientation_free_distance(variants_a: np.ndarray, hashes_b: np.ndarray) -> np.ndarray:
    """Distance between images a (all 8 orientations) and b (as is): the best-matching orientation."""
    return np.min([hamming_matrix(variants_a[:, k], hashes_b) for k in range(variants_a.shape[1])], axis=0)


# --- Copy detection with SSCD -------------------------------------------------
# Why not only the perceptual hash above: in these datasets most copies are rotated by
# random angles, cropped, colour-shifted or have noise added. A hash can't tell those copies
# apart from genuinely different fruits on a white background (we checked by eye), and CLIP
# embeddings group *similar-looking* fruits rather than *copies*. SSCD (Meta AI, MIT licence)
# is a model trained specifically for copy detection: it gives an image and its edited copies
# almost the same 512-number fingerprint, and different photos clearly different ones.

SSCD_URL = "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt"
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _sscd_model():
    import urllib.request

    import torch

    f = path("models_train", create=True) / "sscd_disc_mixup.torchscript.pt"
    if not f.exists():
        urllib.request.urlretrieve(SSCD_URL, f)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    return torch.jit.load(str(f), map_location="cpu").to(device).eval(), device


def sscd_embeddings(paths: list[Path], batch: int = 64) -> np.ndarray:
    """L2-normalised SSCD fingerprints (n, 512), cached in data/interim/sscd_cache.npz.

    The cache is keyed by file path, so re-running the pipeline (or the notebook) only
    computes fingerprints for new images. 13.6k large fruit images take ~20 min the first time.
    """
    import torch

    cache_file = path("data_interim", create=True) / "sscd_cache.npz"
    cache: dict[str, np.ndarray] = {}
    if cache_file.exists():
        z = np.load(cache_file, allow_pickle=False)
        cache = dict(zip(z["paths"].tolist(), z["emb"]))
    todo = [p for p in map(str, paths) if p not in cache]
    if todo:
        model, device = _sscd_model()
        with torch.no_grad():
            for i in range(0, len(todo), batch):
                # SSCD's "skew_320" preprocessing: resize to 320x320 + ImageNet normalisation
                x = np.stack([((np.asarray(Image.open(p).convert("RGB").resize((320, 320), Image.BILINEAR),
                                           dtype=np.float32) / 255 - _MEAN) / _STD).transpose(2, 0, 1)
                              for p in todo[i:i + batch]])
                e = model(torch.from_numpy(x).to(device)).cpu().numpy()
                for p, v in zip(todo[i:i + batch], e):
                    cache[p] = v / np.linalg.norm(v)
        np.savez(cache_file, paths=np.array(list(cache)), emb=np.stack(list(cache.values())))
    return np.stack([cache[str(p)] for p in paths]).astype(np.float32)


def similarity_thresholds(sources: pd.Series) -> np.ndarray:
    """Min SSCD similarity that counts as a copy, per image (some sources need a stricter rule)."""
    cfg = load_config()["images"]["dedup"]
    overrides = cfg.get("overrides") or {}
    return sources.map(lambda s: overrides.get(s, cfg["min_similarity"])).to_numpy()


def copy_groups(emb: np.ndarray, thresholds: np.ndarray, seed: int) -> np.ndarray:
    """Group copies with "leader" clustering; returns the leader index for every image.

    Images are visited in a random (seeded) order. Each image joins the most similar existing
    leader if the similarity is above the threshold, otherwise it becomes a new leader.
    Why not "link every similar pair": copy-of-a-copy chains then merge thousands of
    different fruits into one giant group. Here every member is directly similar to its leader.
    """
    order = np.random.default_rng(seed).permutation(len(emb))
    leaders: list[int] = []
    assign = np.empty(len(emb), dtype=int)
    L = np.zeros((0, emb.shape[1]), dtype=emb.dtype)
    for i in order:
        if leaders:
            sims = L @ emb[i]
            k = int(sims.argmax())
            if sims[k] >= min(thresholds[i], thresholds[leaders[k]]):
                assign[i] = leaders[k]
                continue
        leaders.append(i)
        assign[i] = i
        L = np.vstack([L, emb[i]])
    return assign


def remove_near_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep one image per copy group; drop groups whose copies disagree on the label.

    Why before splitting: if two copies of one photo end up in train and test, the model
    has effectively seen the test image, and test accuracy is inflated. Removing copies first
    makes that very unlikely (a final check after the split catches the rest). If the same
    photo is labelled both "damaged" and "intact", we cannot know which is right, so all copies go.
    """
    df = df.reset_index(drop=True).copy()
    emb = sscd_embeddings(list(df["path"]))
    df["group"] = copy_groups(emb, similarity_thresholds(df["source"]), load_config()["seed"])
    sizes = df.groupby("group")["path"].transform("size")
    n_labels = df.groupby("group")["label"].transform("nunique")
    n_sources = df.groupby("group")["source"].transform("nunique")

    report = {
        "images": len(df),
        "copy_groups": int(df.loc[sizes > 1, "group"].nunique()),
        "images_in_copy_groups": int((sizes > 1).sum()),
        "groups_across_sources": int(df.loc[n_sources > 1, "group"].nunique()),
        "groups_with_conflicting_labels": int(df.loc[n_labels > 1, "group"].nunique()),
    }
    # The leader (first image of the group) is the one we keep.
    kept = df[(n_labels == 1) & (df["group"] == df.index)]
    report["kept"] = len(kept)
    report["removed"] = len(df) - len(kept)
    return kept.drop(columns="group").reset_index(drop=True), report


def closest_pairs(df: pd.DataFrame, method: str = "sscd", n: int = 6, same_label: bool | None = None,
                  band: tuple[float, float] | None = None) -> pd.DataFrame:
    """For the notebook: each image's closest other image, by perceptual hash or by SSCD.

    method="phash": `score` = hash distance (lower = more alike).
    method="sscd":  `score` = SSCD similarity (higher = more alike).
    `band` keeps only pairs whose score lies in [low, high]; pairs are spread over the band.
    Used to show why the hash fails on these datasets and what SSCD treats as a copy.
    """
    df = df.reset_index(drop=True)
    if method == "phash":
        variants = np.array([perceptual_hashes(p) for p in df["path"]], dtype=np.uint64)
        score = orientation_free_distance(variants, variants[:, 0]).astype(float)
        np.fill_diagonal(score, np.inf)
        nn = score.argmin(axis=1)
    else:
        emb = sscd_embeddings(list(df["path"]))
        score = emb @ emb.T
        np.fill_diagonal(score, -np.inf)
        nn = score.argmax(axis=1)
    out = pd.DataFrame({"path": df["path"], "label": df["label"], "source": df["source"],
                        "nn_path": df["path"][nn].to_numpy(), "nn_label": df["label"][nn].to_numpy(),
                        "score": score[np.arange(len(df)), nn].round(3)})
    if same_label is not None:
        out = out[(out["label"] == out["nn_label"]) == same_label]
    if band is not None:
        out = out[out["score"].between(*band)]
    out = out.sort_values("score").reset_index(drop=True)
    return out.iloc[np.linspace(0, len(out) - 1, min(n, len(out))).astype(int)].reset_index(drop=True)


def copy_group_example(df: pd.DataFrame, min_size: int = 8) -> list[Path]:
    """Paths of one SSCD copy group (an original photo and its augmented copies), for the notebook."""
    df = df.reset_index(drop=True)
    emb = sscd_embeddings(list(df["path"]))
    groups = copy_groups(emb, similarity_thresholds(df["source"]), load_config()["seed"])
    sizes = pd.Series(groups).value_counts()
    g = sizes[sizes >= min_size].index[len(sizes[sizes >= min_size]) // 2]  # a typical big group
    return df["path"][groups == g].tolist()[:12]


# ------------------------------------------------------- resize + split --

def export_resized(df: pd.DataFrame, out_dir: Path, size: int) -> pd.DataFrame:
    """Save every image as a size x size RGB JPEG into out_dir/<class>/<source>__<name>.jpg.

    "Stretch" resize (no cropping), the same as the Roboflow export already uses, so all
    sources look alike to the model. The source name in the file name keeps track of where
    each image came from after splitfolders has moved it around.
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)
    names = []
    for row in df.itertuples(index=False):
        dst = out_dir / row.label
        dst.mkdir(parents=True, exist_ok=True)
        name = f"{row.source}__{row.subtype.replace(' ', '-')}__{Path(row.path).stem}.jpg"
        with Image.open(row.path) as im:
            ImageOps.exif_transpose(im).convert("RGB").resize((size, size), Image.BILINEAR).save(dst / name, quality=92)
        names.append(name)
    return df.assign(file=names)


def split_folders(all_dir: Path, split_dir: Path, seed: int) -> None:
    """70/10/20 split with `splitfolders` (course tool), separately inside each class folder."""
    import splitfolders

    r = load_config()["images"]["split"]
    if split_dir.exists():
        shutil.rmtree(split_dir)
    splitfolders.ratio(str(all_dir), output=str(split_dir), seed=seed,
                       ratio=(r["train"], r["val"], r["test"]), move=False)


def read_split(split_dir: Path) -> pd.DataFrame:
    """Which file ended up in which split (from the folders splitfolders created)."""
    rows = [{"file": p.name, "split": p.parts[-3], "label": p.parts[-2]}
            for p in split_dir.glob("*/*/*.jpg")]
    return pd.DataFrame(rows)


def remove_cross_split_copies(manifest: pd.DataFrame, base: Path) -> tuple[pd.DataFrame, int]:
    """Final safety check after splitting: any val/test image that is still a copy of a train
    image (same SSCD rule) is removed from val/test. Returns (manifest, number removed).

    Leader clustering keeps one image per group, but two kept leaders can still be just above
    the threshold for each other; this step makes sure none of those pairs crosses splits.
    """
    emb = sscd_embeddings([Path(p) for p in manifest["raw_path"]])
    thr = similarity_thresholds(manifest["source"])
    is_train = (manifest["split"] == "train").to_numpy()
    sims = emb[~is_train] @ emb[is_train].T
    limit = np.minimum(thr[~is_train][:, None], thr[is_train][None, :])
    leaking = np.zeros(len(manifest), dtype=bool)
    leaking[np.where(~is_train)[0]] = (sims >= limit).any(axis=1)
    for rel in manifest.loc[leaking, "path"]:
        (base.parent / rel).unlink(missing_ok=True)
    return manifest[~leaking].reset_index(drop=True), int(leaking.sum())


# --------------------------------------------------------------------- plots --

def plot_sample_grid(manifest: pd.DataFrame, domain: str, per_group: int = 4, group_col: str = "source"):
    """Grid of processed images: one row per (class, source or fruit), `per_group` images each."""
    import matplotlib.pyplot as plt

    from src.plots import apply_style, save

    apply_style()
    seed = load_config()["seed"]
    groups = manifest.groupby(["label", group_col])
    fig, axes = plt.subplots(groups.ngroups, per_group, figsize=(per_group * 1.9, groups.ngroups * 2.1))
    for r, ((label, group), g) in enumerate(groups):
        picks = g.sample(min(per_group, len(g)), random_state=seed)
        for c in range(per_group):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(picks):
                ax.imshow(Image.open(path("data_processed", "images") / picks["path"].iloc[c]))
            if c == 0:
                ax.set_title(f"{label} · {group}", loc="left", fontsize=9)
    fig.suptitle(f"{domain}: processed 224×224 samples", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, f"image_samples_{domain}")
    return fig


def plot_pairs(pairs: pd.DataFrame, title: str, filename: str, score_name: str = "similarity"):
    """Show image pairs side by side with their labels and their distance/similarity score."""
    import matplotlib.pyplot as plt

    from src.plots import apply_style, save

    apply_style()
    fig, axes = plt.subplots(len(pairs), 2, figsize=(4.4, 2.3 * len(pairs)))
    axes = np.atleast_2d(axes)
    for r, row in enumerate(pairs.itertuples()):
        for c, (p, lab) in enumerate([(row.path, row.label), (row.nn_path, row.nn_label)]):
            axes[r, c].imshow(Image.open(p).convert("RGB"))
            axes[r, c].axis("off")
            axes[r, c].set_title(f"{lab}" + (f"   ({score_name} {row.score})" if c == 0 else ""), loc="left", fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, filename)
    return fig


def plot_images(paths: list[Path], title: str, filename: str, cols: int = 6):
    """A simple grid of images (e.g. one copy group)."""
    import matplotlib.pyplot as plt

    from src.plots import apply_style, save

    apply_style()
    rows = int(np.ceil(len(paths) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
    for ax, p in zip(np.ravel(axes), list(paths) + [None] * (rows * cols - len(paths))):
        ax.axis("off")
        if p is not None:
            ax.imshow(Image.open(p).convert("RGB"))
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, filename)
    return fig


# ---------------------------------------------------------------------- main --

def build_image_dataset(domain: str, verbose: bool = True) -> dict:
    """Run the whole pipeline for one domain and save images + manifest + count tables."""
    cfg = load_config()
    size, seed = cfg["images"]["size"], cfg["seed"]

    raw = index_domain(domain)
    readable, broken = check_readable(raw)
    deduped, dup_report = remove_near_duplicates(readable)

    # Resized images go to a temporary folder in data/interim first, then splitfolders copies
    # them into data/processed/images/<domain>/{train,val,test}/.
    staging = path("data_interim", "images_resized", domain)
    base = path("data_processed", "images", domain)
    exported = export_resized(deduped, staging, size)
    split_folders(staging, base, seed)
    shutil.rmtree(staging)

    manifest = exported.rename(columns={"path": "raw_path"}).merge(read_split(base), on=["file", "label"])
    manifest["path"] = [str(Path(domain) / r.split / r.label / r.file) for r in manifest.itertuples()]
    manifest, leaks = remove_cross_split_copies(manifest, base)
    manifest.to_csv(path("data_processed", "images") / f"{domain}_manifest.csv", index=False)

    result = {
        "raw_counts": pd.crosstab(raw["source"], raw["label"], margins=True, margins_name="total"),
        "broken": broken,
        "duplicate_report": dup_report,
        "split_counts": pd.crosstab([manifest["source"], manifest["label"]], manifest["split"],
                                    margins=True, margins_name="total")[["train", "val", "test", "total"]],
        "subtype_counts": pd.crosstab(manifest["subtype"], manifest["label"]),
        "cross_split_copies_removed": leaks,
        "manifest": manifest,
    }
    reports = path("results", create=True)
    result["split_counts"].to_csv(reports / f"image_data_{domain}_split_counts.csv")
    pd.Series({**dup_report, "broken_files": len(broken), "cross_split_copies_removed": leaks}).to_csv(
        reports / f"image_data_{domain}_quality.csv", header=["value"])
    if verbose:
        print(f"\n===== {domain} =====")
        print(result["raw_counts"], f"\nbroken files: {len(broken)}", f"\nduplicates: {dup_report}",
              f"\nval/test copies of train images removed after split: {leaks}", "\n", result["split_counts"], sep="\n")
    return result


if __name__ == "__main__":
    import sys

    for d in sys.argv[1:] or ["packages", "produce"]:
        build_image_dataset(d)
