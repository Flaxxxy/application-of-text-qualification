"""Read, check and process many orders at once (CSV + optional photos). Used by the website and the CLI.

CSV format (header required):  order_id, category, review_text
Photos (optional): a ZIP file or a folder with images named by order id, e.g. A-1001.jpg.

Deploy-safe: only the standard library, Pillow and the deployed models (no pandas).
"""

from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, UnidentifiedImageError

from src.config import load_config

REQUIRED_COLUMNS = ("order_id", "category", "review_text")
CATEGORIES = ("electronics", "grocery", "other")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")
MAX_TEXT_CHARS = 5000


class InputError(ValueError):
    """A problem with the user's input, with a message that can be shown as is."""


@dataclass
class Order:
    order_id: str
    category: str
    review_text: str


def validate_text(text: str) -> str:
    text = (text or "").strip()
    if len(text.split()) < 3:
        raise InputError("Please enter a review of at least 3 words.")
    if len(text) > MAX_TEXT_CHARS:
        raise InputError(f"The review is too long (max {MAX_TEXT_CHARS} characters).")
    return text


def validate_category(category: str) -> str:
    category = (category or "").strip().lower()
    if category not in CATEGORIES:
        raise InputError(f"Unknown category '{category}'. Use one of: {', '.join(CATEGORIES)}.")
    return category


def open_image(data: bytes, name: str = "photo") -> Image.Image:
    """Open uploaded image bytes safely; raise InputError with a friendly message if it isn't a photo."""
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()                      # cheap check for corrupt files
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise InputError(f"'{name}' is not a readable image (use JPG, PNG or WEBP).") from None
    return img


def read_orders_csv(data: bytes | str) -> list[Order]:
    """Parse and validate an orders CSV. All problems are collected and reported together."""
    text = data.decode("utf-8-sig", errors="replace") if isinstance(data, bytes) else data
    reader = csv.DictReader(io.StringIO(text))
    header = [h.strip().lower() for h in (reader.fieldnames or [])]
    missing = [c for c in REQUIRED_COLUMNS if c not in header]
    if missing:
        raise InputError(f"The CSV needs the columns {', '.join(REQUIRED_COLUMNS)}; missing: {', '.join(missing)}.")
    max_rows = load_config()["app"]["max_batch_rows"]
    orders, problems, seen = [], [], set()
    for line, row in enumerate(reader, start=2):
        row = {k.strip().lower(): (v or "") for k, v in row.items() if k}
        if not any(row.values()):
            continue
        try:
            order_id = row["order_id"].strip()
            if not order_id:
                raise InputError("empty order_id")
            if order_id in seen:
                raise InputError(f"duplicate order_id '{order_id}'")
            seen.add(order_id)
            orders.append(Order(order_id, validate_category(row["category"]), validate_text(row["review_text"])))
        except InputError as e:
            problems.append(f"line {line}: {e}")
        if len(orders) + len(problems) > max_rows:
            raise InputError(f"Too many rows: the limit is {max_rows} orders per upload.")
    if problems:
        raise InputError("Please fix the CSV: " + "; ".join(problems[:10]) + (" …" if len(problems) > 10 else ""))
    if not orders:
        raise InputError("The CSV has no orders.")
    return orders


def read_images_zip(data: bytes, order_ids: Iterable[str]) -> dict[str, Image.Image]:
    """Photos from a ZIP, matched to orders by file name (A-1001.jpg -> order A-1001).

    Files are read in memory only (never extracted to disk), so a malicious ZIP can't write files.
    Files that don't match an order id are ignored.
    """
    wanted = set(order_ids)
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise InputError("The photo file is not a valid ZIP.") from None
    images = {}
    for info in zf.infolist():
        p = Path(info.filename)
        if info.is_dir() or p.name.startswith(".") or p.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if p.stem in wanted:
            images[p.stem] = open_image(zf.read(info), p.name)
    return images


def read_images_folder(folder: Path, order_ids: Iterable[str]) -> dict[str, Image.Image]:
    """Same as read_images_zip, for a folder on disk (used by the CLI)."""
    wanted = set(order_ids)
    return {p.stem: open_image(p.read_bytes(), p.name) for p in sorted(Path(folder).iterdir())
            if p.suffix.lower() in IMAGE_SUFFIXES and p.stem in wanted}


def process_orders(orders: list[Order], images: dict[str, Image.Image], predictor) -> list[dict[str, Any]]:
    """Run text + photo models and the fusion for every order; flat rows ready for a table or CSV."""
    from src.fusion import flag_order

    rows = []
    for o in orders:
        d = flag_order(o.review_text, o.category, images.get(o.order_id), predictor)
        img = d["details"]["image"]
        rows.append({
            "order_id": o.order_id, "category": o.category, "review_text": o.review_text,
            "status": d["status"], "risk": d["risk"], "sentiment": d["text"]["label"],
            "p_negative": d["text"]["probs"]["negative"], "p_neutral": d["text"]["probs"]["neutral"],
            "p_positive": d["text"]["probs"]["positive"], "tags": ", ".join(d["text"]["tags"]),
            "has_photo": img is not None, "photo_model": img["domain"] if img else "",
            "photo_result": img["label"] if img else "", "p_bad": img["p_bad"] if img else "",
            "reasons": " | ".join(r["text"] for r in d["reasons"]), "main_reason": d["reasons"][0]["text"],
        })
    return rows


def rows_to_csv(rows: list[dict[str, Any]]) -> str:
    out = io.StringIO()
    if rows:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return out.getvalue()
