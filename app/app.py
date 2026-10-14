"""Flask website: check one order, check a batch, see the history and the model results.

Run locally:   python -m app            (then open http://127.0.0.1:5000)
Production:    gunicorn app.app:app --workers 1 --timeout 120

The models are loaded ONCE when the app starts (Predictor below) and shared by all requests;
loading them per request would take seconds and use extra memory. One worker keeps a single copy
in memory, which matters on the 512 MB server.
"""

from __future__ import annotations

import base64
import csv
import io
import json
from pathlib import Path

from flask import Flask, abort, render_template, request, send_from_directory
from PIL import Image

from app.db import init_db, log_rows, recent
from src.batch import (CATEGORIES, InputError, Order, open_image, process_orders, read_images_zip, read_orders_csv,
                       rows_to_csv, validate_category, validate_text)
from src.config import ROOT, load_config, load_env, path
from src.inference import Predictor

load_env()
CFG = load_config()
DB = ROOT / CFG["app"]["db_path"]
DEMO = ROOT / "demo_data"
FIGURES = path("figures")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = CFG["app"]["max_batch_upload_mb"] * 1024 * 1024  # whole request (CSV + ZIP)
PREDICTOR = Predictor()
init_db(DB)


# ------------------------------------------------------------------ helpers --

def demo_orders() -> list[dict]:
    with open(DEMO / "orders.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def thumbnail_data_uri(img: Image.Image, size: int = 360) -> str:
    """Small copy of an uploaded photo embedded in the page (uploads are never stored on disk)."""
    img = img.convert("RGB")
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def highlight(text: str, tags: dict[str, str]) -> list[tuple[str, str | None]]:
    """Split the review into pieces so the template can highlight the phrases that triggered each tag."""
    low = text.lower()
    spans = []
    for tag, phrase in tags.items():
        i = low.find(phrase.lower())
        if i >= 0:
            spans.append((i, i + len(phrase), tag))
    spans.sort()
    pieces, pos = [], 0
    for start, end, tag in spans:
        if start < pos:
            continue  # overlapping phrases: keep the first
        pieces.append((text[pos:start], None))
        pieces.append((text[start:end], tag))
        pos = end
    pieces.append((text[pos:], None))
    return [p for p in pieces if p[0]]


def load_results() -> dict:
    """Numbers for the Home and Models pages, read from results/metrics.json (real runs only)."""
    f = path("results") / "metrics.json"
    return json.loads(f.read_text()) if f.exists() else {}


# ------------------------------------------------------------------- routes --

@app.get("/")
def home():
    m = load_results()
    stats = {
        "text_f1": m.get("deploy", {}).get("summary", {}).get("text_int8_test", {}).get("macro_f1"),
        "packages_f1": m.get("image_packages", {}).get("mobilenetv2", {}).get("test", {}).get("macro_f1"),
        "produce_f1": m.get("image_produce", {}).get("mobilenetv2", {}).get("test", {}).get("macro_f1"),
        "fusion": m.get("fusion", {}).get("simulated_orders", {}).get("comparison_test", {}),
    }
    return render_template("home.html", stats=stats)


@app.route("/check", methods=["GET", "POST"])
def check():
    demos = demo_orders()
    form = {"review_text": "", "category": "electronics", "demo": ""}
    if request.method == "GET" and request.args.get("demo"):
        d = next((o for o in demos if o["order_id"] == request.args["demo"]), None)
        if d:
            form = {"review_text": d["review_text"], "category": d["category"], "demo": d["order_id"] if d["image"] else ""}
    result, error, photo_uri = None, None, None
    if request.method == "POST":
        form = {k: request.form.get(k, "") for k in ("review_text", "category", "demo")}
        try:
            text = validate_text(form["review_text"])
            category = validate_category(form["category"])
            img = None
            upload = request.files.get("photo")
            if upload and upload.filename:
                data = upload.read()
                if len(data) > CFG["app"]["max_upload_mb"] * 1024 * 1024:
                    raise InputError(f"The photo is too large (max {CFG['app']['max_upload_mb']} MB).")
                img = open_image(data, upload.filename)
            elif form["demo"]:
                d = next((o for o in demos if o["order_id"] == form["demo"] and o["image"]), None)
                if d:
                    img = Image.open(DEMO / d["image"])
            row = process_orders([Order(form["demo"] or "web", category, text)],
                                 {form["demo"] or "web": img} if img else {}, PREDICTOR)[0]
            log_rows(DB, [row], "single")
            text_result = PREDICTOR.text(text)
            result = {"row": row, "probs": text_result["probs"], "tags": text_result["tags"],
                      "pieces": highlight(text, text_result["tags"]), "reasons": row["reasons"].split(" | "),
                      "photo_threshold": PREDICTOR.image_models[PREDICTOR.image_domain(category)].threshold}
            photo_uri = thumbnail_data_uri(img) if img else None
        except InputError as e:
            error = str(e)
    return render_template("check.html", form=form, result=result, error=error, photo_uri=photo_uri,
                           categories=CATEGORIES, demos=demos, routing=PREDICTOR.settings["routing"])


@app.route("/batch", methods=["GET", "POST"])
def batch():
    rows, error, source = None, None, None
    if request.method == "POST":
        try:
            if request.form.get("action") == "demo":
                orders = [Order(o["order_id"], o["category"], o["review_text"]) for o in demo_orders()]
                images = {o["order_id"]: Image.open(DEMO / o["image"]) for o in demo_orders() if o["image"]}
                source = "demo"
            else:
                csv_file = request.files.get("orders")
                if not csv_file or not csv_file.filename:
                    raise InputError("Please choose a CSV file with your orders.")
                orders = read_orders_csv(csv_file.read())
                zip_file = request.files.get("photos")
                images = read_images_zip(zip_file.read(), [o.order_id for o in orders]) \
                    if zip_file and zip_file.filename else {}
                source = "batch"
            rows = process_orders(orders, images, PREDICTOR)
            log_rows(DB, rows, source)
        except InputError as e:
            error = str(e)
    csv_uri = None
    if rows:
        csv_uri = "data:text/csv;base64," + base64.b64encode(rows_to_csv(rows).encode()).decode()
    counts = {s: sum(r["status"] == s for r in rows) for s in ("URGENT", "REVIEW", "OK")} if rows else {}
    return render_template("batch.html", rows=rows, error=error, csv_uri=csv_uri, counts=counts, source=source,
                           max_rows=CFG["app"]["max_batch_rows"])


@app.get("/history")
def history():
    return render_template("history.html", rows=recent(DB, CFG["app"]["history_limit"]),
                           limit=CFG["app"]["history_limit"])


def model_tables(m: dict) -> dict:
    """Rows for the Models page tables, sorted best first (plain Python, no pandas on the server)."""
    def group(name: str) -> str:
        return "ready-made" if name.startswith("bench_") else "fine-tuned" if name.endswith("_finetuned") else "ours"

    text = [{"name": n, "group": group(n), "type": v.get("type", ""), "f1": v["test"]["macro_f1"],
             "neutral": v["test"]["per_class"]["neutral"]["f1"], "external": v.get("external", {}).get("macro_f1"),
             "ms": v.get("ms_per_review"), "mb": v.get("size_mb")} for n, v in m.get("text", {}).items()]
    images = {}
    for domain in ("packages", "produce"):
        rows = []
        for n, v in m.get(f"image_{domain}", {}).items():
            bad = v["labels"][1]
            rows.append({"name": n, "type": v.get("type", ""), "f1": v["test"]["macro_f1"],
                         "recall": v["test"]["per_class"][bad]["recall"],
                         "precision": v["test"]["per_class"][bad]["precision"], "threshold": v["threshold"],
                         "groups": v.get("test_per_group", {})})
        images[domain] = {"bad": {"packages": "damaged", "produce": "rotten"}[domain],
                          "rows": sorted(rows, key=lambda r: -r["f1"])}
    fusion = m.get("fusion", {}).get("simulated_orders", {})
    return {"text": sorted(text, key=lambda r: -r["f1"]), "images": images, "fusion": fusion,
            "deploy": m.get("deploy", {}).get("summary", {})}


@app.get("/models")
def models():
    figures = {f.stem for f in FIGURES.glob("*.png")}
    return render_template("models.html", t=model_tables(load_results()), figures=figures)


@app.get("/figures/<path:name>")
def figure(name: str):
    if not name.endswith(".png"):
        abort(404)
    return send_from_directory(FIGURES, name)  # send_from_directory refuses paths outside the folder


@app.get("/demo-files/<path:name>")
def demo_file(name: str):
    if name not in ("orders.csv", "photos.zip") and not name.startswith("images/"):
        abort(404)
    if name == "photos.zip":
        import zipfile

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            for p in sorted((DEMO / "images").glob("*.jpg")):
                z.write(p, p.name)
        buf.seek(0)
        from flask import send_file

        return send_file(buf, mimetype="application/zip", as_attachment=True, download_name="demo_photos.zip")
    return send_from_directory(DEMO, name, as_attachment=name == "orders.csv")


@app.get("/health")
def health():
    return {"status": "ok", "models": ["text"] + sorted(PREDICTOR.image_models)}


@app.errorhandler(413)
def too_large(_):
    return render_template("error.html", message=f"The upload is too large (max {CFG['app']['max_batch_upload_mb']} MB "
                                                 "in total). Try fewer or smaller photos."), 413


@app.errorhandler(404)
def not_found(_):
    return render_template("error.html", message="This page doesn't exist."), 404


@app.errorhandler(500)
def server_error(_):
    return render_template("error.html", message="Something went wrong on our side. Please try again."), 500
