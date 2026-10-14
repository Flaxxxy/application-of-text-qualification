"""Build demo_data/: hand-picked orders that show every fusion rule, for the video and the website.

Usage:  python scripts/make_demo_data.py

Reviews are hand-picked from the TEXT TEST split (short, clear, model is confident). Photos are
picked automatically from the IMAGE TEST splits: for each needed class, the test photo the
deployed model is most confident about (so the live demo behaves as described). Like the
simulated orders, each review/photo pair is artificial.

Writes:
    demo_data/orders.csv        order_id, category, review_text, image (empty = no photo), what it shows
    demo_data/images/<id>.jpg   the photos, named by order_id (also the format the Batch page accepts as a ZIP)
    demo_data/expected.csv      the decision the deployed models give for each order
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import ROOT, path  # noqa: E402

# (order_id, category, review_id from text_test.csv, photo needed, what the order demonstrates)
ORDERS = [
    ("DEMO-01", "electronics", "amazon2023_electronics_89936", ("packages", "damaged"), "negative review + damaged box -> URGENT"),
    ("DEMO-02", "grocery", "amazon2023_grocery_255379", ("produce", "rotten"), "negative review + rotten fruit -> URGENT"),
    ("DEMO-03", "electronics", "amazon2023_electronics_122142", ("packages", "damaged"), "positive review but damaged box -> REVIEW"),
    ("DEMO-04", "grocery", "amazon2023_grocery_99811", ("produce", "rotten"), "'fresh, well packaged' but rotten photo -> REVIEW"),
    ("DEMO-05", "electronics", "amazon2023_electronics_222544", ("packages", "intact"), "claims it came broken, box looks intact -> REVIEW (possible false claim)"),
    ("DEMO-06", "grocery", "amazon2023_grocery_257201", ("produce", "fresh"), "claims stale/hard fruit, photo looks fresh -> REVIEW (possible false claim)"),
    ("DEMO-07", "electronics", "amazon2023_electronics_179942", ("packages", "intact"), "stopped working, box fine -> REVIEW (problem a photo can't show)"),
    ("DEMO-08", "electronics", "amazon2023_electronics_104112", ("packages", "intact"), "happy customer, intact box -> OK"),
    ("DEMO-09", "grocery", "amazon2023_grocery_58435", ("produce", "fresh"), "happy customer, fresh fruit -> OK"),
    ("DEMO-10", "other", "flipkart_76081", None, "angry review, no photo -> text-only decision"),
    ("DEMO-11", "grocery", "amazon2023_grocery_226277", None, "happy review, no photo -> OK (text only)"),
    ("DEMO-12", "grocery", "amazon2023_grocery_156382", ("produce", "rotten"), "neutral review, rotten photo -> REVIEW"),
]


def pick_photos(needs: list[tuple[str, str]]) -> list[Path]:
    """Most confident test photo of each requested class, never the same photo twice.

    Packages photos come only from BoxCondition (real cardboard photos, CC BY 4.0).
    """
    chosen, used = [], set()
    for domain, label in needs:
        pred = pd.read_csv(path("results", "predictions") / f"image_{domain}_mobilenetv2_test.csv")
        pool = pred[pred["label"] == label]
        if domain == "packages":
            pool = pool[pool["source"] == "packages_boxcondition"]
        bad = label in ("damaged", "rotten")
        pool = pool.sort_values("p_bad", ascending=not bad)
        p = next(x for x in pool["path"] if x not in used)
        used.add(p)
        chosen.append(path("data_processed", "images") / p)
    return chosen


def main() -> None:
    reviews = pd.read_csv(path("data_processed") / "text_test.csv").set_index("review_id")
    out = ROOT / "demo_data"
    if (out / "images").exists():
        shutil.rmtree(out / "images")
    (out / "images").mkdir(parents=True)

    photos = iter(pick_photos([o[3] for o in ORDERS if o[3]]))
    rows = []
    for order_id, category, review_id, need, shows in ORDERS:
        image = ""
        if need:
            image = f"images/{order_id}.jpg"
            shutil.copy(next(photos), out / image)
        rows.append({"order_id": order_id, "category": category, "review_text": reviews.loc[review_id, "text"],
                     "image": image, "shows": shows, "source_review_id": review_id})
    orders = pd.DataFrame(rows)
    orders.to_csv(out / "orders.csv", index=False)

    from PIL import Image

    from src.fusion import flag_order
    from src.inference import Predictor

    predictor = Predictor()
    expected = []
    for r in orders.itertuples():
        img = Image.open(out / r.image) if r.image else None
        d = flag_order(r.review_text, r.category, img, predictor)
        expected.append({"order_id": r.order_id, "status": d["status"], "risk": d["risk"],
                         "sentiment": d["text"]["label"], "tags": ", ".join(d["text"]["tags"]),
                         "photo_p_bad": (d["details"]["image"] or {}).get("p_bad", ""),
                         "reasons": " | ".join(x["text"] for x in d["reasons"]), "shows": r.shows})
    pd.DataFrame(expected).to_csv(out / "expected.csv", index=False)
    print(pd.DataFrame(expected)[["order_id", "status", "sentiment", "photo_p_bad", "shows"]].to_string(index=False))


if __name__ == "__main__":
    main()
