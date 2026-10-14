"""Command-line tool: flag one order or a whole CSV, without the website.

One order:
    python scripts/flag.py "The box arrived crushed, I want a refund" --category electronics --image photo.jpg

Many orders (CSV with order_id, category, review_text; photos optional, named by order id):
    python scripts/flag.py --csv demo_data/orders.csv --images demo_data/images --out flagged.csv
    python scripts/flag.py --csv orders.csv --images photos.zip

Needs only the slim app requirements (requirements-app.txt).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.batch import (CATEGORIES, InputError, open_image, process_orders, read_images_folder,  # noqa: E402
                       read_images_zip, read_orders_csv, rows_to_csv)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("review", nargs="?", help="review text of ONE order (in quotes)")
    ap.add_argument("--category", choices=CATEGORIES, default="other", help="product category of the order")
    ap.add_argument("--image", help="customer photo for ONE order")
    ap.add_argument("--csv", help="CSV with many orders: order_id, category, review_text")
    ap.add_argument("--images", help="folder or ZIP with photos named by order id (for --csv)")
    ap.add_argument("--out", help="where to write the results CSV (default: print a summary)")
    args = ap.parse_args()

    from src.inference import Predictor

    try:
        if args.csv:
            orders = read_orders_csv(Path(args.csv).read_bytes())
            images = {}
            if args.images:
                p = Path(args.images)
                ids = [o.order_id for o in orders]
                images = read_images_zip(p.read_bytes(), ids) if p.suffix == ".zip" else read_images_folder(p, ids)
            rows = process_orders(orders, images, Predictor())
            if args.out:
                Path(args.out).write_text(rows_to_csv(rows), encoding="utf-8")
                print(f"wrote {len(rows)} results to {args.out}")
            for r in rows:
                print(f"{r['order_id']:>10}  {r['status']:<7} {r['main_reason']}")
        elif args.review:
            from src.fusion import flag_order

            img = open_image(Path(args.image).read_bytes(), args.image) if args.image else None
            print(json.dumps(flag_order(args.review, args.category, img), indent=2, default=str))
        else:
            ap.error("give a review text, or --csv for many orders")
    except InputError as e:
        sys.exit(f"Input problem: {e}")


if __name__ == "__main__":
    main()
