"""Pack the processed images (+ manifests) into images.zip for Google Colab.

Usage:  python scripts/make_images_zip.py
Then upload images.zip to Google Drive as MyDrive/rrf/images.zip.
Only data/processed/images/ is included (224x224 JPEGs, ~45 MB), never raw data or keys.
"""

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import ROOT, path  # noqa: E402

src = path("data_processed", "images")
out = ROOT / "images.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:  # JPEGs are already compressed
    for f in sorted(src.rglob("*")):
        if f.is_file():
            z.write(f, f.relative_to(ROOT))
print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) -> upload it to Google Drive as MyDrive/rrf/images.zip")
