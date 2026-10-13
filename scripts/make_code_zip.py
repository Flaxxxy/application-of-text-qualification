"""Pack the project code (everything tracked by git, no data or secrets) into code.zip for Colab.

Usage:  python scripts/make_code_zip.py
Then upload code.zip to Google Drive as MyDrive/rrf/code.zip. The Colab notebooks use it
instead of cloning the private GitHub repo, so no GitHub token is needed.

`git archive` takes only committed files, so .env (keys) and data/ can never end up in the zip.
"""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
out = ROOT / "code.zip"
subprocess.run(["git", "archive", "--format=zip", "-o", str(out), "HEAD"], cwd=ROOT, check=True)
print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) -> upload it to Google Drive as MyDrive/rrf/code.zip")
