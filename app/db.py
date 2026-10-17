"""SQLite log of every prediction the website makes (shown on the History page).

SQLite is a single file and needs no server, which suits a demo. On Render the file is reset on
every redeploy, which is fine for a demo (noted in the README).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT NOT NULL,
    source      TEXT NOT NULL,          -- single / batch / demo
    order_id    TEXT,
    category    TEXT,
    review_text TEXT,
    sentiment   TEXT,
    p_negative  REAL,
    tags        TEXT,
    has_photo   INTEGER,
    photo_result TEXT,
    p_bad       REAL,
    status      TEXT,
    risk        REAL,
    reasons     TEXT,
    visitor     TEXT                    -- anonymous id of the browser, so each visitor sees only their own checks
)
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


def init_db(db_path: Path) -> None:
    with connect(db_path) as con:
        con.execute(SCHEMA)
        columns = {row["name"] for row in con.execute("PRAGMA table_info(predictions)")}
        if "visitor" not in columns:  # database created by an older version of the app
            con.execute("ALTER TABLE predictions ADD COLUMN visitor TEXT")


def log_rows(db_path: Path, rows: list[dict[str, Any]], source: str, visitor: str | None = None) -> None:
    """Store result rows (as produced by src.batch.process_orders). Parameters are bound, never pasted into SQL."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with connect(db_path) as con:
        con.executemany(
            "INSERT INTO predictions (created_at, source, order_id, category, review_text, sentiment, p_negative, "
            "tags, has_photo, photo_result, p_bad, status, risk, reasons, visitor) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(now, source, r["order_id"], r["category"], r["review_text"][:1000], r["sentiment"], r["p_negative"],
              r["tags"], int(r["has_photo"]), r["photo_result"], r["p_bad"] if r["p_bad"] != "" else None,
              r["status"], r["risk"], r["reasons"], visitor) for r in rows])


def recent(db_path: Path, visitor: str | None, limit: int = 20) -> list[sqlite3.Row]:
    """The visitor's own latest checks (other visitors' checks are never shown)."""
    with connect(db_path) as con:
        return con.execute("SELECT * FROM predictions WHERE visitor = ? ORDER BY id DESC LIMIT ?",
                           (visitor, limit)).fetchall()
