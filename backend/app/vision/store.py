"""Bước cuối: Lưu ảnh, kết quả và giao dịch.

SQLite + thư mục ảnh. Ảnh gốc được giữ lại để (a) đối chiếu khi công nhân khiếu
nại, (b) làm dữ liệu gắn nhãn lại cho vòng huấn luyện sau — đề bài yêu cầu cập
nhật mô hình liên tục khi món ăn thay đổi.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "storage"
IMG_DIR = ROOT / "images"
DB_PATH = ROOT / "transactions.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
    id            TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    canteen_id    TEXT,
    shift         TEXT,
    worker_id     TEXT,
    total_vnd     INTEGER NOT NULL,
    n_items       INTEGER NOT NULL,
    decision      TEXT NOT NULL,
    confirmed     INTEGER NOT NULL DEFAULT 0,
    min_conf      REAL,
    image_path    TEXT,
    items_json    TEXT NOT NULL,
    flags_json    TEXT NOT NULL,
    latency_ms    INTEGER
);
CREATE INDEX IF NOT EXISTS idx_tx_created ON transactions(created_at);
"""


def _conn() -> sqlite3.Connection:
    ROOT.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    return c


def save(result: dict, bgr: np.ndarray, *, canteen_id: str, shift: str,
         worker_id: str | None) -> str:
    tx_id = uuid.uuid4().hex[:12]
    now = datetime.now()
    day_dir = IMG_DIR / now.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    img_path = day_dir / f"{tx_id}.jpg"
    cv2.imwrite(str(img_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])

    with _conn() as c:
        c.execute(
            "INSERT INTO transactions (id, created_at, canteen_id, shift, worker_id,"
            " total_vnd, n_items, decision, confirmed, min_conf, image_path,"
            " items_json, flags_json, latency_ms)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                tx_id, now.isoformat(timespec="seconds"), canteen_id, shift, worker_id,
                result["total"], len(result["items"]),
                result["integrity"]["decision"], 0,
                result["integrity"]["min_confidence"],
                str(img_path.relative_to(ROOT)),
                json.dumps(result["items"], ensure_ascii=False),
                json.dumps(result["integrity"]["flags"], ensure_ascii=False),
                result.get("latency_ms"),
            ),
        )
    return tx_id


def confirm(tx_id: str, total_vnd: int | None = None) -> bool:
    with _conn() as c:
        if total_vnd is None:
            cur = c.execute("UPDATE transactions SET confirmed=1 WHERE id=?", (tx_id,))
        else:
            cur = c.execute(
                "UPDATE transactions SET confirmed=1, total_vnd=? WHERE id=?",
                (total_vnd, tx_id),
            )
        return cur.rowcount > 0


def recent(limit: int = 20) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT id, created_at, canteen_id, shift, worker_id, total_vnd, n_items,"
            " decision, confirmed, min_conf, latency_ms FROM transactions"
            " ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def stats() -> dict:
    with _conn() as c:
        r = c.execute(
            "SELECT COUNT(*) n, COALESCE(SUM(total_vnd),0) revenue,"
            " COALESCE(AVG(total_vnd),0) avg_tray,"
            " COALESCE(SUM(decision='need_confirm'),0) need_confirm,"
            " COALESCE(AVG(latency_ms),0) avg_latency FROM transactions"
        ).fetchone()
    return {
        "transactions": r["n"],
        "revenue": int(r["revenue"]),
        "avg_tray": int(r["avg_tray"]),
        "need_confirm": int(r["need_confirm"]),
        "auto_rate": round(1 - r["need_confirm"] / r["n"], 3) if r["n"] else None,
        "avg_latency_ms": int(r["avg_latency"]),
    }
