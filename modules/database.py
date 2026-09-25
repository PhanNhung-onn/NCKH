"""
modules/database.py
Lưu trữ sự kiện bất thường vào SQLite — không cần cài thêm gì, dùng stdlib.

Schema:
    events   — mỗi lần AnomalyScorer phát hiện bất thường
    sessions — mỗi lần chạy inference_pipeline

Cách dùng:
    from modules.database import DatabaseStorage
    db = DatabaseStorage("data/events.db")
    db.log_event(track_id=3, score=0.82, ts=time.time(), bbox=[x1,y1,x2,y2])
    db.close()

Query kết quả:
    python -c "from modules.database import DatabaseStorage; \\
               db=DatabaseStorage('data/events.db'); \\
               print(db.get_events(limit=20))"
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# SCHEMA SQL
# ══════════════════════════════════════════════════════════════════════

_DDL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;

CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  REAL    NOT NULL,
    ended_at    REAL,
    source      TEXT,
    model_path  TEXT,
    threshold   REAL,
    notes       TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER REFERENCES sessions(id),
    track_id    INTEGER NOT NULL,
    score       REAL    NOT NULL,
    ts          REAL    NOT NULL,               -- unix timestamp
    bbox        TEXT,                           -- JSON [x1,y1,x2,y2]
    label       TEXT    DEFAULT 'ANOMALY',
    frame_idx   INTEGER,
    extra       TEXT                            -- JSON cho dữ liệu tuỳ chỉnh
);

CREATE INDEX IF NOT EXISTS idx_events_ts       ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_track    ON events(track_id);
CREATE INDEX IF NOT EXISTS idx_events_score    ON events(score);
CREATE INDEX IF NOT EXISTS idx_events_session  ON events(session_id);
"""


# ══════════════════════════════════════════════════════════════════════
# DATABASE STORAGE
# ══════════════════════════════════════════════════════════════════════

class DatabaseStorage:
    """
    Lưu trữ sự kiện bất thường vào SQLite.
    Thread-safe: dùng lock, mỗi write được batch qua queue nhỏ.
    """

    def __init__(
        self,
        db_path:    str  = "data/events.db",
        source:     str  = "",
        model_path: str  = "",
        threshold:  float = 0.65,
        batch_size: int  = 20,         # flush mỗi N event (giảm I/O)
    ):
        self._path = db_path
        self._batch_size = batch_size
        self._lock = threading.Lock()
        self._pending: List[dict] = []

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.executescript(_DDL)
        self._conn.commit()

        # Ghi session mới
        cur = self._conn.execute(
            "INSERT INTO sessions (started_at, source, model_path, threshold) "
            "VALUES (?, ?, ?, ?)",
            (time.time(), source, model_path, threshold),
        )
        self._conn.commit()
        self._session_id = cur.lastrowid
        log.info(f"DB mở: {db_path}  session_id={self._session_id}")

    # ------------------------------------------------------------------
    def log_event(
        self,
        track_id:  int,
        score:     float,
        ts:        float,
        bbox:      List[float],
        label:     str   = "ANOMALY",
        frame_idx: Optional[int] = None,
        extra:     Optional[dict] = None,
    ):
        """
        Ghi 1 sự kiện bất thường.
        Được buffer và flush theo batch để giảm I/O disk.
        """
        row = {
            "session_id": self._session_id,
            "track_id":   track_id,
            "score":      round(score, 5),
            "ts":         round(ts, 4),
            "bbox":       json.dumps([round(v, 1) for v in bbox]),
            "label":      label,
            "frame_idx":  frame_idx,
            "extra":      json.dumps(extra) if extra else None,
        }
        with self._lock:
            self._pending.append(row)
            if len(self._pending) >= self._batch_size:
                self._flush()

    # ------------------------------------------------------------------
    def _flush(self):
        """Ghi toàn bộ pending xuống SQLite (gọi trong lock)."""
        if not self._pending:
            return
        self._conn.executemany(
            """INSERT INTO events
               (session_id, track_id, score, ts, bbox, label, frame_idx, extra)
               VALUES
               (:session_id,:track_id,:score,:ts,:bbox,:label,:frame_idx,:extra)""",
            self._pending,
        )
        self._conn.commit()
        self._pending.clear()

    # ------------------------------------------------------------------
    def flush(self):
        """Flush thủ công — gọi trước khi close hoặc muốn đảm bảo ghi ngay."""
        with self._lock:
            self._flush()

    # ------------------------------------------------------------------
    def close(self):
        """Đóng DB, ghi thời gian kết thúc session."""
        with self._lock:
            self._flush()
            self._conn.execute(
                "UPDATE sessions SET ended_at=? WHERE id=?",
                (time.time(), self._session_id),
            )
            self._conn.commit()
            self._conn.close()
        log.info(f"DB đóng: {self._path}")

    # ------------------------------------------------------------------
    # ── Truy vấn ──────────────────────────────────────────────────────

    def get_events(
        self,
        limit:       int            = 100,
        min_score:   float          = 0.0,
        track_id:    Optional[int]  = None,
        session_id:  Optional[int]  = None,
        since_ts:    Optional[float] = None,
    ) -> List[dict]:
        """Lấy danh sách sự kiện theo bộ lọc."""
        self.flush()
        conds = ["score >= ?"]
        params: list = [min_score]

        if track_id   is not None: conds.append("track_id=?");   params.append(track_id)
        if session_id is not None: conds.append("session_id=?");  params.append(session_id)
        if since_ts   is not None: conds.append("ts >= ?");       params.append(since_ts)

        where = " AND ".join(conds)
        params.append(limit)

        rows = self._conn.execute(
            f"SELECT * FROM events WHERE {where} ORDER BY ts DESC LIMIT ?",
            params,
        ).fetchall()

        cols = ["id","session_id","track_id","score","ts","bbox","label","frame_idx","extra"]
        result = []
        for r in rows:
            d = dict(zip(cols, r))
            d["bbox"]  = json.loads(d["bbox"])  if d["bbox"]  else []
            d["extra"] = json.loads(d["extra"]) if d["extra"] else {}
            result.append(d)
        return result

    def get_summary(self) -> dict:
        """Thống kê tổng hợp cho session hiện tại."""
        self.flush()
        cur = self._conn.execute(
            """SELECT
                COUNT(*)          AS total_events,
                COUNT(DISTINCT track_id) AS unique_tracks,
                MAX(score)        AS max_score,
                AVG(score)        AS avg_score,
                MIN(ts)           AS first_ts,
                MAX(ts)           AS last_ts
               FROM events WHERE session_id = ?""",
            (self._session_id,),
        ).fetchone()
        cols = ["total_events","unique_tracks","max_score","avg_score","first_ts","last_ts"]
        return dict(zip(cols, cur))

    def get_sessions(self, limit: int = 10) -> List[dict]:
        """Lấy danh sách các session gần nhất."""
        rows = self._conn.execute(
            "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        cols = ["id","started_at","ended_at","source","model_path","threshold","notes"]
        return [dict(zip(cols, r)) for r in rows]

    def export_csv(self, path: str, session_id: Optional[int] = None):
        """Xuất sự kiện ra CSV để phân tích ngoài."""
        import csv
        self.flush()
        sid = session_id or self._session_id
        rows = self._conn.execute(
            "SELECT * FROM events WHERE session_id=? ORDER BY ts",
            (sid,),
        ).fetchall()
        cols = ["id","session_id","track_id","score","ts","bbox","label","frame_idx","extra"]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerows(rows)
        log.info(f"Xuất {len(rows)} events → {path}")
        return len(rows)

    # ------------------------------------------------------------------
    @property
    def session_id(self) -> int:
        return self._session_id
