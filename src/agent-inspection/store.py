"""质检结果入库（记忆模块）：SQLite 检测台账。"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    machine_id TEXT NOT NULL,
    image_path TEXT,
    defect_class TEXT NOT NULL,
    confidence REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_machine_ts ON detections(machine_id, ts);
"""


def connect(db_path: str | Path = "results/agent-inspection/detections.db") -> sqlite3.Connection:
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def record(conn: sqlite3.Connection, machine_id: str, image_path: str | None,
           defect_class: str, confidence: float) -> int:
    cur = conn.execute(
        "INSERT INTO detections(ts, machine_id, image_path, defect_class, confidence) "
        "VALUES(?,?,?,?,?)",
        (datetime.now().isoformat(timespec="seconds"), machine_id,
         image_path, defect_class, round(float(confidence), 4)))
    conn.commit()
    return int(cur.lastrowid)


def class_counts(conn: sqlite3.Connection, machine_id: str | None = None,
                 hours: float | None = None) -> list[tuple[str, int]]:
    sql = "SELECT defect_class, COUNT(*) AS n FROM detections"
    conds, params = [], []
    if machine_id:
        conds.append("machine_id = ?")
        params.append(machine_id)
    if hours:
        conds.append("ts >= ?")
        params.append((datetime.now() - timedelta(hours=hours)).isoformat(timespec="seconds"))
    if conds:
        sql += " WHERE " + " AND ".join(conds)
    sql += " GROUP BY defect_class ORDER BY n DESC"
    return [(r["defect_class"], r["n"]) for r in conn.execute(sql, params)]


def daily_trend(conn: sqlite3.Connection, machine_id: str | None = None,
                days: int = 7) -> list[tuple[str, str, int]]:
    """按 天×缺陷类别 统计。返回 (date, defect_class, n)。"""
    sql = ("SELECT substr(ts,1,10) AS d, defect_class, COUNT(*) AS n FROM detections")
    params = []
    if machine_id:
        sql += " WHERE machine_id = ?"
        params.append(machine_id)
    sql += " GROUP BY d, defect_class ORDER BY d DESC, n DESC"
    return [(r["d"], r["defect_class"], r["n"]) for r in conn.execute(sql, params)]


def recent(conn: sqlite3.Connection, n: int = 10,
           machine_id: str | None = None) -> list[sqlite3.Row]:
    sql = "SELECT * FROM detections"
    params = []
    if machine_id:
        sql += " WHERE machine_id = ?"
        params.append(machine_id)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(n)
    return list(conn.execute(sql, params))


def machines(conn: sqlite3.Connection) -> list[str]:
    return [r["machine_id"] for r in
            conn.execute("SELECT DISTINCT machine_id FROM detections ORDER BY machine_id")]


def trailing_streaks(conn: sqlite3.Connection, machine_id: str | None = None
                     ) -> dict[str, tuple[str, int]]:
    """各机台最近连续出现的同类缺陷（连续按插入顺序，中间换类即断）。"""
    sql = "SELECT machine_id, defect_class FROM detections"
    params = []
    if machine_id:
        sql += " WHERE machine_id = ?"
        params.append(machine_id)
    sql += " ORDER BY id"
    streak: dict[str, tuple[str, int]] = {}
    for r in conn.execute(sql, params):
        m, c = r["machine_id"], r["defect_class"]
        last_cls, last_n = streak.get(m, ("", 0))
        streak[m] = (c, last_n + 1) if c == last_cls else (c, 1)
    return streak


def total(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0])
