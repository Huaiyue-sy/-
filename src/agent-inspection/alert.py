"""决策模块：连续缺陷预警。

某机台同类缺陷连续出现 ≥ streak 次时触发预警，
并自动关联 RAG 工艺文件给出可能原因（对应应急规范二级响应）。
"""
from __future__ import annotations

import sqlite3

import store
from common.neu import CLASS_CN


def find_alerts(conn: sqlite3.Connection, rag, streak: int = 3) -> list[str]:
    alerts: list[str] = []
    for machine, (cls, n) in store.trailing_streaks(conn).items():
        if n < streak:
            continue
        alerts.append(
            f"⚠️ 预警（二级响应）：{machine} 连续 {n} 次检出「{CLASS_CN.get(cls, cls)}（{cls}）」。\n"
            f"疑似工艺原因：\n{rag.causes_for_class(cls)}\n"
            "请按《缺陷应急处理规范》10 分钟内复核是否误报，并按对应工艺文件排查。")
    return alerts


def status_report(conn: sqlite3.Connection, rag) -> str:
    lines = [f"检测记录总数：{store.total(conn)}"]
    streaks = store.trailing_streaks(conn)
    for m in store.machines(conn):
        counts = store.class_counts(conn, machine_id=m)
        top = "、".join(f"{CLASS_CN.get(c, c)}×{n}" for c, n in counts[:3]) or "无"
        s = streaks.get(m)
        s_txt = f"，最近连续：{CLASS_CN.get(s[0], s[0])}×{s[1]}" if s else ""
        lines.append(f"  {m}：{sum(n for _, n in counts)} 条记录，Top：{top}{s_txt}")
    alerts = find_alerts(conn, rag)
    lines.append("\n预警：" + ("\n\n".join(alerts) if alerts else "当前无预警。"))
    return "\n".join(lines)
