"""推理模块：自然语言查询（规则意图匹配 + RAG 补充）。

支持的问题示例：
  - 最近哪种缺陷最多？ / 缺陷排行
  - 3号机台有没有异常趋势？
  - 划伤出现了多少次？
  - 各机台的缺陷分布
  - 麻点的工艺原因是什么？
"""
from __future__ import annotations

import re
import sqlite3

import store
from common.neu import CLASSES, CLASS_CN

CLASS_ALIASES = {**{c: c for c in CLASSES}, **{v: k for k, v in CLASS_CN.items()}}


def extract_machine(question: str) -> str | None:
    m = re.search(r"(\d+)\s*号\s*(机台|机器|机组|线)?", question)
    return f"{m.group(1)}号机台" if m else None


def extract_class(question: str) -> str | None:
    low = question.lower()
    for alias, cls in CLASS_ALIASES.items():
        if alias in low:
            return cls
    return None


def answer(question: str, conn: sqlite3.Connection, rag) -> str:
    q = question.strip()
    low = q.lower()

    if re.search(r"帮助|help|怎么用|支持.*问题", low):
        return __doc__ or "见 /help"

    machine = extract_machine(q)
    cls = extract_class(q)

    # 1) 类别工艺原因（RAG）
    if re.search(r"原因|为什么|怎么造成|工艺", low) and cls:
        return f"【RAG 检索】{CLASS_CN[cls]}（{cls}）的常见工艺原因：\n\n" + rag.causes_for_class(cls)

    # 2) 某类缺陷计数
    if re.search(r"多少次|几次|多少个|次数", low) and cls:
        counts = dict(store.class_counts(conn))
        n = counts.get(cls, 0)
        lines = [f"{CLASS_CN[cls]}（{cls}）累计检出 {n} 次。"]
        if n:
            lines.append(f"常见工艺原因：{rag.causes_for_class(cls)}")
        return "\n".join(lines)

    # 3) 最近哪种缺陷最多 / 排行
    if re.search(r"最多|排行|排名|top", low) or ("最近" in q and "缺陷" in q):
        scope = f"{machine} " if machine else ""
        counts = store.class_counts(conn, machine_id=machine)
        if not counts:
            return f"{scope}还没有检测记录，先用 /inspect 产线检测几张图吧。"
        total = sum(n for _, n in counts)
        head = [f"{scope}累计缺陷排行（共 {total} 条记录）："]
        head += [f"  {i+1}. {CLASS_CN.get(c, c)}（{c}）：{n} 次" for i, (c, n) in enumerate(counts)]
        top_cls = counts[0][0]
        head.append(f"\n占比最高的 {CLASS_CN.get(top_cls, top_cls)}，"
                    f"常见工艺原因：\n{rag.causes_for_class(top_cls)}")
        return "\n".join(head)

    # 4) 机台趋势/异常
    if re.search(r"趋势|异常|情况|状态", low) and machine:
        trend = store.daily_trend(conn, machine_id=machine)
        streak = store.trailing_streaks(conn).get(machine)
        if not trend:
            return f"{machine}还没有检测记录。"
        lines = [f"{machine}检测趋势（按天）："]
        for d, c, n in trend[:10]:
            lines.append(f"  {d}  {CLASS_CN.get(c, c)}：{n} 次")
        if streak:
            s_cls, s_n = streak
            lines.append(f"\n最近连续检出：{CLASS_CN.get(s_cls, s_cls)} × {s_n} 次"
                         + ("（⚠️ 触发连续缺陷预警，建议排查）" if s_n >= 3 else ""))
            if s_n >= 3:
                lines.append(f"关联工艺原因：\n{rag.causes_for_class(s_cls)}")
        return "\n".join(lines)

    # 5) 机台/整体分布
    if re.search(r"分布|概览|所有|全部|统计", low):
        lines = []
        for m in store.machines(conn):
            counts = store.class_counts(conn, machine_id=m)
            top = "、".join(f"{CLASS_CN.get(c, c)}×{n}" for c, n in counts[:3]) or "无记录"
            lines.append(f"  {m}：共 {sum(n for _, n in counts)} 条，Top：{top}")
        return "各机台缺陷分布：\n" + "\n".join(lines) if lines else "还没有任何检测记录。"

    # 6) 兜底：把问题丢给 RAG 检索工艺文件
    hits = rag.retrieve(q, top_k=1)
    if hits:
        return f"没有直接命中的统计问题。按语义检索到工艺文件《{hits[0][0].stem}》（相似度 {hits[0][1]:.2f}）：\n\n" \
            + rag.snippet(hits[0][0], 300) + "\n\n提示：可问「最近哪种缺陷最多？」「3号机台有没有异常趋势？」"
    return ("听不太懂这个问题。试试：\n"
            "  最近哪种缺陷最多？\n  3号机台有没有异常趋势？\n  划伤出现了多少次？\n  麻点的工艺原因是什么？")
