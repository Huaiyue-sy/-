"""视觉质检智能体 —— CLI 入口。

模块：视觉（检测模型作工具）→ 记忆（结果入库 SQLite）→
推理（自然语言查询 + 工艺文件 RAG）→ 决策（连续缺陷预警）。

用法（在项目根目录）：
    python3 src/agent-inspection/main.py --model models/resnet18_neu.pth

命令：
    /inspect <图片或目录> [机台号]   检测并入库（目录=批量，机台号缺省 1号机台）
    /query <自然语言>               例如：最近哪种缺陷最多？3号机台有没有异常趋势？
    /status                         各机台概览 + 当前预警
    /alerts                         只看连续缺陷预警
    /rag <问题>                     直接检索工艺文件
    /help /quit
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.neu import CLASS_CN

import alert
import query
import rag
import store
from detect_tool import VisionTool


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="models/resnet18_neu.pth")
    ap.add_argument("--db", default="results/agent-inspection/detections.db")
    args = ap.parse_args()

    try:
        tool = VisionTool(args.model)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    conn = store.connect(args.db)
    knowledge = rag.CraftRAG()

    print("== 视觉质检智能体 ==")
    print(f"视觉模型：{tool.model_path}（测试集准确率 {tool.test_acc}）"
          if tool.test_acc else f"视觉模型：{tool.model_path}")
    print(f"记忆库：{args.db}（当前 {store.total(conn)} 条记录）")
    print("输入 /help 查看命令。\n")

    while True:
        try:
            line = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        cmd, *rest = line.split(maxsplit=1)
        if cmd in ("/quit", "/exit"):
            break
        if cmd == "/help":
            print(__doc__)
        elif cmd == "/inspect":
            if not rest:
                print("用法：/inspect <图片或目录> [机台号]")
                continue
            parts = rest[0].split()
            target = Path(parts[0])
            machine = parts[1] if len(parts) > 1 else "1号机台"
            imgs = [target] if target.is_file() else tool.inspect_dir(target)
            if not imgs:
                print(f"{target} 下没有图片。")
                continue
            print(f"Agent> 开始检测 {len(imgs)} 张图（记入 {machine}）…")
            for p in imgs:
                cls, conf = tool.inspect(p)
                store.record(conn, machine, str(p), cls, conf)
                mark = "⚠️" if conf < 0.5 else "✔"
                print(f"  {mark} {p.name} -> {CLASS_CN.get(cls, cls)}（置信度 {conf:.3f}）")
            for a in alert.find_alerts(conn, knowledge):
                print("Agent> " + a)
            print()
        elif cmd == "/query":
            if not rest:
                print("用法：/query <自然语言问题>")
                continue
            print("\nAgent> " + query.answer(rest[0], conn, knowledge) + "\n")
        elif cmd == "/status":
            print("\nAgent> " + alert.status_report(conn, knowledge) + "\n")
        elif cmd == "/alerts":
            found = alert.find_alerts(conn, knowledge)
            print("\nAgent> " + ("\n\n".join(found) if found else "当前无预警。") + "\n")
        elif cmd == "/rag":
            if not rest:
                print("用法：/rag <问题>，例如：/rag 划伤和轧制速度有什么关系？")
                continue
            hits = knowledge.retrieve(rest[0], top_k=2)
            if not hits:
                print("未检索到相关工艺文件。")
                continue
            for f, score in hits:
                print(f"\n《{f.stem}》（相似度 {score:.2f}）：\n{knowledge.snippet(f, 350)}")
            print()
        else:
            print("未知命令，输入 /help 查看用法。")
    conn.close()


if __name__ == "__main__":
    main()
