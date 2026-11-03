"""设备运维诊断智能体 —— 对话界面入口。

模拟「工人提问 → Agent 诊断」场景的 CLI 交互。

用法（在项目根目录）：
    python3 src/agent-maintenance/main.py --features results/week1/features.csv

命令：
    /scan <csv文件> [工况]     对单个振动文件执行完整诊断
    /report <csv文件> [工况]   生成结构化诊断报告（大模型或离线降级）
    /ask <问题>                结合最近一次诊断结果自由提问（大模型）
    /score <轴承编号>          查看设备健康分、等级、趋势与最近告警
    /help /quit
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.bearing import BearingGeometry
from common.confirm import confirm_offline
from common.health import LEVEL_NAMES, STAGE_NAMES, score_bearing

from diagnose import MaintenanceAgent
from llm import DeepSeekClient, llm_backend_name
from report import build_report, diagnosis_facts

ASK_SYSTEM_PROMPT = (
    "你是设备运维诊断智能体的问答模块。请结合给定的最近诊断事实，"
    "用中文简洁回答工人的问题；事实里没有的信息要明说不知道，不要编造。"
)


def parse_condition(arg: str | None, agent: MaintenanceAgent,
                    csv_path: str | None = None, default: str = "35Hz12kN") -> str:
    """工况参数可省略：优先从文件路径识别，否则取特征表里最常见的工况。"""
    if arg:
        return arg
    if csv_path:
        m = re.search(r"(\d+(?:\.\d+)?Hz\d+(?:\.\d+)?kN)", str(csv_path))
        if m:
            return m.group(1)
    try:
        return agent.df["condition"].mode().iloc[0]
    except Exception:
        return default


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="results/week1/features.csv")
    ap.add_argument("--classifier", default="models/week1_classifier.joblib")
    ap.add_argument("--rul", default="models/week1_rul.joblib")
    ap.add_argument("--contamination", type=float, default=0.05)
    args = ap.parse_args()

    llm = DeepSeekClient()
    print("== 设备运维诊断智能体 ==")
    print(f"大模型后端：{llm_backend_name(llm)}")
    agent = MaintenanceAgent(args.features, args.classifier, args.rul, args.contamination)
    n_cls = "已加载" if agent.classifier is not None else "未找到（先跑 04_classify.py）"
    n_rul = "已加载" if agent.rul_model is not None else "未找到（先跑 05_rul_regression.py）"
    print(f"分类模型：{n_cls}；RUL 模型：{n_rul}")
    print("输入 /help 查看命令。\n")

    last_diagnosis = None
    last_condition = None
    while True:
        try:
            line = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        cmd, *rest = line.split()
        if cmd in ("/quit", "/exit"):
            break
        if cmd == "/help":
            print(__doc__)
        elif cmd == "/scan":
            if not rest:
                print("用法：/scan <csv文件> [工况，如 35Hz12kN]")
                continue
            cond = parse_condition(rest[1] if len(rest) > 1 else None, agent, rest[0])
            diag = agent.diagnose_file(rest[0], cond, BearingGeometry())
            last_diagnosis, last_condition = diag, cond
            print(f"\nAgent> 诊断完成：")
            print(f"  异常判定：{'⚠️ 异常' if diag.is_anomaly else '正常'}"
                  f"（异常分 {diag.anomaly_score:.4f}）")
            if diag.is_anomaly:
                print(f"  健康阶段：{diag.stage_name}")
                print(f"  疑似故障元件：{diag.fault_location}")
                print(f"  证据：{diag.fault_evidence}")
                if diag.rel_rul is not None:
                    print(f"  {diag.rul_note}")
            print("  输入 /ask 继续追问，或 /report <csv> [工况] 生成完整报告。\n")
        elif cmd == "/report":
            if not rest:
                print("用法：/report <csv文件> [工况]")
                continue
            cond = parse_condition(rest[1] if len(rest) > 1 else None, agent, rest[0])
            diag = agent.diagnose_file(rest[0], cond, BearingGeometry())
            last_diagnosis, last_condition = diag, cond
            print("\nAgent> " + build_report(llm, diag, cond) + "\n")
        elif cmd == "/score":
            if not rest:
                print("用法：/score <轴承编号>，例如：/score Bearing1_1")
                continue
            points = score_bearing(agent.df, " ".join(rest))
            if not points:
                available = sorted(agent.df["bearing"].astype(str).unique())[:8]
                print("未找到该轴承。可用示例：" + "、".join(available) + "\n")
                continue
            latest = points[-1]
            print(f"\nAgent> 设备 {latest.bearing_key}（第 {latest.file_idx} 个文件，共 {len(points)} 个）")
            print(f"  健康分：{latest.score:.2f} / 100（{LEVEL_NAMES[latest.level]}）")
            print(f"  阶段：{STAGE_NAMES.get(latest.stage, '未知')}；相对 RUL：{latest.rel_rul:.2f}；趋势：{latest.trend}")
            print(f"  评分依据：{latest.explanation}")
            recent = [p for p in points if p.alert_message][-5:]
            if recent:
                print("  最近告警：")
                for point in recent:
                    print(f"    · 文件 {point.file_idx}（{point.level_name}）：{point.alert_message}")
            print()
        elif cmd == "/ask":
            if not rest:
                print("用法：/ask <问题>，例如：/ask 还能坚持多久？")
                continue
            question = " ".join(rest)
            if last_diagnosis is None:
                print("请先 /scan <csv文件>，再提问。\n")
                continue
            facts = diagnosis_facts(last_diagnosis, last_condition)
            reply = llm.chat(ASK_SYSTEM_PROMPT,
                             f"最近诊断事实：\n{facts}\n\n工人提问：{question}")
            if reply:
                print("\nAgent> " + reply + "\n")
            elif confirm_offline("自由问答需要在线大模型"):
                print("\nAgent> （离线模式不支持自由问答；可参考最近诊断事实：）\n" + facts + "\n")
            else:
                print("\nAgent> 已取消离线回答。\n")
        else:
            print("未知命令，输入 /help 查看用法。")


if __name__ == "__main__":
    main()
