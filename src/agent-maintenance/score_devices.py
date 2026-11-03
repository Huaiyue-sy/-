"""对轴承特征表做设备健康打分，输出看板数据。

用法（项目根目录）：
    .venv/bin/python src/agent-maintenance/score_devices.py
    .venv/bin/python src/agent-maintenance/score_devices.py --top 15

输出：
    results/health_scores.csv    每个文件一行的评分、分项得分与告警
    results/health_devices.json  每台设备的最新状态、趋势与告警摘要
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.health import build_device_summary, score_dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="results/week1/features.csv")
    parser.add_argument("--out", default="results/health_scores.csv")
    parser.add_argument("--summary", default="results/health_devices.json")
    parser.add_argument("--top", type=int, default=0, help="只打印前 N 台设备；0 表示全部")
    args = parser.parse_args()

    features_path = Path(args.features)
    if not features_path.exists():
        raise SystemExit(f"未找到特征表：{features_path}（先运行 02_features.py）")

    df = pd.read_csv(features_path)
    scored, alerts_by_device = score_dataframe(df)
    summary = build_device_summary(scored, alerts_by_device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_path, index=False, encoding="utf-8-sig")

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"已写入 {out_path}（{len(scored)} 行）")
    print(f"已写入 {summary_path}（{len(summary['devices'])} 台设备）")
    counts = summary["summary"]
    print(
        f"健康 {counts['healthy']} · 注意 {counts['attention']} · "
        f"警告 {counts['warning']} · 危险 {counts['danger']}"
    )
    print()
    print(f"{'设备':<24}{'分数':>7}  {'等级':<4}{'阶段':<4}{'趋势':<4}{'告警':>4}")
    print("-" * 58)
    devices = summary["devices"]
    if args.top:
        devices = devices[: args.top]
    for item in devices:
        print(
            f"{item['id']:<24}{item['score']:>7.2f}  "
            f"{item['levelName']:<4}{item['stageName']:<4}{item['trend']:<4}"
            f"{item['alertCount']:>4}"
        )


if __name__ == "__main__":
    main()