"""第1周·第2天（上）：特征工程——时域/频域/包络谱特征提取。

遍历所有文件生成 features.csv，并按 RMS 退化曲线打健康阶段标签
（0=健康 1=退化 2=故障）和相对剩余寿命 rel_rul（1→0）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.bearing import (
    FS_HZ, BearingGeometry, condition_rpm, discover,
    health_stage_labels, rms_series,
)
from common.features import extract_channels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/XJTU-SY_Bearing_Datasets")
    ap.add_argument("--out", default="results/week1")
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--bearings", nargs="*", default=None)
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--geometry-json", default=None)
    ap.add_argument("--healthy-ratio", type=float, default=0.2)
    ap.add_argument("--degrade-mult", type=float, default=1.5)
    ap.add_argument("--failure-mult", type=float, default=3.0)
    args = ap.parse_args()

    geometry = BearingGeometry.from_json(args.geometry_json) if args.geometry_json else BearingGeometry()

    runs = discover(args.data_dir, args.conditions, args.bearings, args.max_files)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for r in runs:
        fr = condition_rpm(r.condition) / 60.0
        stage = health_stage_labels(
            rms_series(r, channel=1), args.healthy_ratio, args.degrade_mult, args.failure_mult)
        n = len(r.files)
        for i, f in enumerate(r.files):
            row = extract_channels(f, FS_HZ, geometry, fr)
            row.update({
                "condition": r.condition, "bearing": r.name, "file_idx": i + 1,
                "bearing_key": r.key, "stage": int(stage[i]),
                "rel_rul": (n - i) / n, "file_path": str(f.resolve()),
            })
            rows.append(row)
        print(f"  {r.key}: {n} 个文件，阶段分布 {np.bincount(stage, minlength=3).tolist()}")

    df = pd.DataFrame(rows)
    csv_path = out / "features.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"特征表：{csv_path}（{df.shape[0]} 行 × {df.shape[1]} 列）")
    print("阶段分布：", df["stage"].value_counts().sort_index().to_dict())

    meta = {
        "fs_hz": FS_HZ,
        "n_runs": len(runs), "total_rows": int(df.shape[0]),
        "label_rule": {
            "healthy_ratio": args.healthy_ratio,
            "degrade_mult": args.degrade_mult, "failure_mult": args.failure_mult,
        },
        "geometry": geometry.__dict__,
    }
    (out / "features_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
