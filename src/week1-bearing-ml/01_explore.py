"""第1周·第1天：XJTU-SY 数据集探索。

产出数据集概览表、RMS 退化曲线、示例波形/频谱/包络谱图。
用法：
    python3 src/week1-bearing-ml/01_explore.py --data-dir data/XJTU-SY_Bearing_Datasets
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
    BearingGeometry, condition_rpm, discover, load_vibration, rms_series,
)
from common.plotting import save_fig, setup_chinese_font

import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/XJTU-SY_Bearing_Datasets")
    ap.add_argument("--out", default="results/week1")
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--bearings", nargs="*", default=None)
    ap.add_argument("--max-files", type=int, default=None, help="每个轴承最多取前 N 个文件（数据太大时用）")
    ap.add_argument("--geometry-json", default=None)
    args = ap.parse_args()

    setup_chinese_font()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    geometry = BearingGeometry.from_json(args.geometry_json) if args.geometry_json else BearingGeometry()

    runs = discover(args.data_dir, args.conditions, args.bearings, args.max_files)
    print(f"发现 {len(runs)} 个轴承运行记录：")

    rows = []
    for r in runs:
        rows.append({
            "工况": r.condition, "轴承": r.name, "文件数": len(r.files),
            "转速rpm": condition_rpm(r.condition),
            "首文件": r.files[0].name, "末文件": r.files[-1].name,
        })
    overview = pd.DataFrame(rows)
    overview.to_csv(out / "overview.csv", index=False, encoding="utf-8-sig")
    print(overview.to_string(index=False))
    print(f"概览表已保存：{out / 'overview.csv'}")

    # RMS 退化曲线（每轴承一张子图）
    show_runs = runs[:8]
    fig, axes = plt.subplots(max(1, (len(show_runs) + 1) // 2), 2, figsize=(11, 2.6 * ((len(show_runs) + 1) // 2)), squeeze=False)
    for ax, r in zip(axes.flat, show_runs):
        h = rms_series(r, channel=0)
        v = rms_series(r, channel=1)
        ax.plot(np.arange(1, len(h) + 1), h, label="水平")
        ax.plot(np.arange(1, len(v) + 1), v, label="垂直")
        ax.set_title(f"{r.key}（{len(r.files)} 个文件）", fontsize=10)
        ax.set_xlabel("文件序号")
        ax.set_ylabel("RMS")
        ax.legend(fontsize=8)
    for ax in axes.flat[len(show_runs):]:
        ax.axis("off")
    save_fig(fig, out / "rms_degradation.png")

    # 示例文件：波形 + 频谱 + 包络谱
    mid = show_runs[0].files[len(show_runs[0].files) // 2]
    data = load_vibration(mid)
    fs = 25_600.0
    fr = condition_rpm(show_runs[0].condition) / 60.0

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    t = np.arange(len(data)) / fs
    axes[0].plot(t, data[:, 0], lw=0.4)
    axes[0].set_title(f"{mid} 水平通道波形")
    axes[1].plot(t, data[:, 1], lw=0.4, color="tab:orange")
    axes[1].set_title("垂直通道波形")
    axes[1].set_xlabel("时间 (s)")
    axes[1].set_ylabel("幅值")
    save_fig(fig, out / "waveform.png")

    x = data[:, 1]
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    mag = np.abs(np.fft.rfft(x))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(freqs, mag, lw=0.5)
    ax.set_title(f"{mid} 垂直通道 FFT 频谱")
    ax.set_xlabel("频率 (Hz)")
    ax.set_ylabel("幅值")
    save_fig(fig, out / "spectrum.png")

    from scipy.signal import hilbert
    env = np.abs(hilbert(x))
    env_spec = np.abs(np.fft.rfft(env - env.mean()))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(freqs, env_spec, lw=0.5)
    for name, f_c in geometry.fault_frequencies(fr).items():
        ax.axvline(f_c, ls="--", lw=0.8, alpha=0.7)
        ax.text(f_c, ax.get_ylim()[1] * 0.92, name, fontsize=8, ha="center")
    ax.set_title(f"{mid} 垂直通道包络谱（虚线为理论特征频率）")
    ax.set_xlabel("频率 (Hz)")
    ax.set_ylabel("幅值")
    save_fig(fig, out / "envelope_spectrum.png")

    summary = {
        "runs": len(runs),
        "total_files": int(overview["文件数"].sum()),
        "conditions": sorted(overview["工况"].unique().tolist()),
        "geometry": geometry.__dict__,
        "example_file": str(mid),
    }
    (out / "explore_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"图表输出目录：{out}/")


if __name__ == "__main__":
    main()
