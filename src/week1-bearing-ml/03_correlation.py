"""第1周·第2天（下）：特征相关性分析与筛选。

产出：特征相关性热力图、特征-标签关联排序（互信息 + ANOVA F 值）、
top-K 特征散点图、selected_features.json 供后续建模使用。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.plotting import save_fig, setup_chinese_font

import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif, mutual_info_classif

META_COLS = ("condition", "bearing", "file_idx", "bearing_key", "stage", "rel_rul", "file_path")


def load_features(path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path)
    feats = [c for c in df.columns if c not in META_COLS]
    if not feats:
        raise ValueError(f"{path} 里没有特征列，请先运行 02_features.py")
    return df, feats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="results/week1/features.csv")
    ap.add_argument("--out", default="results/week1")
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()

    setup_chinese_font()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df, feats = load_features(Path(args.features))
    X = df[feats].to_numpy()
    y = df["stage"].to_numpy()

    # 1. Pearson 相关性热力图
    corr = df[feats].corr("pearson")
    fig, ax = plt.subplots(figsize=(0.28 * len(feats) + 4, 0.22 * len(feats) + 3))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feats)), feats, rotation=90, fontsize=6)
    ax.set_yticks(range(len(feats)), feats, fontsize=6)
    ax.set_title("特征 Pearson 相关系数矩阵")
    fig.colorbar(im, shrink=0.8)
    save_fig(fig, out / "correlation_pearson.png")

    # 2. 特征-标签关联排序（互信息 & ANOVA F）
    mi = mutual_info_classif(X, y, random_state=42)
    f_score, p_value = f_classif(X, y)
    rank = pd.DataFrame({
        "feature": feats, "mutual_info": mi, "anova_f": f_score, "anova_p": p_value,
    }).sort_values("mutual_info", ascending=False)
    rank.to_csv(out / "feature_ranking.csv", index=False, encoding="utf-8-sig")
    print(rank.head(10).to_string(index=False))

    top_k = rank["feature"].head(args.top_k).tolist()
    (out / "selected_features.json").write_text(
        json.dumps({"features": top_k, "top_k": args.top_k}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"已选出 top-{args.top_k} 特征 -> {out / 'selected_features.json'}")

    # 3. top-K 特征条形图
    fig, ax = plt.subplots(figsize=(9, 0.3 * args.top_k + 2))
    top = rank.head(args.top_k).iloc[::-1]
    ax.barh(top["feature"], top["mutual_info"])
    ax.set_xlabel("与健康阶段标签的互信息")
    ax.set_title(f"特征关联度 Top-{args.top_k}")
    save_fig(fig, out / "feature_ranking.png")

    # 4. top-4 特征两两散点（按阶段着色）
    top4 = rank["feature"].head(4).tolist()
    colors = {0: "tab:green", 1: "tab:orange", 2: "tab:red"}
    names = {0: "健康", 1: "退化", 2: "故障"}
    fig, axes = plt.subplots(3, 2, figsize=(11, 11))
    for ax, (i, j) in zip(axes.flat, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]):
        for s in (0, 1, 2):
            m = df["stage"] == s
            ax.scatter(df.loc[m, top4[i]], df.loc[m, top4[j]], s=6, alpha=0.5,
                       color=colors[s], label=names[s])
        ax.set_xlabel(top4[i], fontsize=8)
        ax.set_ylabel(top4[j], fontsize=8)
    axes.flat[0].legend(fontsize=8)
    save_fig(fig, out / "top_feature_pairs.png")
    print(f"图表输出目录：{out}/")


if __name__ == "__main__":
    main()
