"""第3周·第3天：模型优化与评估——学习率 × 损失函数消融实验。

在 ResNet18 上网格搜索学习率（默认 1e-2 / 1e-3 / 1e-4）与
损失函数（交叉熵 / 标签平滑交叉熵 / Focal Loss），输出对比表与热力图，
选最优组合。建议先用较小的 --epochs 快速筛选，再用最优组合全量训练。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.cv import FocalLoss, classification_pipeline
from common.plotting import save_fig, setup_chinese_font

import matplotlib.pyplot as plt


def make_loss(name: str) -> tuple[torch.nn.Module, str, str]:
    """返回 (损失模块, 显示名, 文件名安全标签)。"""
    if name == "ce":
        return torch.nn.CrossEntropyLoss(), "CE", "CE"
    if name == "smooth":
        return torch.nn.CrossEntropyLoss(label_smoothing=0.1), "CE+LS0.1", "CE_LS0.1"
    if name == "focal":
        return FocalLoss(gamma=2.0), "Focal(γ=2)", "Focal_g2"
    raise ValueError(f"未知损失函数 {name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--neu-dir", default="data/NEU-DET")
    ap.add_argument("--epochs", type=int, default=6, help="消融实验每组的训练轮数")
    ap.add_argument("--lrs", nargs="*", type=float, default=[1e-2, 1e-3, 1e-4])
    ap.add_argument("--losses", nargs="*", default=["ce", "smooth", "focal"])
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "efficientnet_b0"])
    ap.add_argument("--img-size", type=int, default=160)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--out", default="results/week2_3")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    setup_chinese_font()
    out = Path(args.out)
    summaries = []
    total = len(args.lrs) * len(args.losses)
    k = 0
    for lr in args.lrs:
        for loss_key in args.losses:
            k += 1
            loss_fn, loss_name, loss_label = make_loss(loss_key)
            print(f"\n===== 消融组合 {k}/{total}：lr={lr:g}  loss={loss_name} =====")
            s = classification_pipeline(
                neu_dir=args.neu_dir, arch=args.arch, epochs=args.epochs,
                batch_size=args.batch_size, lr=lr, img_size=args.img_size,
                pretrained=not args.no_pretrained, seed=args.seed,
                loss_fn=loss_fn, out_dir=args.out, save_model=None,
                loss_name=loss_label,
            )
            summaries.append(s)

    df = pd.DataFrame(summaries)
    csv_path = out / "optimization_results.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print("\n===== 消融实验汇总 =====")
    print(df[["lr", "loss", "test_acc", "test_macroF1"]].to_string(index=False))
    best = df.loc[df["test_acc"].idxmax()]
    print(f"最优组合：lr={best['lr']:g}，loss={best['loss']}，"
          f"test_acc={best['test_acc']:.4f}，macroF1={best['test_macroF1']:.4f}")

    loss_names = [make_loss(k_)[1] for k_ in args.losses]
    loss_labels = [make_loss(k_)[2] for k_ in args.losses]
    grid = np.zeros((len(args.lrs), len(args.losses)))
    for i, lr in enumerate(args.lrs):
        for j, lbl in enumerate(loss_labels):
            row = df[(df["lr"] == lr) & (df["loss"] == lbl)]
            if len(row):
                grid[i, j] = row.iloc[0]["test_acc"]
    fig, ax = plt.subplots(figsize=(1.6 * len(loss_names) + 3, 1.2 * len(args.lrs) + 2))
    im = ax.imshow(grid, cmap="YlGnBu", vmin=max(grid.min(), 0), vmax=grid.max())
    ax.set_xticks(range(len(loss_names)), loss_names)
    ax.set_yticks(range(len(args.lrs)), [f"{lr:g}" for lr in args.lrs])
    for i in range(len(args.lrs)):
        for j in range(len(loss_names)):
            ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center",
                    color="black", fontsize=9)
    ax.set_xlabel("损失函数")
    ax.set_ylabel("学习率")
    ax.set_title(f"{args.arch} 学习率×损失函数 测试准确率热力图")
    fig.colorbar(im, shrink=0.8)
    save_fig(fig, out / "optimization_heatmap.png")
    print(f"结果表：{csv_path}")


if __name__ == "__main__":
    main()
