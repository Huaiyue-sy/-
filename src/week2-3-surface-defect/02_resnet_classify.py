"""第2周·第4天：ResNet18 图像分类（迁移学习，六类瑕疵）。

流程：分层划分 8:1:1 → 数据增强（翻转/旋转）→ ImageNet 预训练
ResNet18 换分类头微调 → 测试集评估（准确率/macro P-R-F1/混淆矩阵）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.cv import classification_pipeline


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--neu-dir", default="data/NEU-DET")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--out", default="results/week2_3")
    ap.add_argument("--save-model", default="models/resnet18_neu.pth")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    summary = classification_pipeline(
        neu_dir=args.neu_dir, arch="resnet18", epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, img_size=args.img_size,
        pretrained=not args.no_pretrained, seed=args.seed,
        out_dir=args.out, save_model=args.save_model,
    )
    pd.DataFrame([summary]).to_csv(
        Path(args.out) / "resnet18_results.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
