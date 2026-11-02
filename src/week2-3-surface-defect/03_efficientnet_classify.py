"""第3周·第1天：EfficientNet-B0 图像分类，并与 ResNet18 对比。

流程与 ResNet 脚本一致（分层划分、数据增强、ImageNet 预训练换头微调），
结束后若存在 ResNet 结果则打印两者对比。
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
    ap.add_argument("--save-model", default="models/efficientnet_b0_neu.pth")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    summary = classification_pipeline(
        neu_dir=args.neu_dir, arch="efficientnet_b0", epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, img_size=args.img_size,
        pretrained=not args.no_pretrained, seed=args.seed,
        out_dir=args.out, save_model=args.save_model,
    )
    out = Path(args.out)
    pd.DataFrame([summary]).to_csv(
        out / "efficientnet_b0_results.csv", index=False, encoding="utf-8-sig")

    resnet_csv = out / "resnet18_results.csv"
    if resnet_csv.exists():
        resnet = pd.read_csv(resnet_csv).iloc[0]
        print("\n===== 模型对比 =====")
        print(f"ResNet18        测试 acc={resnet['test_acc']:.4f}  macroF1={resnet['test_macroF1']:.4f}")
        print(f"EfficientNet-B0 测试 acc={summary['test_acc']:.4f}  macroF1={summary['test_macroF1']:.4f}")
    print("提示：两种骨干的结果将作为实训报告对比实验的数据来源。")


if __name__ == "__main__":
    main()
