"""第2周·扩展：一维 CNN 故障分类，补齐「CNN vs LSTM vs 随机森林」对比。

输入与 RNN/LSTM 完全一致（原始振动序列、同划分、同超参框架），
结构：Conv1d×2 + 池化 → 全局平均池化 → 全连接三分类。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.bearing import bearing_split_mask
from common.dl import load_sequences, make_loader, pick_device, predict, train
from common.plotting import plot_confusion_matrix, save_fig, setup_chinese_font

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

STAGE_NAMES = ["健康", "退化", "故障"]


class CNN1DClassifier(torch.nn.Module):
    def __init__(self, seq_len: int, n_classes: int = 3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(1, 32, kernel_size=7, padding=3), torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(32, 64, kernel_size=5, padding=2), torch.nn.ReLU(),
            torch.nn.AdaptiveMaxPool1d(8),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 8, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x.unsqueeze(1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="results/week1/features.csv")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--channel", type=int, default=1)
    ap.add_argument("--out", default="results/week2")
    ap.add_argument("--save-model", default="models/cnn_cls.pth")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    setup_chinese_font()
    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"训练设备：{device}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    X, y, _, keys = load_sequences(args.features, args.seq_len, args.channel)
    tr_mask, te_mask = bearing_split_mask(keys)
    X_tr, X_te, y_tr, y_te = X[tr_mask], X[te_mask], y[tr_mask], y[te_mask]
    print(f"样本：训练 {len(y_tr)} / 测试 {len(y_te)}，序列长度 {args.seq_len}")

    model, history = train(
        CNN1DClassifier(args.seq_len),
        make_loader(X_tr, y_tr, args.batch_size, shuffle=True),
        make_loader(X_te, y_te, args.batch_size, shuffle=False),
        args.epochs, args.lr, device, task="cls",
    )
    y_pred = predict(model, make_loader(X_te, y_te, args.batch_size, shuffle=False), device).argmax(1)
    acc = accuracy_score(y_te, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_te, y_pred, average="macro", zero_division=0)
    print(f"CNN 测试集：acc={acc:.4f} macroP={p:.4f} macroR={r:.4f} macroF1={f1:.4f}")

    pd.DataFrame(history).to_csv(out / "cnn_cls_history.csv", index=False, encoding="utf-8-sig")
    plot_confusion_matrix(y_te, y_pred, STAGE_NAMES, out / "cnn_cls_cm.png", title="CNN 混淆矩阵")

    rows = [{"model": "CNN1D", "test_acc": acc, "test_macroF1": f1,
             "seq_len": args.seq_len, "epochs": args.epochs}]
    for name, csv_name in (("RNN", "rnn_cls_results.csv"), ("LSTM", "lstm_cls_results.csv")):
        csv_path = out / csv_name
        if csv_path.exists():
            row = pd.read_csv(csv_path).iloc[0]
            rows.append({"model": name, "test_acc": row.get("test_acc"),
                         "test_macroF1": None,  # 早期结果表未记录 macroF1，需要时在 02/03 末尾补一列重跑
                         "seq_len": row.get("seq_len"), "epochs": row.get("epochs")})
    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(out / "cnn_vs_lstm_vs_rf.csv", index=False, encoding="utf-8-sig")
    print("\n===== CNN vs RNN/LSTM =====")
    print(cmp_df.to_string(index=False))
    print("提示：随机森林结果在 results/week1/classify_health_results.csv，汇总表见报告第 5 章。")

    save_path = Path(args.save_model)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "seq_len": args.seq_len,
                "test_acc": acc}, save_path)
    print(f"模型已保存：{save_path}")


if __name__ == "__main__":
    main()
