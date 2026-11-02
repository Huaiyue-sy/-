"""第2周·第2天：LSTM 故障（健康阶段）分类，并与 RNN 对比。

结构与 RNN 脚本一致（原始振动序列 → 末步隐状态 → 全连接三分类），
训练结束后自动读取 rnn_cls_results.csv 打印两者对比。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.bearing import bearing_split_mask
from common.dl import load_sequences, make_loader, pick_device, predict, train
from common.plotting import save_fig, setup_chinese_font

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

STAGE_NAMES = {0: "健康", 1: "退化", 2: "故障"}


class LSTMClassifier(torch.nn.Module):
    def __init__(self, hidden=64, layers=2, n_classes=3):
        super().__init__()
        self.lstm = torch.nn.LSTM(1, hidden, layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden, n_classes)

    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(-1))
        return self.fc(out[:, -1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="results/week1/features.csv")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--channel", type=int, default=1)
    ap.add_argument("--out", default="results/week2")
    ap.add_argument("--save-model", default="models/lstm_cls.pth")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    setup_chinese_font()
    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"训练设备：{device}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    X, y_stage, _, keys = load_sequences(args.features, args.seq_len, args.channel)
    tr_mask, te_mask = bearing_split_mask(keys)
    X_tr, X_te, y_tr, y_te = X[tr_mask], X[te_mask], y_stage[tr_mask], y_stage[te_mask]
    print(f"样本：训练 {len(y_tr)} / 测试 {len(y_te)}，序列长度 {args.seq_len}")

    tr_loader = make_loader(X_tr, y_tr, args.batch_size, shuffle=True)
    te_loader = make_loader(X_te, y_te, args.batch_size, shuffle=False)

    model = LSTMClassifier(args.hidden, args.layers)
    model, history = train(model, tr_loader, te_loader, args.epochs, args.lr, device, task="cls")

    y_pred = predict(model, te_loader, device).argmax(1)
    acc = accuracy_score(y_te, y_pred)
    print(f"LSTM 测试集准确率：{acc:.4f}")

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out / "lstm_cls_history.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hist_df["epoch"], hist_df["train_loss"], label="训练损失")
    ax.plot(hist_df["epoch"], hist_df["test_loss"], label="测试损失")
    ax2 = ax.twinx()
    ax2.plot(hist_df["epoch"], hist_df["test_metric"], color="tab:green", label="测试准确率")
    ax.set_xlabel("epoch")
    ax.set_ylabel("损失")
    ax2.set_ylabel("准确率")
    ax.set_title(f"LSTM 分类训练曲线（测试准确率 {acc:.4f}）")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="lower right", fontsize=8)
    save_fig(fig, out / "lstm_cls_curve.png")

    cm = confusion_matrix(y_te, y_pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), [STAGE_NAMES[i] for i in range(3)], rotation=45)
    ax.set_yticks(range(3), [STAGE_NAMES[i] for i in range(3)])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title("LSTM 混淆矩阵")
    save_fig(fig, out / "lstm_cls_cm.png")

    save_path = Path(args.save_model)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "seq_len": args.seq_len,
                "hidden": args.hidden, "layers": args.layers, "test_acc": acc}, save_path)

    rows = [{"model": "LSTM", "test_acc": acc, "seq_len": args.seq_len, "epochs": args.epochs}]
    rnn_csv = out / "rnn_cls_results.csv"
    if rnn_csv.exists():
        rnn_acc = float(pd.read_csv(rnn_csv)["test_acc"].iloc[0])
        rows.append(pd.read_csv(rnn_csv).iloc[0].to_dict())
        print(f"\n与 RNN 对比：RNN acc={rnn_acc:.4f}  LSTM acc={acc:.4f}")
    pd.DataFrame(rows).to_csv(out / "lstm_cls_results.csv", index=False, encoding="utf-8-sig")
    print(f"模型已保存：{save_path}")


if __name__ == "__main__":
    main()
