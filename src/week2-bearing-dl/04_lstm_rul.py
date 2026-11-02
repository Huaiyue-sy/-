"""第2周·第4天（承第1周回归任务）：LSTM 剩余寿命（RUL）回归。

输入：每个轴承按文件顺序滑窗，窗口内 W 个文件的 top-K 特征构成序列
（W×K，已标准化），预测窗口末文件的相对剩余寿命 rel_rul。
按轴承留出测试，输出 MAE / RMSE / R²、预测-真值散点图与模型。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.bearing import bearing_split_mask
from common.dl import make_loader, pick_device, predict, train
from common.plotting import save_fig, setup_chinese_font

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


class LSTMRegressor(torch.nn.Module):
    def __init__(self, in_dim, hidden=64, layers=2):
        super().__init__()
        self.lstm = torch.nn.LSTM(in_dim, hidden, layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


def build_windows(df: pd.DataFrame, feats: list[str], window: int):
    """按轴承滑窗：序列 (window, K) → 末文件 rel_rul。"""
    Xs, ys, keys = [], [], []
    for key, g in df.groupby("bearing_key", sort=True):
        g = g.sort_values("file_idx")
        arr = g[feats].to_numpy()
        ruls = g["rel_rul"].to_numpy()
        for end in range(window - 1, len(g)):
            Xs.append(arr[end - window + 1: end + 1])
            ys.append(ruls[end])
            keys.append(key)
    return (np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), keys)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="results/week1/features.csv")
    ap.add_argument("--selected", default="results/week1/selected_features.json")
    ap.add_argument("--window", type=int, default=8, help="滑窗文件数 W")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="results/week2")
    ap.add_argument("--save-model", default="models/lstm_rul.pth")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    setup_chinese_font()
    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"训练设备：{device}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features)
    sel_path = Path(args.selected)
    feats = json.loads(sel_path.read_text(encoding="utf-8"))["features"] \
        if sel_path.exists() else [c for c in df.columns if c not in
                                    ("condition", "bearing", "file_idx", "bearing_key",
                                     "stage", "rel_rul", "file_path")]
    print(f"窗口 W={args.window}，特征 K={len(feats)}")

    X, y, keys = build_windows(df, feats, args.window)
    tr_mask, te_mask = bearing_split_mask(keys)
    scaler = StandardScaler().fit(X[tr_mask].reshape(-1, X.shape[-1]))
    X_tr = scaler.transform(X[tr_mask].reshape(-1, X.shape[-1])).reshape(X[tr_mask].shape)
    X_te = scaler.transform(X[te_mask].reshape(-1, X.shape[-1])).reshape(X[te_mask].shape)
    X_tr, X_te = X_tr.astype(np.float32), X_te.astype(np.float32)
    y_tr, y_te = y[tr_mask], y[te_mask]
    print(f"窗口样本：训练 {len(y_tr)} / 测试 {len(y_te)}（按轴承留出）")

    tr_loader = make_loader(X_tr, y_tr, args.batch_size, shuffle=True)
    te_loader = make_loader(X_te, y_te, args.batch_size, shuffle=False)
    model = LSTMRegressor(len(feats), args.hidden)
    model, history = train(model, tr_loader, te_loader, args.epochs, args.lr, device, task="reg")

    y_pred = predict(model, te_loader, device).squeeze()
    mae = mean_absolute_error(y_te, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
    r2 = r2_score(y_te, y_pred)
    print(f"LSTM-RUL 测试集：MAE={mae:.4f} RMSE={rmse:.4f} R²={r2:.4f}")

    pd.DataFrame(history).to_csv(out / "lstm_rul_history.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.scatter(y_te, y_pred, s=8, alpha=0.4)
    ax.set_xlabel("真实 rel_rul")
    ax.set_ylabel("预测 rel_rul")
    ax.set_title(f"LSTM 剩余寿命回归（R²={r2:.3f}）")
    save_fig(fig, out / "lstm_rul_pred_vs_true.png")

    save_path = Path(args.save_model)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "features": feats,
                "window": args.window, "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(), "test_r2": r2}, save_path)
    pd.DataFrame([{"model": "LSTM", "mae": mae, "rmse": rmse, "r2": r2,
                   "window": args.window, "epochs": args.epochs}]).to_csv(
        out / "lstm_rul_results.csv", index=False, encoding="utf-8-sig")
    print(f"模型已保存：{save_path}")


if __name__ == "__main__":
    main()
