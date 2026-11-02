"""PyTorch 公共件：设备选择、原始振动序列加载、训练/评估循环。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from common.bearing import load_vibration

META_COLS = ("condition", "bearing", "file_idx", "bearing_key", "stage",
             "rel_rul", "file_path")


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-8)


def load_sequences(
    features_csv: str | Path, seq_len: int, channel: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """把 features.csv 每行映射回原始 CSV，取指定通道降采样成 seq_len 长度序列。

    返回 (X: float32 (n, seq_len), y_stage: int64, y_rul: float32, keys: list)。
    """
    df = pd.read_csv(features_csv)
    if "file_path" not in df.columns:
        raise ValueError("特征表缺少 file_path 列，请用 02_features.py 重新生成。")
    xs, y_stage, y_rul, keys = [], [], [], []
    for _, row in df.iterrows():
        x = load_vibration(row["file_path"])[:, channel]
        stride = max(1, len(x) // seq_len)
        seg = x[::stride][:seq_len]
        if len(seg) < seq_len:
            seg = np.pad(seg, (0, seq_len - len(seg)))
        xs.append(zscore(seg))
        y_stage.append(int(row["stage"]))
        y_rul.append(float(row["rel_rul"]))
        keys.append(str(row["bearing_key"]))
    return (np.asarray(xs, dtype=np.float32), np.asarray(y_stage, dtype=np.int64),
            np.asarray(y_rul, dtype=np.float32), keys)


class TensorDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(X)
        self.y = torch.as_tensor(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int):
        return self.X[i], self.y[i]


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle)


def train(
    model: torch.nn.Module, tr_loader: DataLoader, te_loader: DataLoader,
    epochs: int, lr: float, device: torch.device, task: str = "cls",
) -> tuple[torch.nn.Module, list[dict]]:
    """通用训练循环。task='cls' 用交叉熵，'reg' 用 MSE。返回 (model, history)。"""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss() if task == "cls" else torch.nn.MSELoss()
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device).float(), yb.to(device)
            if task == "reg":
                yb = yb.float().unsqueeze(1)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
        tr_loss = total_loss / len(tr_loader.dataset)
        te_loss, te_metric = evaluate_loss(model, te_loader, device, task)
        history.append({"epoch": epoch, "train_loss": tr_loss,
                        "test_loss": te_loss, "test_metric": te_metric})
        print(f"epoch {epoch:3d}/{epochs}  train_loss={tr_loss:.4f}  "
              f"test_loss={te_loss:.4f}  test_{'acc' if task == 'cls' else 'R2'}={te_metric:.4f}")
    return model, history


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module, loader: DataLoader, device: torch.device, task: str,
) -> tuple[float, float]:
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss() if task == "cls" else torch.nn.MSELoss()
    total, metric_terms = 0.0, []
    preds, trues = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device).float(), yb.to(device)
        y_true = yb.float().unsqueeze(1) if task == "reg" else yb
        out = model(xb)
        total += loss_fn(out, y_true).item() * len(xb)
        if task == "cls":
            preds.append(out.argmax(1).cpu())
            trues.append(yb.cpu())
        else:
            preds.append(out.squeeze(1).cpu())
            trues.append(yb.cpu())
    if not preds:
        return 0.0, 0.0
    p = torch.cat(preds).numpy()
    t = torch.cat(trues).numpy()
    metric = float((p == t).mean()) if task == "cls" else float(
        1 - np.sum((p - t) ** 2) / (np.sum((t - t.mean()) ** 2) + 1e-12))
    return total / len(loader.dataset), metric


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    for xb, _ in loader:
        out.append(model(xb.to(device).float()).detach().cpu())
    return torch.cat(out).numpy()
