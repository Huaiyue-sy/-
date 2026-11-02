"""计算机视觉公共件：图像数据集、预处理、分类模型训练/评估循环。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageDataset(Dataset):
    """灰度瑕疵图 → 3 通道张量（适配 ImageNet 预训练骨干）。"""

    def __init__(self, items: list[tuple[Path, int]], transform=None):
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, label = self.items[i]
        img = Image.open(path).convert("L").convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(label)


def default_transforms(img_size: int, train: bool) -> transforms.Compose:
    ops = [transforms.Resize((img_size, img_size))]
    if train:
        ops += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(ops)


def make_image_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def build_resnet18(n_classes: int, pretrained: bool = True) -> torch.nn.Module:
    from torchvision.models import ResNet18_Weights, resnet18
    try:
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
    except Exception as e:  # 预训练权重下载失败时退回随机初始化
        print(f"[警告] 预训练权重不可用（{e}），改用随机初始化。")
        model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, n_classes)
    return model


def build_efficientnet_b0(n_classes: int, pretrained: bool = True) -> torch.nn.Module:
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
    try:
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
    except Exception as e:
        print(f"[警告] 预训练权重不可用（{e}），改用随机初始化。")
        model = efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, n_classes)
    return model


class FocalLoss(torch.nn.Module):
    """多分类 Focal Loss（调优实验用）。"""

    def __init__(self, gamma: float = 2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = torch.nn.functional.log_softmax(logits, dim=1)
        logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        p_t = logp_t.exp()
        loss = -((1 - p_t) ** self.gamma) * logp_t
        if self.weight is not None:
            loss = loss * self.weight[target]
        return loss.mean()


def train_image_classifier(
    model: torch.nn.Module, tr_loader: DataLoader, va_loader: DataLoader,
    epochs: int, lr: float, device: torch.device,
    loss_fn: torch.nn.Module | None = None, verbose: bool = True,
) -> tuple[torch.nn.Module, list[dict]]:
    """训练循环：以验证集准确率最高的权重为准，返回 (best_model, history)。"""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = loss_fn or torch.nn.CrossEntropyLoss()
    history: list[dict] = []
    best_acc, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
            n += len(xb)
        va_acc, va_loss = evaluate_image_classifier(model, va_loader, device, loss_fn)
        history.append({"epoch": epoch, "train_loss": total_loss / n,
                        "val_loss": va_loss, "val_acc": va_acc})
        if va_acc > best_acc:
            best_acc, best_state = va_acc, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"epoch {epoch:3d}/{epochs}  train_loss={total_loss / n:.4f}  "
                  f"val_loss={va_loss:.4f}  val_acc={va_acc:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def evaluate_image_classifier(
    model: torch.nn.Module, loader: DataLoader, device: torch.device,
    loss_fn: torch.nn.Module | None = None,
) -> tuple[float, float]:
    model.eval()
    loss_fn = loss_fn or torch.nn.CrossEntropyLoss()
    total_loss, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total_loss += loss_fn(logits, yb).item() * len(xb)
        correct += int((logits.argmax(1) == yb).sum())
        n += len(xb)
    return correct / n, total_loss / n


@torch.no_grad()
def predict_images(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    preds, labels = [], []
    for xb, yb in loader:
        preds.append(model(xb.to(device)).argmax(1).cpu())
        labels.append(yb)
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()


@torch.no_grad()
def predict_single(model: torch.nn.Module, img: Image.Image, img_size: int,
                   device: torch.device) -> tuple[int, float]:
    """单张图推理，返回 (类别下标, 置信度)。"""
    t = default_transforms(img_size, train=False)(img).unsqueeze(0).to(device)
    prob = torch.softmax(model(t), dim=1)[0]
    cls = int(prob.argmax())
    return cls, float(prob[cls])


def classification_pipeline(
    neu_dir: str | Path,
    arch: str,
    epochs: int,
    batch_size: int = 32,
    lr: float = 1e-3,
    img_size: int = 224,
    pretrained: bool = True,
    seed: int = 42,
    loss_fn: torch.nn.Module | None = None,
    out_dir: str | Path | None = None,
    save_model: str | Path | None = None,
    loss_name: str = "CE",
) -> dict:
    """NEU-DET 六分类全流程：数据划分 → 训练 → 测试 → 产出图表/模型。

    返回指标摘要 dict（arch/lr/loss/val_acc/test_acc/...）。
    """
    import pandas as pd
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    from common.neu import CLASSES, detect_layout, list_classification_images, stratified_split
    from common.plotting import plot_confusion_matrix, save_fig, setup_chinese_font

    setup_chinese_font()
    neu_dir = Path(neu_dir)
    layout = detect_layout(neu_dir)
    if layout not in ("classification", "voc"):
        raise RuntimeError(
            f"{neu_dir} 不是分类可用布局（每类文件夹或 VOC/官方布局），"
            "分类脚本请指向 NEU-DET 数据集根目录。")
    items = list_classification_images(neu_dir)
    tr_items, va_items, te_items = stratified_split(items, seed=seed)
    print(f"[{arch}] 训练 {len(tr_items)} / 验证 {len(va_items)} / 测试 {len(te_items)}")

    tr_loader = make_image_loader(
        ImageDataset(tr_items, default_transforms(img_size, train=True)), batch_size, True)
    va_loader = make_image_loader(
        ImageDataset(va_items, default_transforms(img_size, train=False)), batch_size, False)
    te_loader = make_image_loader(
        ImageDataset(te_items, default_transforms(img_size, train=False)), batch_size, False)

    torch.manual_seed(seed)
    if arch == "resnet18":
        model = build_resnet18(len(CLASSES), pretrained)
    elif arch == "efficientnet_b0":
        model = build_efficientnet_b0(len(CLASSES), pretrained)
    else:
        raise ValueError(f"不支持的 arch：{arch}")

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if getattr(torch.backends, "mps", None)
                          and torch.backends.mps.is_available() else "cpu")
    import time
    t0 = time.perf_counter()
    model, history = train_image_classifier(
        model, tr_loader, va_loader, epochs, lr, device, loss_fn=loss_fn)
    train_sec = time.perf_counter() - t0

    y_pred, y_true = predict_images(model, te_loader, device)
    test_acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    va_acc = history[-1]["val_acc"] if history else float("nan")
    print(f"[{arch}] 测试集：acc={test_acc:.4f} macroP={p:.4f} macroR={r:.4f} macroF1={f1:.4f}")

    summary = {"arch": arch, "lr": lr, "loss": loss_name, "epochs": epochs,
               "img_size": img_size, "pretrained": bool(pretrained),
               "device": str(device), "train_sec": round(train_sec, 1),
               "sec_per_epoch": round(train_sec / max(1, epochs), 2),
               "best_val_acc": va_acc, "test_acc": test_acc,
               "test_macroP": p, "test_macroR": r, "test_macroF1": f1}

    out_dir = Path(out_dir) if out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = f"{arch}_{loss_name}_lr{lr:g}"
        pd.DataFrame(history).to_csv(
            out_dir / f"history_{tag}.csv", index=False, encoding="utf-8-sig")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([h["epoch"] for h in history], [h["train_loss"] for h in history], label="训练损失")
        ax.plot([h["epoch"] for h in history], [h["val_loss"] for h in history], label="验证损失")
        ax2 = ax.twinx()
        ax2.plot([h["epoch"] for h in history], [h["val_acc"] for h in history],
                 color="tab:green", label="验证准确率")
        ax.set_xlabel("epoch")
        ax.set_ylabel("损失")
        ax2.set_ylabel("准确率")
        ax.set_title(f"{arch} 训练曲线（测试 acc={test_acc:.4f}）")
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="lower right", fontsize=8)
        save_fig(fig, out_dir / f"curve_{tag}.png")
        plot_confusion_matrix(
            y_true, y_pred, CLASSES, out_dir / f"cm_{tag}.png", title=f"{arch} 混淆矩阵")

    if save_model:
        save_path = Path(save_model)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "arch": arch,
                    "classes": CLASSES, "img_size": img_size, "test_acc": test_acc}, save_path)
        print(f"模型已保存：{save_path}")
    return summary
