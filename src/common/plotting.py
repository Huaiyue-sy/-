"""绘图工具：中文字体配置与保存。"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

_FONT_CANDIDATES = (
    "PingFang SC", "Hiragino Sans GB", "Arial Unicode MS",
    "Songti SC", "SimHei", "Microsoft YaHei",
)


def setup_chinese_font() -> str:
    """选择本机第一个可用中文字体，返回字体名。"""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in _FONT_CANDIDATES:
        if name in installed:
            plt.rcParams["font.family"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    plt.rcParams["axes.unicode_minus"] = False
    return "sans-serif"


def save_fig(fig: plt.Figure, out_path: str | Path, dpi: int = 150) -> Path:
    """保存图并关闭句柄，自动建目录。"""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_confusion_matrix(
    y_true, y_pred, classes: list[str], out_path: str | Path, title: str = "混淆矩阵",
) -> Path:
    """绘制并保存混淆矩阵。"""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    fig, ax = plt.subplots(figsize=(max(4.2, 0.7 * len(classes) + 1),) * 2)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("预测")
    ax.set_ylabel("真实")
    fig.colorbar(im, shrink=0.8)
    return save_fig(fig, out_path)
