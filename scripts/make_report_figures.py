"""生成正式报告所需的论文级图表。

输出目录：实训报告/figures/
所有图均为 300 dpi，数据来自 results/ 与 data/ 下的真实实验产物。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image, ImageChops, ImageDraw, ImageFont
from matplotlib import font_manager, pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "实训报告" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

FONT_CANDIDATES = ["PingFang SC", "Songti SC", "Heiti SC", "Arial Unicode MS"]
for name in FONT_CANDIDATES:
    if any(item.name == name for item in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams.update({
    "font.size": 9.5,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#9aa39e",
    "axes.labelcolor": "#1f1f1d",
    "xtick.color": "#3f3f3b",
    "ytick.color": "#3f3f3b",
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
})

INK = "#1f1f1d"
MUTED = "#5f5f5a"
LINE = "#c9cfcc"
GRID = "#e4e7e5"
GREEN = "#3f8a5f"
GREEN_TINT = "#eaf5ee"
BLUE = "#3b6fb5"
BLUE_TINT = "#eaf1f8"
AMBER = "#a9761a"
AMBER_TINT = "#fbf3e2"
PURPLE = "#5b58a8"
PURPLE_TINT = "#efeff8"
GRAY_TINT = "#f4f5f4"

CLASS_CN = ["龟裂", "夹杂", "斑块", "麻点", "氧化铁皮压入", "划痕"]
CLASS_EN = ["crazing", "inclusion", "patches", "pitted_surface", "rolled_in_scale", "scratches"]


def save(fig: plt.Figure, name: str) -> None:
    path = OUT / name
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def rounded(ax, x, y, w, h, text, fc="white", ec=LINE, tc=INK, fs=8.4,
            weight="normal", lw=0.9, z=2) -> None:
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.9",
        fc=fc, ec=ec, lw=lw, zorder=z,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", color=tc, fontsize=fs,
        fontweight=weight, linespacing=1.35, zorder=z + 1,
    )


def arrow(ax, x1, y1, x2, y2, color=LINE, lw=1.1, style="-|>", ls="-",
          rad=0.0, z=1) -> None:
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=9,
        color=color, lw=lw, linestyle=ls, shrinkA=0, shrinkB=0,
        connectionstyle=f"arc3,rad={rad}", zorder=z,
    ))


def trim_white(path: Path, pad: int = 24) -> Image.Image:
    im = Image.open(path)
    if im.mode == "RGBA":
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    im = im.convert("RGB")
    diff = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255)))
    box = diff.getbbox()
    if box:
        left = max(0, box[0] - pad)
        top = max(0, box[1] - pad)
        right = min(im.width, box[2] + pad)
        bottom = min(im.height, box[3] + pad)
        im = im.crop((left, top, right, bottom))
    return im


def font_path() -> str:
    family = plt.rcParams["font.family"]
    if isinstance(family, (list, tuple)):
        family = family[0]
    return font_manager.findfont(family, fallback_to_default=True)


# ---------------------------------------------------------------- 图 1-1 数据样例
def figure_neu_samples() -> None:
    samples = [
        ("crazing_1.jpg", "龟裂（crazing）"),
        ("inclusion_1.jpg", "夹杂（inclusion）"),
        ("patches_1.jpg", "斑块（patches）"),
        ("pitted_surface_1.jpg", "麻点（pitted_surface）"),
        ("rolled-in_scale_1.jpg", "氧化铁皮压入（rolled_in_scale）"),
        ("scratches_1.jpg", "划痕（scratches）"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(8.6, 6.2))
    for ax, (filename, title) in zip(axes.ravel(), samples):
        image = Image.open(ROOT / "data" / "NEU-DET" / "IMAGES" / filename).convert("L")
        ax.imshow(np.asarray(image), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(title, fontsize=10, color=INK, pad=5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(LINE)
            spine.set_linewidth(0.7)
    fig.subplots_adjust(wspace=0.08, hspace=0.16)
    save(fig, "fig_1_1_neu_samples.png")


# ---------------------------------------------------------------- 图 2-1 总体架构
def figure_architecture() -> None:
    fig, ax = plt.subplots(figsize=(12.2, 7.8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    bands = [
        (86.0, 11.0, "应用层", GREEN_TINT),
        (71.0, 11.0, "能力层", BLUE_TINT),
        (56.0, 11.0, "模型层", AMBER_TINT),
        (41.0, 11.0, "内核层", PURPLE_TINT),
        (27.0, 10.0, "公共层", GRAY_TINT),
        (10.0, 12.5, "数据层", GRAY_TINT),
    ]
    for y, h, label, tint in bands:
        ax.add_patch(FancyBboxPatch(
            (1.0, y), 98.0, h, boxstyle="round,pad=0.0,rounding_size=1.0",
            fc=tint, ec=LINE, lw=0.8, zorder=0,
        ))
        ax.add_patch(FancyBboxPatch(
            (2.2, y + 1.0), 11.0, h - 2.0,
            boxstyle="round,pad=0.0,rounding_size=0.8",
            fc=INK, ec=INK, lw=0.8, zorder=1,
        ))
        ax.text(7.7, y + h / 2, label, ha="center", va="center",
                color="white", fontsize=10.5, fontweight="bold", zorder=2)

    # 应用层
    apps = [
        "运维诊断智能体\n异常 · 分类 · RUL · 报告",
        "视觉质检智能体\n分类 · 台账 · 查询 · 预警",
        "统一智能体\n路由 · 工具 · 上下文",
        "Web 可视化智能体\n问答 · 图表 · YOLO 检测",
    ]
    x = 15.5
    for text in apps:
        rounded(ax, x, 87.2, 19.3, 8.6, text, fc="white", ec=GREEN, fs=8.2)
        x += 20.7

    # 能力层
    abilities = [
        "设备状态能力\n异常检测 · 阶段分类 · RUL 回归 · 包络定位",
        "视觉质检能力\n缺陷分类 · 台账记忆 · 自然语言查询 · 连续预警",
        "知识与大模型能力\nTF-IDF RAG · LLM 报告 · 图表规格生成",
    ]
    x = 15.5
    for text in abilities:
        rounded(ax, x, 72.2, 26.6, 8.6, text, fc="white", ec=BLUE, fs=7.9)
        x += 28.1

    # 模型层
    models = [
        "week1_classifier\n.joblib",
        "week1_rul\n.joblib",
        "resnet18_neu\n.pth",
        "efficientnet_b0\n_neu.pth",
        "YOLOv8n\nbest.pt",
    ]
    x = 15.5
    for text in models:
        rounded(ax, x, 57.2, 15.5, 8.6, text, fc="white", ec=AMBER, fs=7.4)
        x += 16.8

    # 内核层
    kernels = [
        "week1-bearing-ml\n特征工程 · RF/SVM/KNN/LR · RUL",
        "week2-bearing-dl\nRNN · LSTM · 1D-CNN · LSTM-RUL",
        "week2-3-surface-defect\nResNet · EfficientNet · YOLOv8n · 消融",
    ]
    x = 15.5
    for text in kernels:
        rounded(ax, x, 42.2, 26.6, 8.6, text, fc="white", ec=PURPLE, fs=7.9)
        x += 28.1

    # 公共层
    commons = ["bearing.py", "features.py", "dl.py", "cv.py", "neu.py", "plotting.py", "confirm.py"]
    x = 15.5
    for text in commons:
        rounded(ax, x, 28.4, 11.3, 7.1, text, fc="white", ec=LINE, fs=7.1)
        x += 12.1

    # 数据层
    data = [
        "XJTU-SY 轴承振动\n15 套全寿命 · 9216 个 CSV",
        "NEU-DET 表面缺陷\n1800 张图像 · VOC / YOLO 标注",
        "工艺文件与台账\n7 份工艺文件 · SQLite 台账",
    ]
    x = 15.5
    for text in data:
        rounded(ax, x, 11.2, 26.6, 10.0, text, fc="white", ec=LINE, fs=8.2)
        x += 28.1

    for x in (32.0, 67.0):
        arrow(ax, x, 86.0, x, 82.0)
        arrow(ax, x, 71.0, x, 67.0)
        arrow(ax, x, 56.0, x, 52.0)
        arrow(ax, x, 41.0, x, 37.0)
        arrow(ax, x, 27.0, x, 22.5)

    ax.text(99.0, 4.6, "调用方向：应用层 → 能力层 → 模型层 → 算法内核 → 公共件 / 数据",
            ha="right", va="center", color=MUTED, fontsize=8.0)
    save(fig, "fig_2_1_architecture.png")


# ---------------------------------------------------------------- 图 2-2 数据流
def figure_data_flow() -> None:
    fig, ax = plt.subplots(figsize=(12.2, 7.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    width = 27.0
    levels = [75.0, 62.6, 50.2, 37.8, 25.4, 13.0]
    linear_lanes = [
        (5.0, "（a）轴承数据流", GREEN, GREEN_TINT, [
            "XJTU-SY CSV\n两通道 · 32768 点",
            "特征提取\n时域 11 + 频域 8 + 包络 5",
            "features.csv\n48 维 + rms_mean",
            "标签构造\n健康 / 退化 / 故障 · rel_rul",
            "模型训练\nRF / GBDT / LSTM",
            "运维诊断智能体\n异常 · 阶段 · RUL · 报告",
        ]),
        (68.0, "（c）智能体数据流", PURPLE, PURPLE_TINT, [
            "用户自然语言\n问题 / 图片 / CSV",
            "意图路由\n路径 · 文件类型 · 关键词",
            "工具调用\n模型 · 台账 · RAG · 检测",
            "结构化证据\n指标 · 检测框 · 检索片段",
            "大模型组织\n事实约束 · 可降级",
            "终端 / 浏览器\n文字 · 图表 · 标注图",
        ]),
    ]

    for left, title, color, tint, steps in linear_lanes:
        rounded(ax, left, 88.0, width, 8.4, title, fc=color, ec=color, tc="white",
                fs=10.0, weight="bold")
        for index, (y, text) in enumerate(zip(levels, steps)):
            rounded(ax, left, y, width, 9.4, text, fc=tint, ec=color, fs=7.8)
            if index < len(steps) - 1:
                arrow(ax, left + width / 2, y, left + width / 2, y - 3.0, color=color)

    # 图像线为分类与检测双分支
    left = 36.5
    rounded(ax, left, 88.0, width, 8.4, "（b）图像数据流：分类 / 检测",
            fc=BLUE, ec=BLUE, tc="white", fs=9.4, weight="bold")
    rounded(ax, left, levels[0], width, 9.4, "NEU-DET 图像与标注\n1800 张 · 6 类缺陷",
            fc=BLUE_TINT, ec=BLUE, fs=7.8)

    half = (width - 1.2) / 2
    left_branch = left
    right_branch = left + half + 1.2
    branch_boxes = [
        (left_branch, "分类布局\n分层 8:1:1"),
        (right_branch, "VOC → YOLO\n分层 8:2"),
    ]
    model_boxes = [
        (left_branch, "ResNet / EfficientNet\n六分类"),
        (right_branch, "YOLOv8n\n检测与验证"),
    ]
    for x, text in branch_boxes:
        rounded(ax, x, levels[1], half, 9.4, text, fc=BLUE_TINT, ec=BLUE, fs=7.4)
        arrow(ax, left + width / 2, levels[0], x + half / 2, levels[1] + 9.4, color=BLUE)
    for x, text in model_boxes:
        rounded(ax, x, levels[2], half, 9.4, text, fc="white", ec=BLUE, fs=7.4)
        arrow(ax, x + half / 2, levels[1], x + half / 2, levels[2] + 9.4, color=BLUE)

    rounded(ax, left, levels[3], width, 9.4,
            "评价口径\n准确率 / macroF1 · mAP50 / mAP50-95",
            fc=BLUE_TINT, ec=BLUE, fs=7.6)
    for x in (left_branch, right_branch):
        arrow(ax, x + half / 2, levels[2], x + half / 2, levels[3] + 9.4, color=BLUE)

    rounded(ax, left, levels[4], width, 9.4,
            "视觉质检 / Web\n分类 · 检测 · 标注图", fc="white", ec=BLUE, fs=7.8)
    rounded(ax, left, levels[5], width, 9.4,
            "台账 · 查询 · 预警\nSQLite · TF-IDF RAG", fc=BLUE_TINT, ec=BLUE, fs=7.8)
    arrow(ax, left + width / 2, levels[3], left + width / 2, levels[4] + 9.4, color=BLUE)
    arrow(ax, left + width / 2, levels[4], left + width / 2, levels[5] + 9.4, color=BLUE)

    save(fig, "fig_2_2_data_flow.png")


# ---------------------------------------------------------------- 图 6-1 标注分布
def figure_dataset_split() -> None:
    train = np.array([547, 833, 694, 345, 507, 434])
    valid = np.array([141, 178, 187, 87, 121, 114])
    assert train.sum() == 3360 and valid.sum() == 828

    x = np.arange(len(CLASS_CN))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    bars_train = ax.bar(x - width / 2, train, width, label=f"训练集（{train.sum()} 个框）",
                        color=GREEN, edgecolor="white", linewidth=0.6)
    bars_valid = ax.bar(x + width / 2, valid, width, label=f"验证集（{valid.sum()} 个框）",
                        color=BLUE, edgecolor="white", linewidth=0.6)

    for bars in (bars_train, bars_valid):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 14, f"{int(height)}",
                    ha="center", va="bottom", fontsize=7.8, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{cn}\n{en}" for cn, en in zip(CLASS_CN, CLASS_EN)], fontsize=8.6)
    ax.set_ylabel("标注框数量 / 个")
    ax.set_ylim(0, 940)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.12), fontsize=8.8)
    fig.tight_layout()
    save(fig, "fig_6_1_dataset_split.png")


# ---------------------------------------------------------------- 图 6-3 训练曲线
def figure_training_curves() -> None:
    table = pd.read_csv(ROOT / "results" / "week2_3" / "yolo" / "train" / "results.csv")
    table.columns = [column.strip() for column in table.columns]
    epoch = table["epoch"].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.4), sharex=True)
    panels = [
        (axes[0, 0], "(a) 训练损失", [
            ("train/box_loss", "box_loss", GREEN),
            ("train/cls_loss", "cls_loss", BLUE),
            ("train/dfl_loss", "dfl_loss", AMBER),
        ], "损失值"),
        (axes[0, 1], "(b) 验证损失", [
            ("val/box_loss", "box_loss", GREEN),
            ("val/cls_loss", "cls_loss", BLUE),
            ("val/dfl_loss", "dfl_loss", AMBER),
        ], "损失值"),
        (axes[1, 0], "(c) 精确率与召回率", [
            ("metrics/precision(B)", "Precision", GREEN),
            ("metrics/recall(B)", "Recall", BLUE),
        ], "指标值"),
        (axes[1, 1], "(d) 平均精度", [
            ("metrics/mAP50(B)", "mAP50", GREEN),
            ("metrics/mAP50-95(B)", "mAP50-95", BLUE),
        ], "指标值"),
    ]

    for ax, title, series, ylabel in panels:
        for column, label, color in series:
            ax.plot(epoch, table[column], color=color, lw=1.5, marker="o",
                    markersize=2.3, markevery=1, label=label)
        ax.set_title(title, loc="left", fontsize=10.0, color=INK, pad=8)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("训练轮次 / epoch")
        ax.set_xlim(0.5, 30.5)
        ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
        ax.grid(color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(frameon=False, fontsize=8.0, loc="best")

    axes[0, 0].set_ylim(0.9, 3.25)
    axes[0, 1].set_ylim(1.0, 3.25)
    axes[1, 0].set_ylim(0.15, 0.85)
    axes[1, 1].set_ylim(0.05, 0.85)

    axes[1, 0].text(
        0.97, 0.05,
        f"末轮 P = {table['metrics/precision(B)'].iloc[-1]:.3f}\n"
        f"末轮 R = {table['metrics/recall(B)'].iloc[-1]:.3f}",
        transform=axes[1, 0].transAxes, ha="right", va="bottom",
        fontsize=7.6, color=MUTED, linespacing=1.35,
    )
    axes[1, 1].text(
        0.97, 0.05,
        f"末轮 mAP50 = {table['metrics/mAP50(B)'].iloc[-1]:.3f}\n"
        f"末轮 mAP50-95 = {table['metrics/mAP50-95(B)'].iloc[-1]:.3f}",
        transform=axes[1, 1].transAxes, ha="right", va="bottom",
        fontsize=7.6, color=MUTED, linespacing=1.35,
    )
    fig.tight_layout(h_pad=1.6, w_pad=1.6)
    save(fig, "fig_6_3_training_curves.png")


# ---------------------------------------------------------------- 图 6-2 训练流程
def figure_yolo_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 6.8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    left_x, right_x, width = 4.0, 59.0, 36.0
    rows = [77.0, 59.0, 41.0, 23.0]

    rounded(ax, left_x, 90.0, width, 7.2, "数据准备", fc=GREEN, ec=GREEN,
            tc="white", fs=10.2, weight="bold")
    rounded(ax, right_x, 90.0, width, 7.2, "训练、验证与落盘", fc=BLUE, ec=BLUE,
            tc="white", fs=10.2, weight="bold")

    left_steps = [
        "NEU-DET 检测版\nVOC XML + 200×200 灰度图",
        "convert_voc_to_yolo.py\n坐标归一化 · 类别映射",
        "YOLO 数据集\nimages/{train,valid} · labels",
        "dataset.yaml\n按类别分层 8:2 · seed=42",
    ]
    right_steps = [
        "yolov8n.pt\nCOCO 预训练权重",
        "微调训练\n30 epoch · 640 px · batch 16",
        "best.pt\n验证集最优权重",
        "验证与预测\nP / R / mAP · conf=0.25",
    ]

    for index, (y, text) in enumerate(zip(rows, left_steps)):
        rounded(ax, left_x, y, width, 11.2, text, fc=GREEN_TINT, ec=GREEN, fs=8.4)
        if index < len(rows) - 1:
            arrow(ax, left_x + width / 2, y, left_x + width / 2, y - 6.8, color=GREEN)

    for index, (y, text) in enumerate(zip(rows, right_steps)):
        rounded(ax, right_x, y, width, 11.2, text, fc=BLUE_TINT, ec=BLUE, fs=8.4)
        if index < len(rows) - 1:
            arrow(ax, right_x + width / 2, y, right_x + width / 2, y - 6.8, color=BLUE)

    ax.plot([left_x + width, 56.0], [28.6, 28.6], color=MUTED, lw=1.2, zorder=3)
    ax.plot([56.0, 56.0], [28.6, 64.6], color=MUTED, lw=1.2, zorder=3)
    arrow(ax, 56.0, 64.6, right_x, 64.6, color=MUTED, lw=1.2, z=4)
    ax.text(47.6, 46.0, "数据集\n就绪", ha="center", va="center",
            color=MUTED, fontsize=8.0, linespacing=1.3)

    rounded(ax, 4.0, 4.0, 92.0, 9.6,
            "落盘产物：yolo_results.csv · results.png · results.csv · PR/F1 曲线 · 混淆矩阵 · predict/*.jpg · best.pt",
            fc=GRAY_TINT, ec=LINE, fs=8.2)
    arrow(ax, right_x + width / 2, 23.0, right_x + width / 2, 13.6, color=MUTED, ls="--")
    save(fig, "fig_6_2_yolo_pipeline.png")


# ---------------------------------------------------------------- 图 7-3 分类别指标
def figure_per_class_metrics() -> None:
    labels = ["crazing", "inclusion", "patches", "pitted_surface", "rolled_in_scale", "scratches", "all"]
    precision = [0.617, 0.709, 0.801, 0.767, 0.608, 0.848, 0.725]
    recall = [0.320, 0.831, 0.936, 0.679, 0.636, 0.877, 0.713]
    map50 = [0.418, 0.873, 0.947, 0.786, 0.666, 0.931, 0.770]
    map5095 = [0.137, 0.509, 0.636, 0.444, 0.343, 0.547, 0.436]
    counts = [141, 178, 187, 87, 121, 114, 828]
    display = ["龟裂", "夹杂", "斑块", "麻点", "氧化铁皮压入", "划痕", "全部"]
    y = np.arange(len(labels))
    height = 0.19
    offsets = [-1.5, -0.5, 0.5, 1.5]

    fig, ax = plt.subplots(figsize=(9.8, 6.6))
    series = [
        ("Precision", precision, GREEN),
        ("Recall", recall, BLUE),
        ("mAP50", map50, AMBER),
        ("mAP50-95", map5095, PURPLE),
    ]
    for (name, values, color), offset in zip(series, offsets):
        bars = ax.barh(y + offset * height, values, height=height, color=color,
                       edgecolor="white", linewidth=0.4, label=name)
        for bar, value in zip(bars, values):
            ax.text(value + 0.008, bar.get_y() + bar.get_height() / 2, f"{value:.3f}",
                    va="center", ha="left", fontsize=7.1, color=INK)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{name}\n(n={count})" for name, count in zip(display, counts)], fontsize=8.6)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.06)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("指标值")
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.09), fontsize=8.4)
    fig.tight_layout()
    save(fig, "fig_7_3_per_class_metrics.png")


# ---------------------------------------------------------------- 图 7-1 PR / 图 7-2 F1
def figure_curves() -> None:
    for source, target in [
        ("BoxPR_curve.png", "fig_7_1_pr_curve.png"),
        ("BoxF1_curve.png", "fig_7_2_f1_curve.png"),
    ]:
        image = trim_white(ROOT / "results" / "week2_3" / "yolo" / "train" / source)
        image.save(OUT / target, dpi=(300, 300))
        print(f"wrote {(OUT / target).relative_to(ROOT)}")


# ---------------------------------------------------------------- 图 7-4 混淆矩阵
def figure_confusion_matrix() -> None:
    matrix_path = OUT / "confusion_matrix_conf025.csv"
    if not matrix_path.exists():
        source = ROOT / "results" / "week2_3" / "yolo" / "train" / "confusion_matrix_normalized.png"
        if source.exists():
            trim_white(source).save(OUT / "fig_7_4_confusion_matrix.png", dpi=(300, 300))
            print(f"wrote {(OUT / 'fig_7_4_confusion_matrix.png').relative_to(ROOT)}")
        return

    matrix = np.loadtxt(matrix_path, delimiter=",")
    names = CLASS_CN + ["背景"]
    column_sum = matrix.sum(axis=0, keepdims=True)
    normalized = np.divide(matrix, column_sum, out=np.zeros_like(matrix), where=column_sum > 0)

    cmap = LinearSegmentedColormap.from_list(
        "report_green", ["#ffffff", "#eaf5ee", "#b9dfc8", "#72b68e", "#2f6f4c"]
    )
    fig, ax = plt.subplots(figsize=(8.4, 7.0))
    image = ax.imshow(normalized, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8.4)
    ax.set_yticklabels(names, fontsize=8.4)
    ax.set_xlabel("真实类别")
    ax.set_ylabel("预测类别")

    for row in range(normalized.shape[0]):
        for column in range(normalized.shape[1]):
            value = normalized[row, column]
            if value < 0.005:
                continue
            ax.text(column, row, f"{value:.2f}", ha="center", va="center",
                    fontsize=7.2, color="white" if value > 0.55 else INK)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("按真实类别归一化的比例", fontsize=8.4)
    colorbar.ax.tick_params(labelsize=7.6)
    fig.tight_layout()
    save(fig, "fig_7_4_confusion_matrix.png")


# ---------------------------------------------------------------- 图 7-5 验证对比
def _draw_detection(draw: ImageDraw.ImageDraw, box, color, text, font) -> None:
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    text_box = draw.textbbox((0, 0), text, font=font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    label_y = y1 - text_height - 8 if y1 - text_height - 8 >= 0 else y1 + 2
    draw.rectangle([x1, label_y, x1 + text_width + 10, label_y + text_height + 8], fill=color)
    draw.text((x1 + 5, label_y + 2), text, fill="white", font=font)


def figure_validation_compare() -> None:
    try:
        from ultralytics import YOLO
    except Exception as error:  # noqa: BLE001
        print(f"skip clean validation comparison: {error}")
        return

    cases = [
        ("crazing_200.jpg", "龟裂（crazing）"),
        ("pitted_surface_102.jpg", "麻点（pitted_surface）"),
        ("rolled-in_scale_105.jpg", "氧化铁皮压入（rolled_in_scale）"),
    ]
    image_dir = ROOT / "data" / "NEU-DET-yolo" / "images" / "valid"
    label_dir = ROOT / "data" / "NEU-DET-yolo" / "labels" / "valid"
    weights = ROOT / "results" / "week2_3" / "yolo" / "train" / "weights" / "best.pt"
    if not weights.exists():
        print("skip clean validation comparison: best.pt not found")
        return

    model = YOLO(weights)
    sources = [str(image_dir / filename) for filename, _ in cases]
    results = model.predict(sources, conf=0.25, iou=0.45, verbose=False)

    tile, gap, left_margin, top_margin = 500, 20, 108, 78
    width = left_margin + len(cases) * tile + (len(cases) - 1) * gap + 20
    height = top_margin + 2 * tile + gap + 20
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(font_path(), 19)
    font_title = ImageFont.truetype(font_path(), 21)
    font_row = ImageFont.truetype(font_path(), 20)

    record = []
    for index, ((filename, title), result) in enumerate(zip(cases, results)):
        image = Image.open(image_dir / filename).convert("RGB").resize((tile, tile), Image.LANCZOS)
        scale = tile / result.orig_shape[1]
        x = left_margin + index * (tile + gap)

        gt = []
        label_path = label_dir / f"{Path(filename).stem}.txt"
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            class_id, cx, cy, w, h = line.split()
            cx, cy, w, h = map(float, (cx, cy, w, h))
            box = [
                (cx - w / 2) * tile, (cy - h / 2) * tile,
                (cx + w / 2) * tile, (cy + h / 2) * tile,
            ]
            gt.append({"class": int(class_id), "box": [round(value, 1) for value in box]})

        preds = []
        if result.boxes is not None:
            for cls_id, confidence, box in zip(
                result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist()
            ):
                preds.append({
                    "class": int(cls_id),
                    "confidence": round(float(confidence), 4),
                    "box": [round(float(value) * scale, 1) for value in box],
                })
        record.append({"image": filename, "ground_truth": gt, "prediction": preds})

        gt_image = image.copy()
        gt_draw = ImageDraw.Draw(gt_image)
        for item in gt:
            _draw_detection(gt_draw, item["box"], GREEN, CLASS_CN[item["class"]], font)

        pred_image = image.copy()
        pred_draw = ImageDraw.Draw(pred_image)
        for item in preds:
            label = f"{CLASS_CN[item['class']]} {item['confidence']:.2f}"
            _draw_detection(pred_draw, item["box"], BLUE, label, font)

        for row_index, tile_image in enumerate((gt_image, pred_image)):
            y = top_margin + row_index * (tile + gap)
            canvas.paste(tile_image, (x, y))
            draw.rectangle([x - 1, y - 1, x + tile, y + tile], outline=LINE, width=2)

        draw.multiline_text((x + tile / 2, 24), title.replace("（", "\n（"),
                            anchor="ma", fill=INK, font=font_title, align="center", spacing=2)

    draw.multiline_text((left_margin / 2, top_margin + tile / 2), "人工\n标注",
                        anchor="mm", fill=INK, font=font_row, align="center", spacing=6)
    draw.multiline_text((left_margin / 2, top_margin + tile + gap + tile / 2), "模型\n预测",
                        anchor="mm", fill=INK, font=font_row, align="center", spacing=6)

    canvas.save(OUT / "fig_7_5_validation_compare.jpg", quality=94)
    (OUT / "validation_examples.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {(OUT / 'fig_7_5_validation_compare.jpg').relative_to(ROOT)}")


# ---------------------------------------------------------------- 图 7-6 龟裂样例
def figure_crazing_cases() -> None:
    cases = [
        ("crazing_10.jpg", "（a）crazing_10\n预测 crazing，置信度 0.28"),
        ("crazing_103.jpg", "（b）crazing_103\n预测 crazing，置信度 0.27"),
        ("crazing_104.jpg", "（c）crazing_104\n误判为 patches，置信度 0.51"),
    ]
    size, gap, margin, top = 420, 24, 16, 66
    canvas = Image.new("RGB", (margin * 2 + size * 3 + gap * 2, top + size + margin), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(font_path(), 20)

    for index, (filename, caption) in enumerate(cases):
        image = Image.open(ROOT / "results" / "week2_3" / "yolo" / "predict" / filename).convert("RGB")
        image = image.resize((size, size), Image.LANCZOS)
        x = margin + index * (size + gap)
        canvas.paste(image, (x, top))
        draw.rectangle([x - 1, top - 1, x + size, top + size], outline=LINE, width=2)
        draw.multiline_text((x, 12), caption, fill=INK, font=font, spacing=4)

    canvas.save(OUT / "fig_7_6_crazing_cases.jpg", quality=94)
    print(f"wrote {(OUT / 'fig_7_6_crazing_cases.jpg').relative_to(ROOT)}")


def main() -> None:
    figure_neu_samples()
    figure_architecture()
    figure_data_flow()
    figure_dataset_split()
    figure_training_curves()
    figure_yolo_pipeline()
    figure_per_class_metrics()
    figure_curves()
    figure_confusion_matrix()
    figure_validation_compare()
    figure_crazing_cases()


if __name__ == "__main__":
    main()