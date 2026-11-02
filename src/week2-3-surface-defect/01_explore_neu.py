"""第2周·第3天：NEU-DET 数据探索与标注格式熟悉。

自动识别布局（分类/YOLO/VOC），产出：类别分布、图像统计、每类示例图、
检测标注的边界框可视化与统计。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.neu import (
    CLASSES, CLASS_CN, _norm_class, detect_layout, list_classification_images, voc_dirs,
)
from common.plotting import save_fig, setup_chinese_font

import matplotlib.pyplot as plt
from PIL import Image


def explore_classification(neu_dir: Path, out: Path) -> None:
    items = list_classification_images(neu_dir)
    counts = pd.Series([c for _, c in items]).value_counts().sort_index()
    print("类别分布：", {f"{CLASSES[i]}({CLASS_CN[CLASSES[i]]})": int(n)
                         for i, n in counts.items()})

    fig, ax = plt.subplots(figsize=(8, 4))
    names = [f"{CLASSES[i]}\n{CLASS_CN[CLASSES[i]]}" for i in counts.index]
    ax.bar(names, counts.values, color="steelblue")
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set_title("NEU-DET 类别分布")
    ax.set_ylabel("图片数")
    save_fig(fig, out / "neu_class_distribution.png")

    sizes = [Image.open(p).size for p, _ in items[:200]]
    uniq = sorted(set(sizes))
    print(f"图像尺寸抽样 {len(sizes)} 张：{uniq[:5]}{' ...' if len(uniq) > 5 else ''}")

    fig, axes = plt.subplots(2, 3, figsize=(10, 6.6))
    for ax, cls in zip(axes.flat, CLASSES):
        sample = next((p for p, c in items if c == CLASSES.index(cls)), None)
        ax.imshow(Image.open(sample), cmap="gray")
        ax.set_title(f"{cls}（{CLASS_CN[cls]}）", fontsize=9)
        ax.axis("off")
    save_fig(fig, out / "neu_samples.png")


def explore_yolo(neu_dir: Path, out: Path) -> None:
    rows = []
    for split in ("train", "valid", "test"):
        img_dir = neu_dir / "images" / split
        lbl_dir = neu_dir / "labels" / split
        if not img_dir.is_dir():
            continue
        n_img = len(list(img_dir.glob("*")))
        n_lbl = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.is_dir() else 0
        boxes = 0
        for lbl in (lbl_dir.glob("*.txt") if lbl_dir.is_dir() else []):
            boxes += sum(1 for line in lbl.read_text().splitlines() if line.strip())
        rows.append({"split": split, "images": n_img, "label_files": n_lbl, "boxes": boxes})
        print(f"  {split}: {n_img} 图 / {n_lbl} 标注 / {boxes} 边界框")

    pd.DataFrame(rows).to_csv(out / "neu_det_splits.csv", index=False, encoding="utf-8-sig")

    box_sizes: list[float] = []
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    ax_list = list(axes.flat)
    shown = 0
    for split in ("train", "valid", "test"):
        for img_path in sorted((neu_dir / "images" / split).glob("*"))[:]:
            lbl = neu_dir / "labels" / split / (img_path.stem + ".txt")
            if not lbl.exists():
                continue
            img = Image.open(img_path)
            w, h = img.size
            ax = ax_list[shown]
            ax.imshow(img, cmap="gray")
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                c, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
                x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
                rect = plt.Rectangle((x1, y1), bw * w, bh * h,
                                     fill=False, color="red", lw=1.5)
                ax.add_patch(rect)
                ax.text(x1, max(y1 - 3, 2), CLASSES[c], color="red", fontsize=7)
                box_sizes.append(bw * bh)
            ax.set_title(f"{split}/{img_path.name}", fontsize=8)
            ax.axis("off")
            shown += 1
            if shown >= 6:
                break
        if shown >= 6:
            break
    for ax in ax_list[shown:]:
        ax.axis("off")
    save_fig(fig, out / "neu_bbox_examples.png")
    if box_sizes:
        print(f"边界框面积占比统计：均值 {np.mean(box_sizes):.3f}，"
              f"中位数 {np.median(box_sizes):.3f}（相对整图）")


def explore_voc(neu_dir: Path, out: Path) -> None:
    import xml.etree.ElementTree as ET

    dirs = voc_dirs(neu_dir)
    if not dirs:
        raise FileNotFoundError(f"{neu_dir} 不是 VOC 布局。")
    ann_dir, _ = dirs
    xmls = list(ann_dir.glob("*.xml"))
    print(f"VOC 布局：{len(xmls)} 个 XML 标注")
    per_class = {c: 0 for c in CLASSES}
    for x in xmls:
        for obj in ET.parse(x).iter("object"):
            name = _norm_class(obj.findtext("name", ""))
            if name in per_class:
                per_class[name] += 1
    print("各缺陷类别边界框数：", per_class)
    pd.DataFrame([{"class": k, "boxes": v} for k, v in per_class.items()]).to_csv(
        out / "neu_voc_boxes.csv", index=False, encoding="utf-8-sig")
    print("提示：检测训练需要 YOLO 布局，先运行 convert_voc_to_yolo.py。")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--neu-dir", default="data/NEU-DET")
    ap.add_argument("--out", default="results/week2_3")
    args = ap.parse_args()

    setup_chinese_font()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    layout = detect_layout(args.neu_dir)
    print(f"识别到布局：{layout}")
    neu_dir = Path(args.neu_dir)
    if layout in ("classification", "voc"):
        explore_classification(neu_dir, out)
    if layout == "yolo":
        explore_yolo(neu_dir, out)
    if layout == "voc":
        explore_voc(neu_dir, out)
    print(f"图表输出目录：{out}/")


if __name__ == "__main__":
    main()
