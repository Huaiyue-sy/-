"""第2-3周辅助：VOC 标注 → YOLO 布局转换（检测训练前置步骤）。

NEU-DET 检测版常以 VOC 发布（ANNOTATIONS/IMAGES 或 Annotations/JPEGImages）。
本脚本转换并按 --val-ratio 做**按类别分层**的 train/valid 划分，输出 YOLO 布局。
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.neu import _norm_class, voc_to_yolo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--neu-dir", required=True, help="VOC 布局的 NEU-DET 根目录")
    ap.add_argument("--out", default="data/NEU-DET-yolo", help="输出的 YOLO 布局目录")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.neu_dir)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"{out} 已存在，如需重跑请先删除该目录。")

    n = voc_to_yolo(src, out, split="train")
    print(f"转换 {n} 张图（train）")

    val_ratio = args.val_ratio
    if val_ratio > 0:
        by_cls: dict[str, list[Path]] = {}
        for img in sorted((out / "images" / "train").iterdir()):
            cls = _norm_class(img.stem.rsplit("_", 1)[0])
            by_cls.setdefault(cls, []).append(img)
        rng = random.Random(args.seed)
        val_imgs: list[Path] = []
        for cls in sorted(by_cls):
            imgs = by_cls[cls]
            rng.shuffle(imgs)
            n_val_cls = int(round(len(imgs) * val_ratio))
            val_imgs.extend(imgs[:n_val_cls])
        for d in ("images", "labels"):
            (out / d / "valid").mkdir(parents=True, exist_ok=True)
        for img in val_imgs:
            (out / "labels" / "train" / (img.stem + ".txt")).rename(
                out / "labels" / "valid" / (img.stem + ".txt"))
            shutil.move(str(img), out / "images" / "valid" / img.name)
        print(f"划分 valid：{len(val_imgs)} 张（按类别分层，比例 {val_ratio:.0%}）")
    print(f"YOLO 布局输出：{out}")


if __name__ == "__main__":
    main()
