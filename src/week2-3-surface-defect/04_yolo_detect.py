"""第3周·第2天：YOLO 缺陷检测（ultralytics YOLOv8n 微调）。

输入要求 YOLO 布局（images/ + labels/）；VOC 版先运行
convert_voc_to_yolo.py。预训练权重下载失败时自动退回从零训练。
产出：训练/验证指标（mAP50 等）、若干预测可视化图。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.neu import detect_layout, make_dataset_yaml


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--neu-dir", default="data/NEU-DET-yolo", help="YOLO 布局根目录")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--model", default="yolov8n", help="yolov8n / yolov8s ...")
    ap.add_argument("--from-scratch", action="store_true", help="不从预训练权重微调")
    ap.add_argument("--out", default="results/week2_3")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from ultralytics import YOLO

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    neu_dir = Path(args.neu_dir)
    layout = detect_layout(neu_dir)
    if layout == "voc":
        raise RuntimeError(
            f"{neu_dir} 是 VOC 布局。请先运行：\n"
            f"  python3 src/week2-3-surface-defect/convert_voc_to_yolo.py "
            f"--neu-dir {neu_dir} --out data/NEU-DET-yolo")
    if layout != "yolo":
        raise RuntimeError(
            f"{neu_dir} 不是 YOLO 布局（images/ + labels/）。"
            "检测训练需要带边界框标注的检测版 NEU-DET，详见 data/README.md。")
    if not (neu_dir / "images" / "valid").is_dir():
        raise RuntimeError("缺少 images/valid 划分，YOLO 训练需要 train/valid 两套划分。")

    yaml_path = make_dataset_yaml(neu_dir, out / "yolo_dataset.yaml")
    print(f"数据配置：{yaml_path}")

    if args.from_scratch:
        model = YOLO(f"{args.model}.yaml")
        print("从零训练。")
    else:
        try:
            model = YOLO(f"{args.model}.pt")  # 需联网下载预训练权重
            print("加载预训练权重进行微调。")
        except Exception as e:
            print(f"[警告] 预训练权重下载失败（{type(e).__name__}），退回从零训练。")
            model = YOLO(f"{args.model}.yaml")

    model.train(
        data=str(yaml_path), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        project=str((out / "yolo").resolve()), name="train", exist_ok=True, seed=args.seed,
        plots=True,
    )
    metrics = model.val(data=str(yaml_path))
    print(f"mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}  "
          f"mP={metrics.box.mp:.4f}  mR={metrics.box.mr:.4f}")

    import pandas as pd
    pd.DataFrame([{"model": args.model, "epochs": args.epochs, "imgsz": args.imgsz,
                   "mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map),
                   "mP": float(metrics.box.mp), "mR": float(metrics.box.mr)}]).to_csv(
        out / "yolo_results.csv", index=False, encoding="utf-8-sig")

    # 预测可视化
    val_imgs = sorted((neu_dir / "images" / "valid").glob("*"))[:3]
    if val_imgs:
        model.predict(source=[str(p) for p in val_imgs], save=True, conf=0.25,
                      project=str((out / "yolo").resolve()), name="predict", exist_ok=True)
        print(f"预测可视化已保存：{out / 'yolo' / 'predict'}")
    print(f"训练产物目录：{out / 'yolo'}")


if __name__ == "__main__":
    main()
