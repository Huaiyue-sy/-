"""合成样例数据生成器：仅用于冒烟测试，验证代码链路可运行。

生成内容：
- XJTU-SY 布局的轴承振动 CSV（故障特征频率冲击 + 退化趋势）
- NEU-DET 分类布局（每类文件夹）与 YOLO 检测布局（images/ + labels/）

【重要】这些合成数据的质量不代表真实数据集，产出的任何指标都不得
写入实训手册或报告，只用于确认脚本无报错。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.bearing import FS_HZ, BearingGeometry, condition_rpm
from common.neu import CLASSES

FAULT_MODES = ("BPFO", "BPFI", "BSF", "none")


def _simulate_bearing_signal(
    n_samples: int, fs: float, fr: float, geometry: BearingGeometry,
    fault: str, progress: float, rng: np.random.Generator,
) -> np.ndarray:
    """按退化进度合成一段含故障冲击的振动信号。

    progress ∈ [0,1]：越大冲击幅值越大，模拟轴承从健康走向失效。
    """
    t = np.arange(n_samples) / fs
    sig = 0.10 * np.sin(2 * np.pi * fr * t) + 0.05 * np.sin(4 * np.pi * fr * t)
    sig += rng.normal(0, 0.03, n_samples)

    fault_amp = 0.0
    if fault != "none" and progress > 0.4:
        fault_amp = 0.4 * ((progress - 0.4) / 0.6) ** 2
    if fault_amp > 0:
        f_c = geometry.fault_frequencies(fr)[fault]
        f_res, tau = 3000.0, 0.004
        period = 1.0 / f_c
        k = 0
        while k * period < t[-1]:
            t0 = k * period * (1 + rng.uniform(-0.02, 0.02))
            mask = t >= t0
            if mask.any():
                dt = t[mask] - t0
                sig[mask] += fault_amp * np.exp(-dt / tau) * np.sin(2 * np.pi * f_res * dt)
            k += 1
    # 整体退化：噪声随进度缓慢增大
    sig += rng.normal(0, 0.01 + 0.02 * progress, n_samples)
    return sig


def gen_bearing(
    out_dir: str | Path,
    n_samples: int = 4096,
    n_files: int = 12,
    fs: float = FS_HZ,
    seed: int = 7,
) -> Path:
    """生成 XJTU-SY 布局的小型合成数据集（2 工况 × 2 轴承）。"""
    out = Path(out_dir)
    rng = np.random.default_rng(seed)
    geometry = BearingGeometry()
    plan = {
        "35Hz12kN": {"Bearing1_1": "BPFO", "Bearing1_2": "BPFI"},
        "37.5Hz11kN": {"Bearing2_1": "BSF", "Bearing2_2": "none"},
    }
    for cond, bearings in plan.items():
        fr = condition_rpm(cond) / 60.0
        for name, fault in bearings.items():
            d = out / cond / name
            d.mkdir(parents=True, exist_ok=True)
            for i in range(1, n_files + 1):
                progress = (i - 1) / max(n_files - 1, 1)
                sig = _simulate_bearing_signal(n_samples, fs, fr, geometry, fault, progress, rng)
                horiz = sig * 0.7 + rng.normal(0, 0.02, n_samples)
                vert = sig + rng.normal(0, 0.02, n_samples)
                data = np.column_stack([horiz, vert])
                np.savetxt(d / f"{i}.csv", data, delimiter=",", fmt="%.6f")
    print(f"[synthesize] 轴承合成数据 -> {out}（{sum(len(v) for v in plan.values())} 个轴承 × {n_files} 文件）")
    return out


def _draw_defect(draw, size: int, cls: str, rng: np.random.Generator) -> tuple[int, int, int, int]:
    """在灰度钢面纹理上画一类缺陷，返回缺陷区域 bbox (x1,y1,x2,y2)。"""
    v = rng.integers(40, 90)

    def crack(points):
        for a, b in zip(points[:-1], points[1:]):
            draw.line([a, b], fill=int(v), width=2)
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return min(xs), min(ys), max(xs), max(ys)

    if cls == "crazing":
        pts = [(rng.integers(10, size - 10), rng.integers(10, size - 10))]
        for _ in range(8):
            last = pts[-1]
            pts.append((int(last[0] + rng.integers(-12, 13)), int(last[1] + rng.integers(-12, 13))))
        return crack([(int(x), int(y)) for x, y in pts])
    if cls == "inclusion":
        x, y = rng.integers(20, size - 20, 2)
        r = int(rng.integers(6, 12))
        draw.ellipse([x - r, y - r, x + r, y + r], fill=int(v))
        return x - r, y - r, x + r, y + r
    if cls == "patches":
        x, y = rng.integers(24, size - 24, 2)
        rx, ry = int(rng.integers(14, 22)), int(rng.integers(12, 18))
        draw.ellipse([x - rx, y - ry, x + rx, y + ry], fill=int(v + 30))
        return x - rx, y - ry, x + rx, y + ry
    if cls == "pitted_surface":
        xs, ys = [], []
        cx, cy = rng.integers(30, size - 30, 2)
        for _ in range(24):
            x = int(cx + rng.integers(-18, 19))
            y = int(cy + rng.integers(-18, 19))
            draw.point((x, y), fill=int(v))
            draw.point((x + 1, y), fill=int(v))
            xs += [x, x + 1]
            ys += [y]
        return min(xs), min(ys), max(xs), max(ys)
    if cls == "rolled_in_scale":
        y = int(rng.integers(18, size - 18))
        h = int(rng.integers(6, 12))
        draw.rectangle([10, y, size - 10, y + h], fill=int(v + 20))
        return 10, y, size - 10, y + h
    # scratches
    x0 = int(rng.integers(5, size - 5))
    x1 = int(min(size - 5, x0 + rng.integers(40, size - 20)))
    y0 = int(rng.integers(5, size - 5))
    y1 = int(np.clip(y0 + rng.integers(-25, 26), 5, size - 5))
    return crack([(x0, y0), (x1, y1)])


def gen_neu(
    out_root: str | Path, per_class: int = 8, size: int = 96, seed: int = 11,
) -> tuple[Path, Path]:
    """生成 NEU-DET 分类布局与 YOLO 检测布局的小型合成数据集。"""
    from PIL import Image, ImageDraw

    out_root = Path(out_root)
    rng = np.random.default_rng(seed)

    cls_root = out_root / "NEU-DET"
    det_root = out_root / "NEU-DET-det"
    margin = 3

    counter = 0
    for cls_idx, cls in enumerate(CLASSES):
        for j in range(per_class):
            img = Image.new("L", (size, size), 125)
            arr = np.asarray(img).astype(np.int16)
            arr += rng.integers(-12, 13, (size, size), dtype=np.int16)
            arr += rng.integers(-6, 7, (size, 1), dtype=np.int16)  # 水平轧制纹理
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            draw = ImageDraw.Draw(img)
            x1, y1, x2, y2 = _draw_defect(draw, size, cls, rng)
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(size, x2 + margin)
            y2 = min(size, y2 + margin)

            d = cls_root / cls
            d.mkdir(parents=True, exist_ok=True)
            img.save(d / f"{cls}_{j:03d}.png")

            split = "train" if j % 4 != 3 else "valid"
            img_d = det_root / "images" / split
            lbl_d = det_root / "labels" / split
            img_d.mkdir(parents=True, exist_ok=True)
            lbl_d.mkdir(parents=True, exist_ok=True)
            name = f"{counter:04d}"
            img.save(img_d / f"{name}.png")
            cx = (x1 + x2) / 2 / size
            cy = (y1 + y2) / 2 / size
            bw = (x2 - x1) / size
            bh = (y2 - y1) / size
            (lbl_d / f"{name}.txt").write_text(
                f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")
            counter += 1
    print(f"[synthesize] NEU 分类布局 -> {cls_root}；YOLO 检测布局 -> {det_root}（{len(CLASSES)} 类 × {per_class} 张）")
    return cls_root, det_root


def main() -> None:
    ap = argparse.ArgumentParser(description="生成冒烟测试用合成数据")
    ap.add_argument("--out", default="data/_smoke", help="输出根目录")
    ap.add_argument("--bearing", action="store_true")
    ap.add_argument("--neu", action="store_true")
    ap.add_argument("--per-class", type=int, default=8)
    args = ap.parse_args()
    do_all = not (args.bearing or args.neu)
    if args.bearing or do_all:
        gen_bearing(Path(args.out) / "XJTU-SY_Bearing_Datasets")
    if args.neu or do_all:
        gen_neu(Path(args.out), per_class=args.per_class)


if __name__ == "__main__":
    main()
