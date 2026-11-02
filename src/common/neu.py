"""NEU-DET 金属表面瑕疵数据集：布局识别、分类图列表、VOC→YOLO 标注转换。"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

CLASSES = (
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled_in_scale", "scratches",
)
CLASS_CN = {
    "crazing": "龟裂", "inclusion": "夹杂", "patches": "斑块",
    "pitted_surface": "麻点", "rolled_in_scale": "氧化铁皮压入", "scratches": "划痕",
}


def _norm_class(name: str) -> str:
    s = re.sub(r"[\s\-]+", "_", name.strip().lower())
    return s.rstrip("_")


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def voc_dirs(root: str | Path) -> tuple[Path, Path] | None:
    """定位 VOC 布局的 (标注目录, 图片目录)，兼容官方 NEU-DET 的
    ANNOTATIONS/IMAGES 与标准 VOC 的 Annotations/JPEGImages（大小写不敏感）。"""
    root = Path(root)
    if not root.is_dir():
        return None
    by_lower = {p.name.lower(): p for p in root.iterdir() if p.is_dir()}
    ann = by_lower.get("annotations")
    img = by_lower.get("jpegimages") or by_lower.get("images")
    if ann and img and ann != img:
        return ann, img
    return None


def detect_layout(neu_dir: str | Path) -> str:
    """识别数据集布局：classification / yolo / voc。"""
    root = Path(neu_dir)
    if not root.exists():
        raise FileNotFoundError(f"未找到 {root}，请先下载 NEU-DET（见 data/README.md）。")
    if voc_dirs(root):
        return "voc"
    if (root / "images").is_dir() and any((root / "images").iterdir()):
        sub = next(iter((root / "images").iterdir()))
        if (root / "labels").is_dir() and (root / "images" / sub.name).is_dir():
            return "yolo"
    folders = {_norm_class(p.name) for p in root.iterdir() if p.is_dir()}
    if folders & set(CLASSES):
        return "classification"
    raise RuntimeError(
        f"{root} 无法识别布局。期望三种布局之一（见 data/README.md）：\n"
        "  A 分类布局：每类一个文件夹；B YOLO 布局：images/ + labels/；\n"
        "  C VOC 布局：Annotations/JPEGImages/ 或官方 NEU-DET 的 ANNOTATIONS/IMAGES"
    )


def list_classification_images(neu_dir: str | Path) -> list[tuple[Path, int]]:
    """扫描分类布局，返回 (图片路径, 类别下标) 列表。类别名大小写/空格不敏感。

    若目录不是每类一个文件夹（如官方 VOC 布局），退化为按文件名前缀判类：
    crazing_123.jpg → crazing。
    """
    root = Path(neu_dir)
    items: list[tuple[Path, int]] = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        norm = _norm_class(p.name)
        if norm not in CLASSES:
            continue
        idx = CLASSES.index(norm)
        imgs = sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        items.extend((f, idx) for f in imgs)
    if not items:
        dirs = voc_dirs(root)
        if dirs:
            for f in sorted(dirs[1].iterdir()):
                if f.suffix.lower() not in IMAGE_EXTS:
                    continue
                norm = _norm_class(f.stem.rsplit("_", 1)[0])
                if norm in CLASSES:
                    items.append((f, CLASSES.index(norm)))
    if not items:
        raise RuntimeError(f"{root} 下没有找到类别图片（类别：{CLASSES}）。")
    return items


def stratified_split(
    items: list[tuple[Path, int]], ratios=(0.8, 0.1, 0.1), seed: int = 42,
) -> tuple[list, list, list]:
    """按类别分层划分为 train/val/test。"""
    import random

    by_class: dict[int, list] = {}
    for it in items:
        by_class.setdefault(it[1], []).append(it)
    parts: list[list] = [[], [], []]
    rng = random.Random(seed)
    for cls_items in by_class.values():
        rng.shuffle(cls_items)
        n = len(cls_items)
        n_train = int(round(ratios[0] * n))
        n_val = int(round(ratios[1] * n))
        parts[0].extend(cls_items[:n_train])
        parts[1].extend(cls_items[n_train:n_train + n_val])
        parts[2].extend(cls_items[n_train + n_val:])
    return parts[0], parts[1], parts[2]


def voc_to_yolo(voc_dir: str | Path, out_dir: str | Path, split: str = "train") -> int:
    """把 VOC 标注转成 YOLO 格式，写入 out_dir/images/<split> 与 labels/<split>。

    返回转换的图片数。类别映射固定为 CLASSES 的下标（VOC 名统一小写匹配）。
    """
    root = Path(voc_dir)
    dirs = voc_dirs(root)
    if not dirs:
        raise FileNotFoundError(f"{root} 不是 VOC 布局（需 ANNOTATIONS/IMAGES 或 Annotations/JPEGImages）。")
    ann_dir, img_dir = dirs
    img_out = Path(out_dir) / "images" / split
    lbl_out = Path(out_dir) / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    count = 0
    for xml in sorted(ann_dir.glob("*.xml")):
        tree = ET.parse(xml)
        size = tree.find("size")
        w = float(size.find("width").text)
        h = float(size.find("height").text)
        lines: list[str] = []
        for obj in tree.iter("object"):
            norm = _norm_class(obj.findtext("name", ""))
            if norm not in CLASSES:
                continue
            b = obj.find("bndbox")
            x1, y1 = float(b.findtext("xmin")), float(b.findtext("ymin"))
            x2, y2 = float(b.findtext("xmax")), float(b.findtext("ymax"))
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            bw = min(max(bw, 1e-6), 1.0)
            bh = min(max(bh, 1e-6), 1.0)
            lines.append(f"{CLASSES.index(norm)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        if not lines:
            continue
        stems = [p for p in img_dir.glob(xml.stem + ".*")]
        if not stems:
            continue
        (lbl_out / (xml.stem + ".txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")
        (img_out / stems[0].name).write_bytes(stems[0].read_bytes())
        count += 1
    return count


def make_dataset_yaml(yolo_root: str | Path, yaml_path: str | Path) -> Path:
    """生成 ultralytics 可用的 dataset yaml。"""
    root = Path(yolo_root).resolve()
    lines = [f"path: {root.as_posix()}", "train: images/train",
             "val: images/valid", "test: images/test", "names:"]
    lines += [f"  {i}: {name}" for i, name in enumerate(CLASSES)]
    out = Path(yaml_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
