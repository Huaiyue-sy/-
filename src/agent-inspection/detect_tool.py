"""视觉模块：把第 2-3 周训练的瑕疵分类模型加载为智能体工具。"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.cv import build_efficientnet_b0, build_resnet18, predict_single
from common.dl import pick_device

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class VisionTool:
    """ResNet18 / EfficientNet-B0 分类模型工具。"""

    def __init__(self, model_path: str | Path = "models/resnet18_neu.pth"):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"未找到分类模型 {self.model_path}。请先运行：\n"
                "  python3 src/week2-3-surface-defect/02_resnet_classify.py")
        ckpt = torch.load(self.model_path, map_location="cpu", weights_only=False)
        self.classes = list(ckpt["classes"])
        self.img_size = int(ckpt.get("img_size", 224))
        arch = ckpt.get("arch", "resnet18")
        if arch == "efficientnet_b0":
            model = build_efficientnet_b0(len(self.classes), pretrained=False)
        else:
            model = build_resnet18(len(self.classes), pretrained=False)
        model.load_state_dict(ckpt["state_dict"])
        self.device = pick_device()
        self.model = model.to(self.device).eval()
        self.test_acc = ckpt.get("test_acc")

    def inspect(self, image_path: str | Path) -> tuple[str, float]:
        """单张图推理，返回 (类别名, 置信度)。"""
        from PIL import Image

        img = Image.open(image_path).convert("L").convert("RGB")
        cls_idx, conf = predict_single(self.model, img, self.img_size, self.device)
        return self.classes[cls_idx], conf

    def inspect_dir(self, dir_path: str | Path) -> list[Path]:
        return sorted(p for p in Path(dir_path).rglob("*") if p.suffix.lower() in IMG_EXTS)
