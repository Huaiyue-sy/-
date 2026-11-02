"""第2周·第1天（上）：PyTorch 环境验证（对应实训指南附录）。"""
from __future__ import annotations

import torch


def main() -> None:
    print("PyTorch 版本：", torch.__version__)
    print("构建所用 CUDA 版本：", torch.version.cuda)
    print("CUDA 设备数：", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("GPU 设备名：", torch.cuda.get_device_name(0))
        print("cuDNN 版本：", torch.backends.cudnn.version())
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("Apple MPS 可用（Mac GPU 加速）")
    x = torch.randn(512, 512)
    y = x @ x
    print(f"矩阵乘法自检：{tuple(y.shape)}，均值 {y.mean():.4f}")
    print("环境验证通过。")


if __name__ == "__main__":
    main()
