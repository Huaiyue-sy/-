"""YOLOv8 缺陷检测 CLI：给 Node 服务调用。

读取一张图片，输出一个 JSON 对象（stdout），包含检测框、原图尺寸和
带标注框的 JPEG（base64）。错误也以 JSON 输出，方便 Node 解析。

用法：
    python yolo_detect.py --image /tmp/a.jpg --model results/.../best.pt --conf 0.25
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="待检测图片路径")
    parser.add_argument("--model", required=True, help="YOLO 权重路径")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值")
    args = parser.parse_args()

    started = time.perf_counter()
    try:
        logging.getLogger("ultralytics").setLevel(logging.ERROR)
        from PIL import Image
        from ultralytics import YOLO

        model = YOLO(args.model)
        results = model.predict(
            source=args.image, conf=args.conf, iou=args.iou, verbose=False,
        )
        result = results[0]
        names = result.names

        detections = []
        boxes = result.boxes
        if boxes is not None:
            for cls_id, score, xyxy in zip(
                boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist(),
            ):
                detections.append({
                    "class_name": names[int(cls_id)],
                    "confidence": round(float(score), 4),
                    "box": [round(float(value), 2) for value in xyxy],
                })

        # result.plot() 返回 BGR ndarray，转成 RGB 后编码为 JPEG。
        annotated = result.plot()[:, :, ::-1].copy()
        buffer = io.BytesIO()
        Image.fromarray(annotated).save(buffer, format="JPEG", quality=90)

        payload = {
            "ok": True,
            "detections": detections,
            "width": int(result.orig_shape[1]),
            "height": int(result.orig_shape[0]),
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
            "annotated": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "names": {str(key): value for key, value in names.items()},
        }
        print(json.dumps(payload, ensure_ascii=False))
    except Exception as error:  # noqa: BLE001 - 以 JSON 形式把错误交回 Node
        print(json.dumps({
            "ok": False,
            "error": f"{type(error).__name__}: {error}",
        }, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
