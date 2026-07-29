"""Fine-tune YOLO11-seg trên menu nhà ăn 16 nhóm món.

    python -m scripts.train --epochs 60 --model yolo11s-seg.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/yolo_canteen/canteen.yaml")
    ap.add_argument("--model", default="yolo11s-seg.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--name", default="canteen-seg")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(Path(args.data).resolve()),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=0,
        workers=8,
        project="runs",
        name=args.name,
        exist_ok=True,
        patience=20,
        # augment nhẹ cho biến thiên ánh sáng / màu nước sốt / góc nhìn
        hsv_h=0.02, hsv_s=0.55, hsv_v=0.45,
        degrees=12, translate=0.10, scale=0.35, fliplr=0.5,
        mosaic=1.0, close_mosaic=10, mixup=0.05,
        plots=True,
    )
    print("best ->", Path("runs") / args.name / "weights" / "best.pt")


if __name__ == "__main__":
    main()
