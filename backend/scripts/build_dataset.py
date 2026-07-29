"""FoodSeg103 (mask semantic) -> YOLO instance-segmentation theo 16 nhóm món.

Với mỗi ảnh: decode mask semantic, gộp class id 103 -> nhóm menu, tách thành
các connected component (mỗi component = 1 instance), xuất polygon chuẩn YOLO.

    python -m scripts.build_dataset --out data/yolo_canteen
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.menu_config import GROUP_NAMES, GROUP_TO_IDX, group_of_fs103  # noqa: E402

MIN_AREA_PX = 400        # bỏ mảnh vụn nhiễu
MIN_POLY_POINTS = 3


def mask_to_polygons(binary: np.ndarray) -> list[np.ndarray]:
    """Tách mask nhị phân thành các polygon (mỗi connected component 1 instance)."""
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1
    )
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < MIN_AREA_PX:
            continue
        eps = 0.004 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(approx) >= MIN_POLY_POINTS:
            polys.append(approx)
    return polys


def convert_split(files: list[str], split: str, out: Path) -> dict:
    img_dir = out / "images" / split
    lbl_dir = out / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    per_class = {g: 0 for g in GROUP_NAMES}
    kept = skipped = 0

    for f in files:
        table = pq.read_table(f)
        for batch in table.to_batches(max_chunksize=64):
            for row in batch.to_pylist():
                classes = set(row["classes_on_image"])
                groups_present = {
                    cid: group_of_fs103(cid) for cid in classes if group_of_fs103(cid)
                }
                if not groups_present:
                    skipped += 1
                    continue

                img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
                seg = np.array(Image.open(io.BytesIO(row["label"]["bytes"])))
                if seg.ndim == 3:
                    seg = seg[:, :, 0]
                H, W = seg.shape

                lines = []
                for gid in sorted(set(groups_present.values())):
                    src_ids = [c for c, g in groups_present.items() if g == gid]
                    merged = np.isin(seg, src_ids).astype(np.uint8)
                    for poly in mask_to_polygons(merged):
                        norm = poly.astype(np.float64)
                        norm[:, 0] = np.clip(norm[:, 0] / W, 0, 1)
                        norm[:, 1] = np.clip(norm[:, 1] / H, 0, 1)
                        coords = " ".join(f"{v:.6f}" for v in norm.reshape(-1))
                        lines.append(f"{GROUP_TO_IDX[gid]} {coords}")
                        per_class[gid] += 1

                if not lines:
                    skipped += 1
                    continue

                stem = f"{split}_{row['id']:06d}"
                img.save(img_dir / f"{stem}.jpg", quality=92)
                (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
                kept += 1

        print(f"  [{split}] {Path(f).name}: kept={kept} skipped={skipped}", flush=True)

    return {"kept": kept, "skipped": skipped, "instances_per_class": per_class}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/foodseg103")
    ap.add_argument("--out", default="data/yolo_canteen")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    stats = {
        "train": convert_split(sorted(glob.glob(str(src / "train-*.parquet"))), "train", out),
        "val": convert_split(sorted(glob.glob(str(src / "validation-*.parquet"))), "val", out),
    }

    yaml_text = (
        f"path: {out.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(GROUP_NAMES)}\n"
        "names:\n"
        + "".join(f"  {i}: {g}\n" for i, g in enumerate(GROUP_NAMES))
    )
    (out / "canteen.yaml").write_text(yaml_text)
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    print("\n=== instances per class ===")
    for g in GROUP_NAMES:
        print(f"  {g:14} train={stats['train']['instances_per_class'][g]:6d} "
              f"val={stats['val']['instances_per_class'][g]:5d}")
    print(f"\nimages: train={stats['train']['kept']} val={stats['val']['kept']}")
    print(f"yaml -> {out / 'canteen.yaml'}")


if __name__ == "__main__":
    main()
