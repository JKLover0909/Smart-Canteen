"""Kiểm tra quy tắc chống gian lận bằng mask tổng hợp (không cần model).

Ba tình huống:
  A. Khay bình thường: món đắt và món rẻ nằm cạnh nhau, kích thước tương xứng
     -> KHÔNG được báo che phủ.
  B. Món đắt bị món rẻ phủ: chỉ còn lộ một mẩu nhỏ, bị rau vây kín
     -> PHẢI báo che phủ.
  C. Món đắt bị tách nhiều mảnh nhỏ
     -> PHẢI báo phân mảnh.

    python -m scripts.test_integrity
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.vision import integrity  # noqa: E402

H = W = 400
TRAY = {"bbox": [0, 0, W, H], "cm2_per_px": 1200.0 / (W * H), "found": True}


def _rect(x, y, w, h) -> np.ndarray:
    m = np.zeros((H, W), bool)
    m[y:y + h, x:x + w] = True
    return m


def _region(group, mask, conf=0.9) -> dict:
    return {"group": group, "mask": mask, "area_px": int(mask.sum()),
            "confidence": conf, "bbox_px": [0, 0, W, H]}


def _items(regions):
    from collections import defaultdict
    from app.vision import quantify
    by = defaultdict(list)
    for r in regions:
        by[r["group"]].append(r)
    return [quantify.quantify(g, rs, TRAY["cm2_per_px"]) for g, rs in by.items()]


def run(name: str, regions: list[dict], expect: str | None) -> bool:
    bgr = np.full((H, W, 3), 235, np.uint8)      # khay trắng
    for r in regions:                            # tô màu vùng món cho _food_like
        bgr[r["mask"]] = (40, 90, 160)
    items = _items(regions)
    res = integrity.check(bgr, regions, items, TRAY, 30_000)
    codes = {f["code"] for f in res["flags"]}
    ok = (expect in codes) if expect else ("che_phu" not in codes and "phan_manh" not in codes)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"         cờ: {sorted(codes) or 'không có'}")
    for f in res["flags"]:
        print(f"         · {f['message']}")
    return ok


def main() -> None:
    print("Kiểm tra quy tắc bất thường:\n")
    results = []

    # A. bình thường: thịt 100x100 cạnh rau 100x100
    results.append(run(
        "A. khay bình thường (thịt cạnh rau, kích thước tương xứng)",
        [_region("thit_do", _rect(60, 60, 100, 100)),
         _region("rau_xanh", _rect(200, 60, 100, 100))],
        None,
    ))

    # B. thịt chỉ lộ mẩu 30x30, bị rau lớn vây kín
    meat = _rect(185, 185, 30, 30)
    veg = _rect(150, 150, 100, 100) & ~meat
    results.append(run(
        "B. thịt bị rau phủ, chỉ lộ mẩu nhỏ",
        [_region("thit_do", meat), _region("rau_xanh", veg)],
        "che_phu",
    ))

    # C. thịt bị tách 5 mảnh nhỏ
    frags = [_region("thit_do", _rect(40 + i * 55, 200, 26, 26)) for i in range(5)]
    results.append(run(
        "C. thịt tách thành 5 mảnh rời",
        frags + [_region("rau_xanh", _rect(40, 60, 260, 90))],
        "phan_manh",
    ))

    print(f"\n{sum(results)}/{len(results)} tình huống đúng")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
