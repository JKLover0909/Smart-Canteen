"""Orchestrator — chạy đúng thứ tự các bước trong sơ đồ pipeline.

    ảnh -> QC -> vùng khay -> phân vùng món -> nhận diện tên -> định lượng
        -> quy đổi gram -> bảng giá ngày -> tính tiền -> tin cậy/bất thường
        -> tự động hiển thị | yêu cầu xác nhận -> lưu ảnh + giao dịch

Mỗi bước trả về log riêng (`steps`) để UI vẽ lại được đường đi của một khay.
"""

from __future__ import annotations

import time
from collections import defaultdict

import cv2
import numpy as np

from ..menu_config import MENU_GROUPS
from . import detector, integrity, pricing, quality, quantify, store, tray

# Ngưỡng phát hiện dùng để TÍNH TIỀN. Với hệ thống thu tiền, thu tiền món công
# nhân không lấy (false positive) tệ hơn là bỏ sót, nên ngưỡng đặt cao hơn mặc
# định của YOLO. Dò trên FoodSeg103 val: conf 0.25 -> P=0.885, conf 0.45 ->
# P=0.938 (recall giảm 0.89 -> 0.83). Món bị bỏ sót vẫn được cờ "vùng lạ" bắt
# và chuyển sang nhân viên xác nhận, nên đánh đổi này là có lợi.
DETECT_CONF = 0.45


def _mask_outline(mask: np.ndarray, W: int, H: int, max_pts: int = 60) -> list[list[float]]:
    """Viền mask -> polygon % để frontend overlay không cần gửi ảnh mask."""
    m = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    c = max(contours, key=cv2.contourArea)
    eps = 0.006 * cv2.arcLength(c, True)
    poly = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
    if len(poly) > max_pts:
        idx = np.linspace(0, len(poly) - 1, max_pts).astype(int)
        poly = poly[idx]
    return [[round(float(x) / W * 100, 2), round(float(y) / H * 100, 2)] for x, y in poly]


def process(
    bgr: np.ndarray,
    *,
    canteen_id: str = "NA1",
    shift: str = "Trưa",
    worker_id: str | None = None,
    conf: float = DETECT_CONF,
    persist: bool = True,
) -> dict:
    t0 = time.time()
    H, W = bgr.shape[:2]
    steps: list[dict] = []

    # ── 1. Kiểm tra chất lượng ảnh ───────────────────────────────────────────
    qc = quality.check(bgr)
    steps.append({"step": "Kiểm tra chất lượng ảnh", "ok": qc["passed"], "detail": qc})
    if not qc["passed"]:
        return {
            "status": "rejected",
            "message": "Ảnh không đạt chất lượng — vui lòng quét lại: "
                       + "; ".join(qc["issues"]),
            "steps": steps, "items": [], "total": 0, "total_display": "0đ",
            "integrity": {"trusted": False, "decision": "need_confirm",
                          "decision_label": "Ảnh lỗi — quét lại",
                          "min_confidence": 0.0, "flags": [], "unlabeled_food_frac": 0.0},
            "latency_ms": int((time.time() - t0) * 1000),
        }

    # ── 2. Phát hiện vùng khay (đồng thời lấy scale px -> cm²) ───────────────
    tray_info = tray.detect(bgr)
    steps.append({
        "step": "Phát hiện vùng khay",
        "ok": tray_info["found"],
        "detail": {k: v for k, v in tray_info.items() if k != "mask"},
    })

    # ── 3 + 4. Phân vùng từng món + nhận diện tên món ────────────────────────
    regions = detector.segment(bgr, conf=conf)
    # chỉ giữ vùng nằm trong khay
    tx, ty, tw, th = tray_info["bbox"]
    keep = []
    for r in regions:
        inside = r["mask"][ty:ty + th, tx:tx + tw].sum()
        if inside >= 0.5 * r["area_px"]:
            keep.append(r)
    regions = keep
    steps.append({
        "step": "Phân vùng + nhận diện món",
        "ok": bool(regions),
        "detail": {
            "n_regions": len(regions),
            "groups": sorted({MENU_GROUPS[r["group"]]["name"] for r in regions}),
        },
    })

    # ── 5 + 6. Ước lượng định lượng -> quy đổi khối lượng ───────────────────
    by_group: dict[str, list[dict]] = defaultdict(list)
    for r in regions:
        by_group[r["group"]].append(r)

    cm2_per_px = tray_info["cm2_per_px"]
    quantified = [quantify.quantify(g, rs, cm2_per_px) for g, rs in by_group.items()]
    steps.append({
        "step": "Ước lượng định lượng & quy đổi khối lượng",
        "ok": bool(quantified),
        "detail": {
            "cm2_per_px": round(cm2_per_px, 6),
            "calib": quantify.calibration().get("source", "default"),
            "quantities": {q["name"]: q["quantity_display"] for q in quantified},
        },
    })

    # ── 7 + 8. Ánh xạ bảng giá ngày -> tính tiền từng món ───────────────────
    items = [pricing.price_item(q) for q in quantified]
    items.sort(key=lambda i: -i["subtotal"])
    tot = pricing.total(items)
    steps.append({
        "step": "Ánh xạ bảng giá & tính tiền",
        "ok": True,
        "detail": {i["name"]: i["formula"] for i in items},
    })

    # ── 9. Kiểm tra tin cậy & bất thường ─────────────────────────────────────
    integ = integrity.check(bgr, regions, items, tray_info, tot["total"])
    steps.append({
        "step": "Kiểm tra tin cậy & bất thường",
        "ok": integ["trusted"],
        "detail": {"decision": integ["decision_label"],
                   "flags": [f["message"] for f in integ["flags"]]},
    })

    # viền mask cho UI
    for it in items:
        it["outlines"] = [
            _mask_outline(r["mask"], W, H) for r in by_group[it["group"]]
        ]

    result = {
        "status": "ready",
        "message": integ["decision_label"],
        "items": items,
        **tot,
        "integrity": integ,
        "tray": {k: v for k, v in tray_info.items()},
        "image_quality": qc,
        "steps": steps,
        "image_size": {"w": W, "h": H},
        "model": detector.meta(),
        "latency_ms": int((time.time() - t0) * 1000),
    }

    # ── 10. Lưu ảnh, kết quả và giao dịch ────────────────────────────────────
    if persist:
        tx_id = store.save(result, bgr, canteen_id=canteen_id, shift=shift,
                           worker_id=worker_id)
        result["transaction_id"] = tx_id
        steps.append({"step": "Lưu ảnh, kết quả và giao dịch", "ok": True,
                      "detail": {"transaction_id": tx_id}})

    return result
