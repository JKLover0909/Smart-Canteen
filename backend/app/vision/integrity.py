"""Bước: Kiểm tra độ tin cậy và bất thường -> Tin cậy cao (tự động) / thấp (yêu cầu xác nhận).

Ngoài confidence của model, kiểm tra các dấu hiệu gian lận / sai số mà đề bài nêu:

  1. che_phu      Món rẻ nằm đè lên / viền sát món đắt  -> nghi phủ món đắt.
  2. phan_manh    Món đắt bị chia thành nhiều mảnh nhỏ  -> nghi bị che một phần.
  3. vung_la      Nhiều diện tích khay có đồ ăn nhưng model không gán được nhãn.
  4. tin_cay_thap Confidence của model dưới ngưỡng.
  5. gia_bat_thuong Tổng tiền lệch xa khoảng thường gặp của 1 khay.

Kết quả quyết định khay được tự động hiển thị hay phải gọi nhân viên xác nhận.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..menu_config import MENU_GROUPS

CONF_AUTO = 0.60            # tự động duyệt; thấp hơn -> nhân viên xác nhận
                            # (cao hơn ngưỡng phát hiện 0.45 của pipeline:
                            #  món 0.45-0.60 vẫn tính tiền nhưng phải xác nhận)
DILATE_PX = 9               # bán kính nới mask khi xét kề nhau
ADJACENCY_FRAC = 0.60       # >60% viền món đắt bị món rẻ bao quanh -> nghi phủ
CHEAP_RATIO = 0.50          # món "rẻ" phải có giá trị/cm² <= 50% món bị phủ
BURIED_RATIO = 0.60         # món đắt nhỏ hơn 60% diện tích món rẻ vây quanh
FRAGMENT_MIN = 4            # món đắt bị tách >=4 mảnh -> nghi bị che
UNLABELED_FRAC = 0.50       # >50% vùng có đồ ăn mà không nhận ra món
UNLABELED_MIN_TRAY = 0.15   # và chiếm >=15% diện tích khay mới đáng báo
TOTAL_MIN, TOTAL_MAX = 3_000, 150_000


def _value_density(group: str) -> float:
    """Giá trị tiền trên một đơn vị diện tích — dùng để xếp món đắt / rẻ."""
    m = MENU_GROUPS[group]
    mode = m["mode"]
    if mode == "weight":
        return m["price_per_kg"] * m.get("g_per_cm2", 0.8) / 1000.0
    if mode == "piece":
        return m["price_per_piece"] / m["piece_area_cm2"]
    if mode == "volume":
        return m["price_per_bowl"] / m["bowl_area_cm2"]
    return m.get("price_per_unit", 0) / 30.0


def _food_like(bgr: np.ndarray) -> np.ndarray:
    """Mask thô 'có đồ ăn': vùng bão hoà màu hoặc nhiều texture (khác khay trắng)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    tex = cv2.Laplacian(gray, cv2.CV_64F)
    tex = (np.abs(tex) > 14).astype(np.uint8) * 255
    tex = cv2.morphologyEx(tex, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    m = ((sat > 60).astype(np.uint8) * 255) | tex
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0


def check(
    bgr: np.ndarray,
    regions: list[dict],
    items: list[dict],
    tray: dict,
    total_vnd: int,
) -> dict:
    flags: list[dict] = []

    # ── 1 + 2. che phủ món đắt & phân mảnh ───────────────────────────────────
    dens = {r["group"]: _value_density(r["group"]) for r in regions}
    k = np.ones((DILATE_PX, DILATE_PX), np.uint8)

    for i, r in enumerate(regions):
        gi = r["group"]
        ring = cv2.dilate(r["mask"].astype(np.uint8), k) - r["mask"].astype(np.uint8)
        ring_area = int(ring.sum())
        if ring_area < 50:
            continue
        cheaper_touch = 0
        cheap_area = 0
        culprits = set()
        for j, o in enumerate(regions):
            if i == j:
                continue
            if dens[o["group"]] > dens[gi] * CHEAP_RATIO:
                continue  # chỉ xét món rẻ hơn rõ rệt
            overlap = int((ring & o["mask"]).sum())
            if overlap > 0:
                cheaper_touch += overlap
                cheap_area += o["area_px"]
                culprits.add(MENU_GROUPS[o["group"]]["name"])
        frac = cheaper_touch / ring_area
        # chỉ báo khi món đắt vừa bị vây kín vừa nhỏ bất thường so với món rẻ
        buried = cheap_area > 0 and r["area_px"] < BURIED_RATIO * cheap_area
        if frac >= ADJACENCY_FRAC and culprits and buried:
            flags.append({
                "code": "che_phu",
                "severity": "high",
                "message": (
                    f"{MENU_GROUPS[gi]['name']} bị {', '.join(sorted(culprits))} "
                    f"bao quanh {frac*100:.0f}% viền — nghi bị phủ để giảm tiền"
                ),
                "group": gi,
            })

    for it in items:
        d = _value_density(it["group"])
        if it["n_regions"] >= FRAGMENT_MIN and d >= 40:
            flags.append({
                "code": "phan_manh",
                "severity": "medium",
                "message": (
                    f"{it['name']} tách thành {it['n_regions']} mảnh rời — "
                    "có thể bị che một phần"
                ),
                "group": it["group"],
            })

    # ── 3. vùng có đồ ăn nhưng không gán được nhãn ───────────────────────────
    x, y, w, h = tray["bbox"]
    crop = bgr[y:y + h, x:x + w]
    food = _food_like(crop)
    covered = np.zeros(food.shape, bool)
    for r in regions:
        covered |= r["mask"][y:y + h, x:x + w]
    food_px = int(food.sum())
    unlabeled = int((food & ~covered).sum())
    unl_frac = unlabeled / food_px if food_px else 0.0
    tray_px = max(1, w * h)
    if unl_frac >= UNLABELED_FRAC and unlabeled >= UNLABELED_MIN_TRAY * tray_px:
        flags.append({
            "code": "vung_la",
            "severity": "medium",
            "message": (
                f"{unl_frac*100:.0f}% vùng có đồ ăn không nhận ra món — "
                "có thể là món ngoài menu hoặc bị che"
            ),
        })

    # ── 4. confidence thấp ───────────────────────────────────────────────────
    for it in items:
        if it["confidence"] < CONF_AUTO:
            flags.append({
                "code": "tin_cay_thap",
                "severity": "medium",
                "message": f"{it['name']}: độ tin cậy {it['confidence']*100:.0f}%",
                "group": it["group"],
            })

    # ── 5. tổng tiền bất thường ──────────────────────────────────────────────
    if items and not (TOTAL_MIN <= total_vnd <= TOTAL_MAX):
        flags.append({
            "code": "gia_bat_thuong",
            "severity": "high",
            "message": f"Tổng tiền {total_vnd:,}đ".replace(",", ".") + " ngoài khoảng thường gặp",
        })
    if not items:
        flags.append({
            "code": "khay_trong",
            "severity": "high",
            "message": "Không nhận ra món nào trên khay",
        })

    high = [f for f in flags if f["severity"] == "high"]
    min_conf = min((i["confidence"] for i in items), default=0.0)
    trusted = not high and not flags and min_conf >= CONF_AUTO

    return {
        "trusted": trusted,
        "decision": "auto_display" if trusted else "need_confirm",
        "decision_label": "Tin cậy cao — tự động hiển thị" if trusted
                          else "Tin cậy thấp — yêu cầu xác nhận",
        "min_confidence": round(min_conf, 3),
        "unlabeled_food_frac": round(unl_frac, 3),
        "flags": flags,
    }
