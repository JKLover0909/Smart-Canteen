"""Bước: Ước lượng số lượng / thể tích  ->  Chuyển đổi sang khối lượng.

Không dùng một công thức duy nhất. Mỗi nhóm món đi theo cách định lượng riêng:

    weight  diện tích mask (cm²) × hệ số g/cm² -> gram   (cơm, thịt thái nhỏ, rau)
    count   đếm số instance mask                -> quả/cái/hộp (trứng, bánh, sữa)
    piece   diện tích / diện tích 1 miếng       -> số miếng (gà, cá)
    volume  diện tích / diện tích 1 bát         -> số bát   (canh)

Hệ số g/cm² lấy từ calib/area_to_gram.json (fit trên nhãn gram thật của
Nutrition5k). Nếu chưa có file calib thì dùng giá trị mặc định trong menu_config.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..menu_config import MENU_GROUPS

CALIB_PATH = Path(__file__).resolve().parents[2] / "calib" / "area_to_gram.json"

_calib: dict | None = None


def calibration() -> dict:
    global _calib
    if _calib is None:
        if CALIB_PATH.exists():
            _calib = json.loads(CALIB_PATH.read_text())
        else:
            _calib = {"source": "default (chưa hiệu chỉnh)", "groups": {}}
    return _calib


def g_per_cm2(group: str) -> tuple[float, str]:
    """Hệ số g/cm² + nguồn của hệ số đó."""
    cal = calibration().get("groups", {})
    if group in cal and cal[group].get("g_per_cm2"):
        return float(cal[group]["g_per_cm2"]), "nutrition5k"
    return float(MENU_GROUPS[group].get("g_per_cm2", 0.8)), "default"


def _level_for(group: str, grams: float):
    """Mức ít/vừa/nhiều nếu nhóm món định giá theo mức."""
    levels = MENU_GROUPS[group].get("levels")
    if not levels:
        return None
    for label, max_g, price in levels:
        if grams <= max_g:
            return {"label": label, "price": price}
    return {"label": levels[-1][0], "price": levels[-1][2]}


def quantify(group: str, regions: list[dict], cm2_per_px: float) -> dict:
    """Gộp các vùng cùng nhóm món -> một dòng định lượng."""
    meta = MENU_GROUPS[group]
    mode = meta["mode"]
    area_px = sum(r["area_px"] for r in regions)
    area_cm2 = area_px * cm2_per_px
    n_regions = len(regions)

    out = {
        "group": group,
        "name": meta["name"],
        "mode": mode,
        "area_cm2": round(area_cm2, 1),
        "n_regions": n_regions,
        "confidence": round(sum(r["confidence"] for r in regions) / n_regions, 3),
    }

    if mode == "weight":
        coef, src = g_per_cm2(group)
        grams = area_cm2 * coef
        grams = max(meta.get("min_g", 0), min(meta.get("max_g", 10**6), grams))
        out.update({
            "grams": round(grams, 1),
            "g_per_cm2": coef,
            "calib_source": src,
            "quantity_display": f"{grams:.0f}g",
            "level": _level_for(group, grams),
        })

    elif mode == "count":
        n = max(1, min(meta.get("max_count", 10), n_regions))
        out.update({
            "count": n,
            "unit": meta["unit"],
            "quantity_display": f"{n} {meta['unit']}",
        })

    elif mode == "piece":
        by_area = area_cm2 / meta["piece_area_cm2"]
        # số miếng = kết hợp số vùng rời rạc và diện tích tổng (đề bài: "đếm
        # miếng hoặc kết hợp kích thước")
        n = round(max(n_regions, by_area))
        n = max(meta.get("min_pieces", 1), min(meta.get("max_pieces", 10), n))
        out.update({
            "pieces": int(n),
            "pieces_by_area": round(by_area, 2),
            "quantity_display": f"{int(n)} miếng",
        })

    elif mode == "volume":
        bowls = round(area_cm2 / meta["bowl_area_cm2"])
        bowls = max(1, min(meta.get("max_bowls", 3), bowls))
        out.update({
            "bowls": int(bowls),
            "quantity_display": f"{int(bowls)} bát",
        })

    else:  # fixed
        out.update({"quantity_display": "1 suất"})

    return out
