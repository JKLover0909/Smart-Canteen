"""Kiosk AI — nhận diện món trên khay, định lượng, tính tiền.

Lớp mỏng giữa API và pipeline thị giác: decode ảnh, gọi pipeline, cho phép
chỉnh tay định lượng rồi tính lại tiền (bước "yêu cầu xác nhận" trong sơ đồ).
"""

from __future__ import annotations

import cv2
import numpy as np

from .menu_config import MENU_GROUPS, price_list
from .vision import detector, pricing, store


def decode_image(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Không đọc được ảnh")
    # giới hạn cạnh dài 1280 cho ổn định tốc độ
    h, w = bgr.shape[:2]
    if max(h, w) > 1280:
        s = 1280 / max(h, w)
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return bgr


def scan(raw: bytes, **kw) -> dict:
    from .vision import pipeline
    return pipeline.process(decode_image(raw), **kw)


def get_menu() -> dict:
    return {
        "menu": price_list(),
        "modes": {
            "count": "Đếm số lượng",
            "weight": "Ước lượng khối lượng",
            "volume": "Ước lượng thể tích",
            "piece": "Đếm miếng / kết hợp kích thước",
            "fixed": "Suất cố định",
        },
    }


def model_status() -> dict:
    from .vision.quantify import calibration
    m = detector.meta()
    cal = calibration()
    return {
        "model": m,
        "calibration": {
            "source": cal.get("source"),
            "groups": sorted(cal.get("groups", {})),
            "eval": cal.get("eval_on_test", {}).get("_overall"),
        },
        "n_groups": len(MENU_GROUPS),
    }


def recalculate(items: list[dict]) -> dict:
    """Nhân viên chỉnh định lượng -> tính lại tiền theo đúng cách định lượng của món."""
    out = []
    for it in items:
        gid = it["group"]
        meta = MENU_GROUPS[gid]
        q = {
            "group": gid,
            "name": meta["name"],
            "mode": meta["mode"],
            "area_cm2": it.get("area_cm2", 0),
            "n_regions": it.get("n_regions", 1),
            "confidence": it.get("confidence", 1.0),
            "outlines": it.get("outlines", []),
            "manual": True,
        }
        if meta["mode"] == "weight":
            from .vision.quantify import _level_for
            g = float(it.get("grams", 0))
            q.update({"grams": round(g, 1), "quantity_display": f"{g:.0f}g",
                      "level": _level_for(gid, g)})
        elif meta["mode"] == "count":
            n = int(it.get("count", 1))
            q.update({"count": n, "unit": meta["unit"],
                      "quantity_display": f"{n} {meta['unit']}"})
        elif meta["mode"] == "piece":
            n = int(it.get("pieces", 1))
            q.update({"pieces": n, "quantity_display": f"{n} miếng"})
        elif meta["mode"] == "volume":
            n = int(it.get("bowls", 1))
            q.update({"bowls": n, "quantity_display": f"{n} bát"})
        out.append(pricing.price_item(q))

    out.sort(key=lambda i: -i["subtotal"])
    return {"status": "ready", "items": out, **pricing.total(out),
            "message": "Đã chỉnh tay — chờ xác nhận"}


def confirm(tx_id: str, total_vnd: int | None) -> dict:
    ok = store.confirm(tx_id, total_vnd)
    return {"ok": ok, "transaction_id": tx_id}
