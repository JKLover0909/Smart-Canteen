"""Bước: Ánh xạ bảng giá trong ngày -> Tính tiền cho từng món.

Bảng giá tách khỏi model: đổi giá không cần train lại. Mỗi cách định lượng có
công thức tính tiền riêng, và công thức được trả về cho UI để công nhân đối chiếu.
"""

from __future__ import annotations

from ..menu_config import MENU_GROUPS


def _vnd(x: float) -> str:
    return f"{int(round(x)):,}đ".replace(",", ".")


def price_item(q: dict) -> dict:
    """Nhận 1 dòng định lượng -> thêm đơn giá, thành tiền, công thức."""
    meta = MENU_GROUPS[q["group"]]
    mode = q["mode"]

    if mode == "weight":
        grams = q["grams"]
        level = q.get("level")
        if level:
            # định giá theo mức ít/vừa/nhiều — khả thi cao hơn cân gram chính xác
            subtotal = level["price"]
            formula = f"{grams:.0f}g → mức {level['label']} = {_vnd(subtotal)}"
            unit_price = f"{level['label']}: {_vnd(level['price'])}"
        else:
            subtotal = grams / 1000.0 * meta["price_per_kg"]
            formula = (
                f"{grams:.0f}g × {meta['price_per_kg'] // 1000}k/kg = {_vnd(subtotal)}"
            )
            unit_price = f"{_vnd(meta['price_per_kg'])}/kg"

    elif mode == "count":
        n = q["count"]
        subtotal = n * meta["price_per_unit"]
        formula = f"{n} {meta['unit']} × {_vnd(meta['price_per_unit'])} = {_vnd(subtotal)}"
        unit_price = f"{_vnd(meta['price_per_unit'])}/{meta['unit']}"

    elif mode == "piece":
        n = q["pieces"]
        subtotal = n * meta["price_per_piece"]
        formula = f"{n} miếng × {_vnd(meta['price_per_piece'])} = {_vnd(subtotal)}"
        unit_price = f"{_vnd(meta['price_per_piece'])}/miếng"

    elif mode == "volume":
        n = q["bowls"]
        subtotal = n * meta["price_per_bowl"]
        formula = f"{n} bát × {_vnd(meta['price_per_bowl'])} = {_vnd(subtotal)}"
        unit_price = f"{_vnd(meta['price_per_bowl'])}/bát"

    else:
        subtotal = meta.get("price_fixed", 0)
        formula = f"suất cố định = {_vnd(subtotal)}"
        unit_price = _vnd(subtotal)

    return {
        **q,
        "unit_price_display": unit_price,
        "subtotal": int(round(subtotal)),
        "subtotal_display": _vnd(subtotal),
        "formula": formula,
    }


def total(items: list[dict]) -> dict:
    s = sum(i["subtotal"] for i in items)
    return {"total": int(s), "total_display": _vnd(s)}
