"""Menu nhà ăn: gộp 103 class nguyên liệu FoodSeg103 -> 15 nhóm món có bảng giá.

Mỗi nhóm gắn 1 trong 5 cách định lượng theo đúng bảng nghiệp vụ:

    count   Đếm số lượng          trứng, bánh mì, hộp sữa, hoa quả nguyên quả
    weight  Ước lượng khối lượng  cơm, mì, rau, thịt thái nhỏ
    volume  Ước lượng thể tích    canh, nước, cháo
    piece   Đếm miếng + kích thước miếng thịt, cá, đùi gà
    fixed   Suất cố định          nhận diện loại suất, không cần định lượng

`levels` dùng cho nhóm định giá theo mức ít/vừa/nhiều (khả thi cao hơn cân gram).
"""

from __future__ import annotations

# ── FoodSeg103 class id -> nhóm món nhà ăn ───────────────────────────────────
FS103_TO_GROUP: dict[int, str] = {
    # cơm
    66: "com",
    # mì / phở / bún
    64: "mi_pho", 65: "mi_pho",
    # thịt đỏ thái nhỏ (heo, bò, cừu, thịt rán)
    46: "thit_do", 47: "thit_do", 50: "thit_do", 51: "thit_do", 60: "thit_do",
    # thịt gà / vịt
    48: "thit_ga",
    # cá & hải sản
    53: "ca_hai_san", 54: "ca_hai_san", 55: "ca_hai_san", 56: "ca_hai_san",
    # trứng
    24: "trung", 2: "trung",
    # đậu phụ
    68: "dau_phu",
    # xúc xích
    49: "xuc_xich",
    # rau xanh
    77: "rau_xanh", 80: "rau_xanh", 85: "rau_xanh", 87: "rau_xanh",
    88: "rau_xanh", 90: "rau_xanh", 91: "rau_xanh", 92: "rau_xanh",
    95: "rau_xanh", 96: "rau_xanh", 102: "rau_xanh", 79: "rau_xanh",
    86: "rau_xanh", 76: "rau_xanh", 89: "rau_xanh",
    # rau củ
    59: "rau_cu", 69: "rau_cu", 70: "rau_cu", 72: "rau_cu", 73: "rau_cu",
    81: "rau_cu", 82: "rau_cu", 83: "rau_cu", 84: "rau_cu", 93: "rau_cu",
    94: "rau_cu", 71: "rau_cu",
    # nấm
    97: "nam", 98: "nam", 99: "nam", 100: "nam", 101: "nam",
    # canh / soup
    57: "canh",
    # bánh mì
    58: "banh_mi", 62: "banh_mi", 63: "banh_mi",
    # trái cây nguyên quả
    25: "trai_cay", 26: "trai_cay", 27: "trai_cay", 28: "trai_cay",
    29: "trai_cay", 34: "trai_cay", 36: "trai_cay", 38: "trai_cay",
    39: "trai_cay", 40: "trai_cay", 42: "trai_cay", 43: "trai_cay",
    44: "trai_cay", 45: "trai_cay", 37: "trai_cay",
    # đồ uống đóng hộp
    13: "do_uong", 14: "do_uong", 15: "do_uong", 16: "do_uong", 12: "do_uong",
    # khoai tây chiên
    3: "khoai_chien",
}

# ── Bảng giá trong ngày + cách định lượng ────────────────────────────────────
MENU_GROUPS: dict[str, dict] = {
    "com": {
        "name": "Cơm trắng",
        "mode": "weight",
        "price_per_kg": 30_000,
        "levels": [("Ít", 150, 3_000), ("Vừa", 250, 5_000), ("Nhiều", 10**9, 7_000)],
        "g_per_cm2": 1.05,      # cơm xới cao -> gram/diện tích lớn
        "min_g": 60, "max_g": 400,
    },
    "mi_pho": {
        "name": "Mì / Phở / Bún",
        "mode": "weight",
        "price_per_kg": 40_000,
        "levels": [("Ít", 150, 5_000), ("Vừa", 260, 8_000), ("Nhiều", 10**9, 12_000)],
        "g_per_cm2": 0.85,
        "min_g": 60, "max_g": 400,
    },
    "thit_do": {
        "name": "Thịt heo / bò",
        "mode": "weight",
        "price_per_kg": 150_000,
        "g_per_cm2": 1.10,
        "min_g": 20, "max_g": 300,
    },
    "thit_ga": {
        "name": "Thịt gà / vịt",
        "mode": "piece",
        "price_per_piece": 15_000,
        "piece_area_cm2": 45.0,   # 1 miếng ~45cm2 -> suy ra số miếng
        "min_pieces": 1, "max_pieces": 4,
    },
    "ca_hai_san": {
        "name": "Cá / hải sản",
        "mode": "piece",
        "price_per_piece": 18_000,
        "piece_area_cm2": 40.0,
        "min_pieces": 1, "max_pieces": 4,
    },
    "trung": {
        "name": "Trứng",
        "mode": "count",
        "price_per_unit": 5_000,
        "unit": "quả",
        "max_count": 4,
    },
    "dau_phu": {
        "name": "Đậu phụ",
        "mode": "count",
        "price_per_unit": 3_000,
        "unit": "miếng",
        "max_count": 6,
    },
    "xuc_xich": {
        "name": "Xúc xích",
        "mode": "count",
        "price_per_unit": 8_000,
        "unit": "cái",
        "max_count": 4,
    },
    "rau_xanh": {
        "name": "Rau xanh",
        "mode": "weight",
        "price_per_kg": 35_000,
        "g_per_cm2": 0.55,        # rau xốp, phủ diện tích rộng nhưng nhẹ
        "min_g": 20, "max_g": 250,
    },
    "rau_cu": {
        "name": "Rau củ",
        "mode": "weight",
        "price_per_kg": 30_000,
        "g_per_cm2": 0.95,
        "min_g": 20, "max_g": 250,
    },
    "nam": {
        "name": "Nấm",
        "mode": "weight",
        "price_per_kg": 60_000,
        "g_per_cm2": 0.60,
        "min_g": 20, "max_g": 200,
    },
    "canh": {
        "name": "Canh",
        "mode": "volume",
        "price_per_bowl": 5_000,
        "bowl_area_cm2": 90.0,    # 1 bát canh chiếm ~90cm2 nhìn từ trên
        "max_bowls": 2,
    },
    "banh_mi": {
        "name": "Bánh mì / bánh bao",
        "mode": "count",
        "price_per_unit": 4_000,
        "unit": "cái",
        "max_count": 4,
    },
    "trai_cay": {
        "name": "Trái cây",
        "mode": "count",
        "price_per_unit": 6_000,
        "unit": "quả",
        "max_count": 5,
    },
    "do_uong": {
        "name": "Đồ uống đóng hộp",
        "mode": "count",
        "price_per_unit": 8_000,
        "unit": "hộp",
        "max_count": 3,
    },
    "khoai_chien": {
        "name": "Khoai tây chiên",
        "mode": "weight",
        "price_per_kg": 60_000,
        "g_per_cm2": 0.50,
        "min_g": 20, "max_g": 200,
    },
}

# Thứ tự class dùng khi train YOLO (ổn định, không phụ thuộc dict order)
GROUP_NAMES: list[str] = sorted(MENU_GROUPS)
GROUP_TO_IDX: dict[str, int] = {g: i for i, g in enumerate(GROUP_NAMES)}
IDX_TO_GROUP: dict[int, str] = {i: g for g, i in GROUP_TO_IDX.items()}


def group_of_fs103(class_id: int) -> str | None:
    return FS103_TO_GROUP.get(class_id)


def price_list() -> list[dict]:
    """Bảng giá trong ngày để hiển thị / kiểm tra."""
    out = []
    for gid in GROUP_NAMES:
        m = MENU_GROUPS[gid]
        row = {"id": gid, "name": m["name"], "mode": m["mode"]}
        if m["mode"] == "weight":
            row["price_display"] = f"{m['price_per_kg']:,}đ/kg".replace(",", ".")
            if "levels" in m:
                row["levels"] = [
                    {"label": lb, "max_g": None if mx > 10**8 else mx, "price": pr}
                    for lb, mx, pr in m["levels"]
                ]
        elif m["mode"] == "piece":
            row["price_display"] = f"{m['price_per_piece']:,}đ/miếng".replace(",", ".")
        elif m["mode"] == "volume":
            row["price_display"] = f"{m['price_per_bowl']:,}đ/bát".replace(",", ".")
        else:
            row["price_display"] = f"{m['price_per_unit']:,}đ/{m['unit']}".replace(",", ".")
        out.append(row)
    return out
