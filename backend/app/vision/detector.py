"""Bước: Phân vùng từng món ăn + Nhận diện tên món.

Một lần chạy YOLO-seg cho ra đồng thời mask từng vùng món (instance
segmentation) và nhãn món. Mask — không phải bbox — mới là cái dùng để định
lượng, vì vùng rau/cơm phủ lẫn nhau rất nhiều.

Hỗ trợ 2 loại weights, chọn bằng biến môi trường CANTEEN_WEIGHTS:

  fs103    Model 103 class nguyên liệu FoodSeg103. Tên class được map về nhóm
           món NGAY LÚC INFERENCE qua FS103_NAME_TO_GROUP. Chính xác hơn (xem
           README) và đổi menu không cần train lại — chỉ sửa menu_config.
  canteen  Model đã fine-tune trực tiếp trên 16 nhóm món (scripts/train.py).
           Nhẹ hơn ~7x, nhanh hơn, nhưng thêm món mới thì phải train lại.

Loại weights được nhận ra tự động từ số class + tên class của chính file .pt.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np

from ..menu_config import FS103_TO_GROUP, GROUP_TO_IDX

_BACKEND = Path(__file__).resolve().parents[2]

# Ưu tiên model 103-class (chính xác hơn), fallback về model 16 nhóm tự train
_DEFAULT_WEIGHTS = [
    _BACKEND / "models" / "foodseg103-seg.pt",
    _BACKEND / "models" / "canteen-seg.pt",
    _BACKEND / "runs" / "canteen-seg" / "weights" / "best.pt",
]

# Tên class FoodSeg103 (chuẩn hoá lower/strip) -> nhóm món
_FS103_ID2NAME = {
    0: "background", 1: "candy", 2: "egg tart", 3: "french fries", 4: "chocolate",
    5: "biscuit", 6: "popcorn", 7: "pudding", 8: "ice cream", 9: "cheese butter",
    10: "cake", 11: "wine", 12: "milkshake", 13: "coffee", 14: "juice", 15: "milk",
    16: "tea", 17: "almond", 18: "red beans", 19: "cashew", 20: "dried cranberries",
    21: "soy", 22: "walnut", 23: "peanut", 24: "egg", 25: "apple", 26: "date",
    27: "apricot", 28: "avocado", 29: "banana", 30: "strawberry", 31: "cherry",
    32: "blueberry", 33: "raspberry", 34: "mango", 35: "olives", 36: "peach",
    37: "lemon", 38: "pear", 39: "fig", 40: "pineapple", 41: "grape", 42: "kiwi",
    43: "melon", 44: "orange", 45: "watermelon", 46: "steak", 47: "pork",
    48: "chicken duck", 49: "sausage", 50: "fried meat", 51: "lamb", 52: "sauce",
    53: "crab", 54: "fish", 55: "shellfish", 56: "shrimp", 57: "soup", 58: "bread",
    59: "corn", 60: "hamburg", 61: "pizza", 62: "hanamaki baozi",
    63: "wonton dumplings", 64: "pasta", 65: "noodles", 66: "rice", 67: "pie",
    68: "tofu", 69: "eggplant", 70: "potato", 71: "garlic", 72: "cauliflower",
    73: "tomato", 74: "kelp", 75: "seaweed", 76: "spring onion", 77: "rape",
    78: "ginger", 79: "okra", 80: "lettuce", 81: "pumpkin", 82: "cucumber",
    83: "white radish", 84: "carrot", 85: "asparagus", 86: "bamboo shoots",
    87: "broccoli", 88: "celery stick", 89: "cilantro mint", 90: "snow peas",
    91: "cabbage", 92: "bean sprouts", 93: "onion", 94: "pepper", 95: "green beans",
    96: "french beans", 97: "king oyster mushroom", 98: "shiitake",
    99: "enoki mushroom", 100: "oyster mushroom", 101: "white button mushroom",
    102: "salad", 103: "other ingredients",
}
FS103_NAME_TO_GROUP: dict[str, str] = {
    _FS103_ID2NAME[cid]: grp
    for cid, grp in FS103_TO_GROUP.items()
    if cid in _FS103_ID2NAME
}

_lock = threading.Lock()
_model = None
_meta: dict = {"loaded": False}
_cls_to_group: dict[int, str] = {}


def weights_path() -> Path | None:
    env = os.environ.get("CANTEEN_WEIGHTS")
    if env and Path(env).exists():
        return Path(env)
    for p in _DEFAULT_WEIGHTS:
        if p.exists():
            return p
    return None


def _build_class_map(names: dict[int, str]) -> tuple[dict[int, str], str]:
    """class index -> nhóm món, và loại weights đã nhận ra."""
    lowered = {i: str(n).strip().lower() for i, n in names.items()}

    # weights fine-tune trên nhóm món: tên class chính là id nhóm
    if all(n in GROUP_TO_IDX for n in lowered.values()):
        return {i: n for i, n in lowered.items()}, "canteen"

    # weights FoodSeg103: map tên nguyên liệu -> nhóm món
    mapped = {i: FS103_NAME_TO_GROUP[n] for i, n in lowered.items()
              if n in FS103_NAME_TO_GROUP}
    if mapped:
        return mapped, "fs103"

    raise RuntimeError(
        f"Không map được class của weights sang nhóm món: {sorted(lowered.values())[:8]}"
    )


def load() -> tuple[object | None, dict]:
    """Lazy-load model, thread-safe. Trả (model, meta)."""
    global _model, _meta, _cls_to_group
    if _model is not None:
        return _model, _meta
    with _lock:
        if _model is not None:
            return _model, _meta
        wp = weights_path()
        if wp is None:
            _meta = {"loaded": False,
                     "error": "Chưa có weights trong backend/models/ — xem README"}
            return None, _meta

        from ultralytics import YOLO
        import torch

        m = YOLO(str(wp))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        m.to(device)
        try:
            _cls_to_group, kind = _build_class_map(dict(m.names))
        except RuntimeError as e:
            _meta = {"loaded": False, "error": str(e)}
            return None, _meta

        _model = m
        _meta = {
            "loaded": True,
            "weights": wp.name,
            "weights_kind": kind,
            "device": device,
            "classes": len(m.names),
            "mapped_groups": sorted(set(_cls_to_group.values())),
        }
        return _model, _meta


def meta() -> dict:
    load()
    return _meta


def segment(bgr: np.ndarray, conf: float = 0.25, iou: float = 0.5) -> list[dict]:
    """Chạy segmentation, trả list vùng món (đã gán nhóm) sort theo diện tích."""
    model, m = load()
    if model is None:
        raise RuntimeError(m.get("error", "model chưa sẵn sàng"))

    res = model.predict(
        bgr, conf=conf, iou=iou, imgsz=640, verbose=False, retina_masks=True
    )[0]
    if res.masks is None or len(res.boxes) == 0:
        return []

    H, W = res.orig_shape
    masks = res.masks.data.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy().astype(int)
    confs = res.boxes.conf.cpu().numpy()
    boxes = res.boxes.xyxy.cpu().numpy()

    out = []
    for i in range(len(cls)):
        group = _cls_to_group.get(int(cls[i]))
        if group is None:          # class ngoài menu (nền, nước sốt, gia vị…)
            continue
        mk = masks[i]
        if mk.shape != (H, W):
            import cv2
            mk = cv2.resize(mk, (W, H), interpolation=cv2.INTER_NEAREST)
        mk = mk > 0.5
        area_px = int(mk.sum())
        if area_px < 200:
            continue
        x1, y1, x2, y2 = boxes[i]
        out.append({
            "group": group,
            "confidence": float(confs[i]),
            "area_px": area_px,
            "mask": mk,
            "bbox_px": [float(x1), float(y1), float(x2), float(y2)],
        })
    out.sort(key=lambda d: -d["area_px"])
    return out
