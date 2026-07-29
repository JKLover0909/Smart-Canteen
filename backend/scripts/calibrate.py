"""Hiệu chỉnh hệ số diện tích -> khối lượng bằng nhãn gram thật của Nutrition5k.

Nutrition5k có khối lượng (gram) đo bằng cân cho từng nguyên liệu của mỗi đĩa.
Ta dùng nó để học hệ số g/cm² cho từng nhóm món, thay vì đoán bằng tay.

Quy trình:
  1. Gộp nguyên liệu của mỗi đĩa về nhóm món -> gram thật theo nhóm.
  2. Chạy model segmentation -> diện tích mask theo nhóm.
  3. Ước lượng scale px -> cm² bằng cách coi đĩa là hình tròn đường kính
     PLATE_DIAMETER_CM (giả định của bộ Nutrition5k, camera cố định).
  4. Hệ số g/cm² = median(gram / diện tích cm²) — median để chịu outlier.
  5. Đánh giá MAE / MAPE trên split test (không dùng để fit).

LƯU Ý: hệ số này gắn với hình học camera + đĩa của Nutrition5k. Khi lắp camera
thật ở nhà ăn phải hiệu chỉnh lại bằng cân điện tử — script này là khung để làm
việc đó, chỉ cần thay parquet đầu vào bằng dữ liệu cân thật.

    python -m scripts.calibrate
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.menu_config import MENU_GROUPS  # noqa: E402
from app.vision import detector  # noqa: E402

PLATE_DIAMETER_CM = 26.0
PLATE_AREA_CM2 = np.pi * (PLATE_DIAMETER_CM / 2) ** 2
MIN_SAMPLES = 8

# tên nguyên liệu Nutrition5k -> nhóm món (khớp theo nhãn fs103_classes có sẵn)
FS103_NAME_TO_GROUP: dict[str, str] = {}


def _build_name_map() -> None:
    import json as _json
    id2label = {int(k): v.strip() for k, v in
                _json.load(open("../data/foodseg103/id2label.json")).items()}
    from app.menu_config import FS103_TO_GROUP
    for cid, grp in FS103_TO_GROUP.items():
        if cid in id2label:
            FS103_NAME_TO_GROUP[id2label[cid].lower()] = grp


def plate_scale(bgr: np.ndarray) -> float:
    """cm² trên mỗi pixel, suy từ diện tích đĩa."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(cv2.GaussianBlur(gray, (9, 9), 0), 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = gray.shape
    area = max((cv2.contourArea(c) for c in cnts), default=0.0)
    if area < 0.08 * H * W:
        area = 0.55 * H * W          # fallback: đĩa chiếm ~55% khung
    return PLATE_AREA_CM2 / area


def collect(parquet: str) -> list[dict]:
    """Trả list mẫu (group, area_cm2, grams) dùng được cho hiệu chỉnh."""
    rows = pq.read_table(parquet).to_pylist()
    samples: list[dict] = []

    for i in range(0, len(rows), 8):
        chunk = rows[i:i + 8]
        imgs = [np.array(Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB"))[:, :, ::-1]
                for r in chunk]
        for bgr, row in zip(imgs, chunk):
            gt: dict[str, float] = defaultdict(float)
            for cls, ing in zip(row["fs103_classes"], row["ingredients"]):
                g = FS103_NAME_TO_GROUP.get(str(cls).strip().lower())
                if g:
                    gt[g] += float(ing["grams"])
            if not gt:
                continue

            regions = detector.segment(np.ascontiguousarray(bgr), conf=0.30)
            pred: dict[str, int] = defaultdict(int)
            for r in regions:
                pred[r["group"]] += r["area_px"]
            if not pred:
                continue

            # chỉ hiệu chỉnh khi tập nhóm khớp nhau -> diện tích gán đúng món
            if set(pred) != set(gt):
                continue

            scale = plate_scale(np.ascontiguousarray(bgr))
            for g, area_px in pred.items():
                area_cm2 = area_px * scale
                if area_cm2 < 3.0:
                    continue
                samples.append({"group": g, "area_cm2": area_cm2, "grams": gt[g]})
    return samples


def fit(samples: list[dict]) -> dict:
    per: dict[str, list[float]] = defaultdict(list)
    for s in samples:
        per[s["group"]].append(s["grams"] / s["area_cm2"])
    out = {}
    for g, ratios in per.items():
        if len(ratios) < MIN_SAMPLES:
            continue
        out[g] = {
            "g_per_cm2": round(float(np.median(ratios)), 4),
            "n_samples": len(ratios),
            "ratio_p25": round(float(np.percentile(ratios, 25)), 4),
            "ratio_p75": round(float(np.percentile(ratios, 75)), 4),
        }
    return out


def evaluate(samples: list[dict], coefs: dict) -> dict:
    err: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for s in samples:
        g = s["group"]
        k = coefs[g]["g_per_cm2"] if g in coefs else MENU_GROUPS[g].get("g_per_cm2", 0.8)
        pred = np.clip(s["area_cm2"] * k,
                       MENU_GROUPS[g].get("min_g", 0),
                       MENU_GROUPS[g].get("max_g", 10**6))
        err[g].append((abs(pred - s["grams"]), s["grams"]))
    rep = {}
    all_ae, all_gt = [], []
    for g, pairs in err.items():
        ae = np.array([p[0] for p in pairs]); gt = np.array([p[1] for p in pairs])
        rep[g] = {
            "n": len(pairs),
            "mae_g": round(float(ae.mean()), 1),
            "mape_pct": round(float((ae / np.maximum(gt, 1)).mean() * 100), 1),
            "mean_gt_g": round(float(gt.mean()), 1),
        }
        all_ae += list(ae); all_gt += list(gt)
    if all_ae:
        ae = np.array(all_ae); gt = np.array(all_gt)
        rep["_overall"] = {
            "n": len(all_ae),
            "mae_g": round(float(ae.mean()), 1),
            "mape_pct": round(float((ae / np.maximum(gt, 1)).mean() * 100), 1),
        }
    return rep


def level_accuracy(samples: list[dict], coefs: dict) -> dict:
    """Định giá theo mức ít/vừa/nhiều có chịu được sai số gram không?

    Dùng ngưỡng mức của nhóm 'com' (150g / 250g) làm thang chung.

    CẢNH BÁO ĐỌC SỐ: phải so với `base_rate_acc` (tỉ lệ của mức phổ biến nhất).
    Suất ăn trong Nutrition5k rất nhỏ nên gần như tất cả rơi vào mức "ít" — khi
    đó `exact_level_acc` cao chỉ là base-rate, KHÔNG chứng minh định giá theo mức
    là khả thi. Chỉ số chỉ có ý nghĩa khi `informative` = true, tức cần dữ liệu
    suất ăn thật trải đều cả ba mức.
    """
    edges = [g for _, g, _ in MENU_GROUPS["com"]["levels"][:-1]]

    def bucket(g: float) -> int:
        return sum(g > e for e in edges)

    hit = tot = 0
    off_by_one = 0
    gt_buckets: list[int] = []
    for s in samples:
        g = s["group"]
        if MENU_GROUPS[g]["mode"] != "weight":
            continue
        k = coefs[g]["g_per_cm2"] if g in coefs else MENU_GROUPS[g].get("g_per_cm2", 0.8)
        pred = float(np.clip(s["area_cm2"] * k,
                             MENU_GROUPS[g].get("min_g", 0),
                             MENU_GROUPS[g].get("max_g", 10**6)))
        b_pred, b_gt = bucket(pred), bucket(s["grams"])
        gt_buckets.append(b_gt)
        tot += 1
        hit += b_pred == b_gt
        off_by_one += abs(b_pred - b_gt) <= 1

    if not tot:
        return {"n": 0, "informative": False}

    counts = [gt_buckets.count(i) for i in range(len(edges) + 1)]
    base = max(counts) / tot
    acc = hit / tot
    return {
        "n": tot,
        "levels": ["<150g", "150-250g", ">250g"],
        "gt_level_counts": counts,
        "exact_level_acc": round(acc, 3),
        "within_1_level_acc": round(off_by_one / tot, 3),
        "base_rate_acc": round(base, 3),
        "lift_over_base": round(acc - base, 3),
        # base-rate > 80% -> tập mẫu lệch hẳn về một mức, số đo không nói lên gì
        "informative": bool(base < 0.80),
        "note": ("Chỉ số CÓ nghĩa." if base < 0.80 else
                 "Chỉ số KHÔNG có nghĩa: mẫu dồn vào một mức, độ chính xác chỉ "
                 "bằng base-rate. Cần dữ liệu suất ăn thật trải đều 3 mức."),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="../data/nutrition5k/train.parquet")
    ap.add_argument("--test", default="../data/nutrition5k/test.parquet")
    ap.add_argument("--out", default="calib/area_to_gram.json")
    args = ap.parse_args()

    _build_name_map()
    print("model:", detector.meta())

    print("\n-- thu mẫu train --")
    tr = collect(args.train)
    print(f"   {len(tr)} mẫu (group, area, gram)")
    coefs = fit(tr)
    for g, v in sorted(coefs.items()):
        print(f"   {g:12} g/cm²={v['g_per_cm2']:<7} n={v['n_samples']}")

    print("\n-- đánh giá trên test --")
    te = collect(args.test)
    rep = evaluate(te, coefs)
    for g, v in sorted(rep.items()):
        print(f"   {g:12} n={v['n']:<4} MAE={v.get('mae_g')}g  MAPE={v.get('mape_pct')}%")

    lvl = level_accuracy(te, coefs)
    lvl_train = level_accuracy(tr, coefs)
    print(f"\n-- định giá theo mức ít/vừa/nhiều (ngưỡng {lvl['levels']}) --")
    for tag, v in (("test", lvl), ("train", lvl_train)):
        print(f"   {tag:5} n={v['n']:<4} đúng mức={v['exact_level_acc']} "
              f"(base-rate={v['base_rate_acc']}, lift={v['lift_over_base']:+}) "
              f"phân bố mức={v['gt_level_counts']}")
    print(f"   => {lvl['note']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "source": "nutrition5k (median gram/cm², đĩa Ø26cm)",
        "plate_diameter_cm": PLATE_DIAMETER_CM,
        "n_train_samples": len(tr),
        "groups": coefs,
        "eval_on_test": rep,
        "level_pricing_on_test": lvl,
        "level_pricing_on_train": lvl_train,
        "caveat": "Hệ số gắn với hình học camera/đĩa của Nutrition5k. "
                  "Khi lắp camera thật phải hiệu chỉnh lại bằng cân điện tử.",
    }, indent=2, ensure_ascii=False))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
