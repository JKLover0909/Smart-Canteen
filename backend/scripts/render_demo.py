"""Vẽ lại kết quả pipeline lên ảnh để kiểm tra bằng mắt.

Dùng đúng polygon phần trăm mà API trả cho frontend, nên nếu ảnh render đúng thì
overlay trên UI cũng đúng.

    python -m scripts.render_demo --out ../demo_out
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import kiosk  # noqa: E402

COLORS = [(255, 229, 0), (79, 195, 247), (67, 112, 255), (105, 187, 102),
          (200, 104, 186), (146, 98, 240), (0, 229, 255), (129, 213, 174)]


def draw(path: Path, out: Path) -> dict:
    raw = path.read_bytes()
    res = kiosk.scan(raw, persist=False)
    bgr = kiosk.decode_image(raw)
    H, W = bgr.shape[:2]
    canvas = bgr.copy()

    for i, it in enumerate(res["items"]):
        color = COLORS[i % len(COLORS)]
        overlay = canvas.copy()
        for poly in it["outlines"]:
            if len(poly) < 3:
                continue
            pts = np.array([[x / 100 * W, y / 100 * H] for x, y in poly], np.int32)
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(canvas, [pts], True, color, 2)
        canvas = cv2.addWeighted(overlay, 0.28, canvas, 0.72, 0)

        # nhãn ở tâm vùng lớn nhất — giống cách frontend đặt thẻ
        big = max(it["outlines"], key=len, default=None)
        if big and len(big) >= 3:
            cx = int(sum(p[0] for p in big) / len(big) / 100 * W)
            cy = int(sum(p[1] for p in big) / len(big) / 100 * H)
            txt = f"{it['name']} {it['quantity_display']} {it['subtotal']:,}d".replace(",", ".")
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            x0, y0 = max(2, cx - tw // 2), max(th + 6, cy)
            cv2.rectangle(canvas, (x0 - 3, y0 - th - 4), (x0 + tw + 3, y0 + 4), (18, 18, 28), -1)
            cv2.rectangle(canvas, (x0 - 3, y0 - th - 4), (x0 + tw + 3, y0 + 4), color, 1)
            cv2.putText(canvas, txt, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (255, 255, 255), 1, cv2.LINE_AA)

    # thanh tổng tiền + quyết định
    bar_h = 30
    canvas = cv2.copyMakeBorder(canvas, bar_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(28, 28, 44))
    ig = res["integrity"]
    head = f"{path.stem}  TONG {res['total']:,}d".replace(",", ".")
    tail = "AUTO" if ig["trusted"] else f"CAN XAC NHAN ({len(ig['flags'])} co)"
    cv2.putText(canvas, head, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (120, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, tail, (canvas.shape[1] - 210, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (135, 231, 126) if ig["trusted"] else (120, 190, 255), 1, cv2.LINE_AA)

    cv2.imwrite(str(out), canvas)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../frontend/public/samples")
    ap.add_argument("--out", default="../demo_out")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tiles = []
    for p in sorted(Path(args.src).glob("*.jpg")):
        res = draw(p, out / f"{p.stem}_annotated.jpg")
        print(f"{p.name}: {res['total']:,}d".replace(",", ".") +
              f"  {len(res['items'])} món  {res['latency_ms']}ms  "
              f"{'auto' if res['integrity']['trusted'] else 'cần xác nhận'}")
        tiles.append(cv2.imread(str(out / f"{p.stem}_annotated.jpg")))

    # ghép lưới 3x2 để xem nhanh
    if tiles:
        hh = 320
        tiles = [cv2.resize(t, (int(t.shape[1] * hh / t.shape[0]), hh)) for t in tiles]
        ww = max(t.shape[1] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, 0, 0, ww - t.shape[1],
                                    cv2.BORDER_CONSTANT, value=(0, 0, 0)) for t in tiles]
        rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
        wmax = max(r.shape[1] for r in rows)
        rows = [cv2.copyMakeBorder(r, 0, 0, 0, wmax - r.shape[1],
                                   cv2.BORDER_CONSTANT, value=(0, 0, 0)) for r in rows]
        cv2.imwrite(str(out / "grid.jpg"), np.vstack(rows))
        print(f"\n-> {out / 'grid.jpg'}")


if __name__ == "__main__":
    main()
