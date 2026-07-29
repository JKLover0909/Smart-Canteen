"""Bước: Phát hiện vùng khay.

Khay là vật sáng, lớn, gần như chữ nhật ở giữa khung. Tìm được vùng khay cho 2 việc:
  1. Crop bỏ nền (bàn, tay công nhân) trước khi phân vùng món.
  2. Lấy tỉ lệ px -> cm² : khay có kích thước vật lý cố định (khai báo trong
     TRAY_W_CM/TRAY_H_CM), nên diện tích khay tính bằng pixel cho ta scale thật.
     Đây là mấu chốt để ước lượng gram không phụ thuộc chiều cao lắp camera.
"""

from __future__ import annotations

import cv2
import numpy as np

TRAY_W_CM = 40.0
TRAY_H_CM = 30.0
TRAY_AREA_CM2 = TRAY_W_CM * TRAY_H_CM

MIN_TRAY_FRAC = 0.12   # khay phải chiếm >=12% khung mới coi là khay


def detect(bgr: np.ndarray) -> dict:
    """Trả về bbox khay + hệ số cm2/px. Fallback: coi cả khung là khay."""
    H, W = bgr.shape[:2]
    frame_area = float(H * W)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in contours:
        area = cv2.contourArea(c)
        if area / frame_area < MIN_TRAY_FRAC:
            continue
        x, y, w, h = cv2.boundingRect(c)
        rect_fill = area / float(w * h)       # khay -> gần chữ nhật -> fill cao
        if rect_fill < 0.55:
            continue
        if best is None or area > best[0]:
            best = (area, (x, y, w, h), rect_fill)

    if best is None:
        return {
            "found": False,
            "bbox": [0, 0, W, H],
            "tray_area_px": frame_area,
            "cm2_per_px": TRAY_AREA_CM2 / frame_area,
            "note": "Không tách được viền khay — dùng toàn khung làm tham chiếu",
        }

    area, (x, y, w, h), fill = best
    return {
        "found": True,
        "bbox": [int(x), int(y), int(w), int(h)],
        "tray_area_px": float(area),
        "cm2_per_px": TRAY_AREA_CM2 / float(area),
        "rect_fill": round(float(fill), 3),
        "frame_frac": round(area / frame_area, 3),
    }
