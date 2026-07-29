"""Bước: Kiểm tra chất lượng ảnh.

Chặn sớm các ảnh không dùng được (mờ, tối, cháy sáng) trước khi chạy model —
tránh tính tiền sai trên ảnh rác.

Độ nét đo bằng variance of Laplacian, nhưng chỉ số này phụ thuộc mạnh vào độ
phân giải (ảnh to luôn cho variance thấp hơn). Nên ảnh được resize về cạnh dài
WORK_EDGE trước khi đo, để một ngưỡng dùng chung cho mọi camera.
"""

from __future__ import annotations

import cv2
import numpy as np

WORK_EDGE = 640         # cạnh dài chuẩn hoá trước khi đo độ nét
BLUR_MIN = 60.0         # variance of Laplacian ở kích thước chuẩn hoá
BRIGHT_MIN = 40.0
BRIGHT_MAX = 225.0
CLIPPED_MAX_PCT = 25.0  # % pixel cháy sáng tối đa


def check(bgr: np.ndarray) -> dict:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    if max(h, w) != WORK_EDGE:
        s = WORK_EDGE / max(h, w)
        norm = cv2.resize(gray, (max(1, int(w * s)), max(1, int(h * s))),
                          interpolation=cv2.INTER_AREA)
    else:
        norm = gray

    blur = float(cv2.Laplacian(norm, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())
    clipped = float((gray >= 250).mean() * 100)

    issues: list[str] = []
    if blur < BLUR_MIN:
        issues.append(f"Ảnh mờ (nét={blur:.0f} < {BLUR_MIN:.0f})")
    if brightness < BRIGHT_MIN:
        issues.append(f"Ảnh quá tối (sáng={brightness:.0f})")
    if brightness > BRIGHT_MAX:
        issues.append(f"Ảnh quá sáng (sáng={brightness:.0f})")
    if clipped > CLIPPED_MAX_PCT:
        issues.append(f"Cháy sáng {clipped:.0f}% diện tích")

    return {
        "passed": not issues,
        "blur_score": round(blur, 1),
        "brightness": round(brightness, 1),
        "contrast": round(contrast, 1),
        "clipped_pct": round(clipped, 1),
        "issues": issues,
    }
