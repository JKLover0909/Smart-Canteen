# CLAUDE.md

Guidance for Claude Code (and similar agents) working in this repository.

## What this repo is

Demo hệ thống nhà ăn buffet tính tiền theo định lượng: camera quét khay → YOLO-seg nhận diện món (fine-tune trên FoodSeg103) → ước lượng gram/thể tích → tra bảng giá → tổng tiền.

```
backend/app/menu_config.py   # bảng giá + cách định lượng 16 nhóm món — sửa giá/thêm món chỉ cần sửa file này
backend/app/vision/          # pipeline nhận diện + ước lượng định lượng
backend/calib/               # hiệu chỉnh diện tích -> gram
backend/models/              # weights .pt — KHÔNG commit, tải lại bằng curl (xem README mục 6)
backend/scripts/             # build_dataset / train / calibrate / render_demo
frontend/                    # React 19 + Vite, overlay mask SVG polygon
```

## Ràng buộc quan trọng

- **License weights**: `yolo26l-seg-foodseg103-nutrition5k` (model đang dùng chính) không có license rõ ràng từ tác giả gốc trên Hugging Face — **không giả định được dùng thương mại tự do**. Nếu cần thay thế, dùng `canteen-seg` (tự train, nhẹ hơn 7x, có sẵn trong repo).
- Weights/`data/`/`backend/runs/`/`backend/storage/`/`demo_out/` đã nằm trong `.gitignore` — không dùng `git add -f` để commit đè.
- `menu_config.py` là single source of truth cho giá — không hard-code giá ở nơi khác (frontend, API response) mà không đồng bộ lại từ đây.
- Phần `/api/forecast`, `/api/dashboard` là dữ liệu mô phỏng, không liên quan pipeline thị giác — đừng nhầm là số liệu thật khi debug.

## Kiểm thử an toàn

Không có GPU/webcam/model weights thật trong môi trường agent:
```bash
python -c "import ast; ast.parse(open('backend/app/menu_config.py').read())"
```
Không chạy `run.py`/`scripts/train.py` thật (cần CUDA + dataset đã tải).
