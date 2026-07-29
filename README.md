# Smart-Canteen — Định giá khay thức ăn bằng AI

Demo hệ thống nhà ăn buffet tính tiền theo định lượng thực lấy: camera quét khay
→ nhận diện từng món → ước lượng định lượng → tra bảng giá → ra tổng tiền.

Trọng tâm: **nhận diện được món ăn** và **tính tiền theo lượng nhiều/ít**.

---

## 1. Bài toán có dataset & model sẵn để làm nhanh không?

Có. Hai bộ dữ liệu công khai ghép lại phủ đúng bài toán:

| Nguồn | Nội dung | Dùng để |
|---|---|---|
| [FoodSeg103](https://huggingface.co/datasets/EduardoPacheco/FoodSeg103) | 7.118 ảnh, mask **segmentation** 103 class nguyên liệu | Nhận diện + phân vùng từng món trên khay |
| [Nutrition5k](https://huggingface.co/datasets/arunapb/nutrition5k-foodseg103) | ~1.000 đĩa, **khối lượng gram đo bằng cân** cho từng nguyên liệu | Hiệu chỉnh diện tích → gram, đo sai số thật |

FoodSeg103 cho *vùng + tên món*, Nutrition5k cho *nhãn gram thật* — đúng hai thứ
cần để đi từ ảnh ra tiền. Class của FoodSeg103 khớp tốt menu nhà ăn Việt: `rice`,
`soup`, `egg`, `tofu`, `pork`, `fish`, `chicken duck`, `noodles`, `cabbage`,
`sausage`, `bread`, `milk`…

Model nền: **YOLO-seg** đã fine-tune trên FoodSeg103, tải sẵn từ Hugging Face
→ không phải train từ đầu.

### Đã thử và loại

Đo ở **mức nhóm món** trên 500 ảnh FoodSeg103 val, conf 0.25:

| Model | P | R | F1 | Kết luận |
|---|---|---|---|---|
| `arunapb/yolo11l-food-segmentation` | — | — | — | README ghi 103 class nhưng weights thật chỉ có **1 class** `food` → không nhận được tên món |
| `magnusdtd/yolov8-foodseg103` | 0.761 | 0.721 | 0.741 | Yếu hơn |
| **`arunapb/yolo26l-seg…nutrition5k`** | **0.886** | **0.880** | **0.883** | **Đang dùng** |
| `canteen-seg` (tự train, 16 nhóm) | 0.739 | 0.824 | 0.779 | Nhẹ hơn 7x (20MB), nhanh hơn — để dự phòng |

> Lưu ý ngược với trực giác: **thu gọn 103 class xuống 16 nhóm rồi train lại KHÔNG
> làm tăng độ chính xác.** Model 103-class có sẵn rồi map class → nhóm lúc inference
> cho kết quả tốt hơn (0.883 vs 0.779) và linh hoạt hơn. Cả hai đều dùng được — hệ
> thống tự nhận ra loại weights từ chính file `.pt`.

---

## 2. Bảng giá & cách định lượng

103 class nguyên liệu được gộp thành **16 nhóm món có giá**, mỗi nhóm gắn một cách
định lượng — đúng nguyên tắc "không dùng một công thức cho mọi món":

| Cách định lượng | Nhóm món | Ví dụ giá |
|---|---|---|
| **Đếm số lượng** | Trứng, đậu phụ, xúc xích, bánh mì, trái cây, đồ uống hộp | Trứng 5.000đ/quả |
| **Ước lượng khối lượng** | Thịt heo–bò, rau xanh, rau củ, nấm, khoai chiên | Thịt 150.000đ/kg |
| **Đếm miếng + kích thước** | Thịt gà/vịt, cá/hải sản | Gà 15.000đ/miếng |
| **Ước lượng thể tích** | Canh | 5.000đ/bát |
| **Theo mức ít/vừa/nhiều** | Cơm, mì/phở | 3.000 / 5.000 / 7.000đ |

Toàn bộ ở [`backend/app/menu_config.py`](backend/app/menu_config.py). Đổi giá hay
thêm món chỉ cần sửa file này — **không train lại**, vì việc map class → nhóm món
diễn ra ngay lúc inference.

---

## 3. Pipeline

Đúng thứ tự trong sơ đồ nghiệp vụ, mỗi bước là một module riêng:

| Bước | File |
|---|---|
| Kiểm tra chất lượng ảnh (mờ / tối / cháy sáng) | [`vision/quality.py`](backend/app/vision/quality.py) |
| Phát hiện vùng khay + lấy tỉ lệ px→cm² | [`vision/tray.py`](backend/app/vision/tray.py) |
| Phân vùng từng món + nhận diện tên món | [`vision/detector.py`](backend/app/vision/detector.py) |
| Ước lượng số lượng/thể tích → khối lượng | [`vision/quantify.py`](backend/app/vision/quantify.py) |
| Ánh xạ bảng giá → tính tiền từng món | [`vision/pricing.py`](backend/app/vision/pricing.py) |
| Kiểm tra tin cậy & bất thường | [`vision/integrity.py`](backend/app/vision/integrity.py) |
| Lưu ảnh, kết quả, giao dịch (SQLite) | [`vision/store.py`](backend/app/vision/store.py) |
| Điều phối toàn bộ | [`vision/pipeline.py`](backend/app/vision/pipeline.py) |

**Quy đổi ra gram không phụ thuộc chiều cao camera:** khay có kích thước vật lý cố
định (mặc định 40×30cm), nên diện tích khay tính bằng pixel cho ra tỉ lệ cm²/px
thật. Từ đó `gram = diện tích mask (cm²) × hệ số g/cm²`.

### Chống gian lận

Đề bài nêu rủi ro *dùng đồ ăn giá thấp phủ lên đồ ăn giá cao*. Hệ thống kiểm 5 dấu hiệu:

- `che_phu` — món đắt bị món rẻ (giá trị/cm² ≤ 50%) vây ≥60% viền **và** diện tích
  nhỏ bất thường so với món rẻ vây quanh
- `phan_manh` — món đắt bị tách ≥4 mảnh rời (nghi bị che một phần)
- `vung_la` — >50% vùng có đồ ăn mà không gán được nhãn
- `tin_cay_thap` — confidence dưới ngưỡng tự động duyệt
- `gia_bat_thuong` — tổng tiền ngoài khoảng thường gặp

Có cờ `high` hoặc confidence thấp → chuyển sang **yêu cầu xác nhận**, ngược lại
**tự động hiển thị**.

Kiểm tra logic bằng mask tổng hợp, độc lập với model (3/3 tình huống đúng: khay
bình thường không báo động, khay bị phủ và khay phân mảnh đều bị bắt):

```bash
cd backend && .venv/bin/python -m scripts.test_integrity
```

### Ngưỡng tin cậy 2 tầng

Với hệ thống thu tiền, **thu tiền món công nhân không lấy tệ hơn bỏ sót**, nên
ngưỡng phát hiện đặt cao hơn mặc định của YOLO. Dò trên 400 ảnh FoodSeg103 val:

| conf | Precision | Recall | F1 |
|---|---|---|---|
| 0.25 | 0.885 | 0.888 | 0.887 |
| **0.45** | **0.938** | **0.827** | **0.879** |
| 0.55 | 0.955 | 0.793 | 0.867 |
| 0.65 | 0.967 | 0.742 | 0.840 |

→ Chọn **0.45 để tính tiền**, **0.60 để tự động duyệt**. Món 0.45–0.60 vẫn tính
tiền nhưng phải xác nhận; món bị bỏ sót được cờ `vung_la` bắt lại.

---

## 4. Kết quả đo được

**Nhận diện món** — FoodSeg103 val, mức nhóm món, conf 0.45:

```
Precision 0.938   Recall 0.827   F1 0.879
Tốc độ    22ms/ảnh (RTX 5070 Ti) · toàn pipeline 26–60ms/khay
```

Model tự train `canteen-seg` (16 nhóm) — mask mAP50 tổng **0.516**, theo nhóm:

| Tốt | Yếu |
|---|---|
| Đồ uống hộp 0.80 · Rau xanh 0.80 · Cơm 0.77 | Nấm 0.34 · Trứng 0.38 · Thịt gà 0.39 |
| Trái cây 0.72 · Rau củ 0.65 · Mì phở 0.63 | Xúc xích 0.13 · Đậu phụ 0.08 |

Nhóm yếu là do dataset ít mẫu (đậu phụ chỉ 105 instance, xúc xích 352).

**Quy đổi gram** — hiệu chỉnh trên nhãn cân thật của Nutrition5k:

```
MAE 41g   MAPE 41.1%
Hệ số học được: com 0.71 g/cm² · rau_cu 1.12 · rau_xanh 0.89 · trai_cay 3.71
```

---

## 5. Giới hạn phải nói rõ

**a) Định giá theo gram bằng camera 2D: chưa cam kết được.** Sai số 41% là quá lớn
để tính tiền theo `gram × giá/kg`. Nguyên nhân bản chất: ảnh 2D không thấy chiều
cao, nên cơm xới cao và cơm dàn mỏng cho cùng diện tích. Muốn chính xác cần
**camera depth** hoặc **cân điện tử** đặt dưới khay.

**b) Chưa chứng minh được định giá theo mức ít/vừa/nhiều.** Script báo độ chính xác
mức 89–92%, nhưng **91.4% mẫu Nutrition5k rơi vào một mức duy nhất** ("ít"), nên
con số này chỉ bằng base-rate — lift âm (−0.11 trên test). Script tự đánh cờ
`informative: false` để không bị đọc sai. Muốn kết luận phải có dữ liệu suất ăn
thật của nhà ăn, trải đều cả ba mức.

**c) Hệ số g/cm² gắn với hình học của Nutrition5k** (đĩa Ø26cm, camera cố định).
Lắp camera thật **phải hiệu chỉnh lại bằng cân điện tử** — `scripts/calibrate.py`
là khung sẵn để làm, chỉ cần thay dữ liệu đầu vào.

**d) Chỉ 4/16 nhóm được hiệu chỉnh bằng dữ liệu** (cơm, rau củ, rau xanh, trái cây)
vì Nutrition5k không đủ mẫu cho các nhóm còn lại — các nhóm khác đang dùng hệ số
đặt tay.

**e) Dữ liệu là ảnh món ăn phương Tây**, không phải khay nhà ăn công nghiệp Việt
Nam. Ảnh mẫu trong demo là ảnh đĩa ăn thật nhưng không phải khay 5 ngăn. Triển khai
thật cần thu và gắn nhãn ảnh khay tại chính nhà ăn.

**f) Món ăn đổi theo ngày** → cần gắn nhãn lại và train định kỳ. Vì vậy pipeline lưu
lại toàn bộ ảnh đã quét (`backend/storage/images/`) để làm dữ liệu cho vòng huấn
luyện sau.

---

## 6. Chạy demo

```bash
./start-demo.sh
```

Meiko: **http://localhost:5180/#meiko** — bấm "Khay mẫu 1–6", tải ảnh, hoặc bật camera.

Chạy tay:

```bash
cd backend && python -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/python run.py
```

```bash
cd frontend && npm install && npm run dev
```

Weights (`backend/models/`) không nằm trong git, tải lại:

```bash
curl -L -o backend/models/foodseg103-seg.pt https://huggingface.co/arunapb/yolo26l-seg-foodseg103-nutrition5k/resolve/main/checkpoints/best.pt
```

### Train lại / hiệu chỉnh lại

```bash
cd backend
.venv/bin/python -m scripts.build_dataset          # FoodSeg103 -> YOLO-seg 16 nhóm
.venv/bin/python -m scripts.train --epochs 60      # ~35 phút trên RTX 5070 Ti
.venv/bin/python -m scripts.calibrate              # hiệu chỉnh diện tích -> gram
.venv/bin/python -m scripts.render_demo            # vẽ kết quả ra demo_out/
```

---

## 7. API

| Endpoint | Mô tả |
|---|---|
| `POST /api/kiosk/scan` | Upload ảnh khay → món, định lượng, tiền, cờ bất thường, log từng bước |
| `POST /api/kiosk/recalculate` | Nhân viên chỉnh định lượng → tính lại tiền |
| `POST /api/kiosk/confirm` | Xác nhận giao dịch |
| `GET /api/kiosk/menu` | Bảng giá + cách định lượng từng nhóm |
| `GET /api/kiosk/status` | Model đang dùng, thiết bị, thông số hiệu chỉnh |
| `GET /api/kiosk/transactions` | Lịch sử giao dịch + tỉ lệ tự động |

```bash
curl -X POST localhost:8000/api/kiosk/scan -F "file=@frontend/public/samples/tray2.jpg"
```

Phần dashboard dự báo suất ăn (`/api/forecast`, `/api/dashboard`) là dữ liệu mô
phỏng, không liên quan tới pipeline thị giác.

---

## 8. Công nghệ & giấy phép

- **Backend** FastAPI · Ultralytics YOLO · OpenCV · PyTorch CUDA · SQLite
- **Frontend** React 19 · Vite (overlay mask bằng SVG polygon)
- **Dữ liệu** FoodSeg103 (Apache-2.0) · Nutrition5k (CC-BY-4.0)
- **Weights `yolo26l-seg`**: repo gốc không khai báo license — **cần xác nhận trước
  khi dùng thương mại**. Model `canteen-seg` tự train từ FoodSeg103 là phương án
  thay thế nếu vướng license.
