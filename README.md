# AI Upscaler ×4

Phóng to ảnh gấp 4 lần bằng [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) x4plus, chạy trên GPU của máy. Có web app kéo-thả chạy cục bộ và CLI xử lý hàng loạt; kết quả là PNG gắn DPI để in.

## Cài đặt nhanh (Windows)

Nhấp đúp `run.bat`. Lần đầu nó tự tạo môi trường Python riêng (`.venv`), tự cài PyTorch (bản GPU nếu có card NVIDIA) và thư viện, tự tải model, rồi mở app. Các lần sau mở ngay. Chỉ cần máy có sẵn [Python](https://www.python.org/downloads/).

## Cài đặt thủ công

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124   # GPU NVIDIA; bỏ --index-url nếu dùng CPU
pip install -r requirements.txt
```

Lần chạy đầu tự tải model (~64 MB) vào `models/`.

## Web app

```bash
python app.py     # hoặc nhấp đúp run.bat trên Windows
```

Mở http://127.0.0.1:7860, kéo thả ảnh, chỉnh DPI/tile, bấm **Tải về**.

## CLI

```bash
python cli.py                       # mọi ảnh trong ./input -> ./output
python cli.py a.jpg b.png --dpi 300 --tile 256
```

Giảm `--tile` nếu thiếu VRAM.

## Trang giới thiệu

Thư mục `docs/` là trang GitHub Pages (Settings → Pages → Branch `main`, folder `/docs`).
