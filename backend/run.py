#!/usr/bin/env python3
"""Chạy API.

Mặc định KHÔNG bật reload: pipeline ghi ảnh + SQLite vào backend/storage/, nếu
watcher theo dõi cả thư mục này thì server sẽ tự restart ngay giữa lúc đang xử
lý một khay và request bị ngắt. Muốn reload khi dev thì đặt RELOAD=1 — lúc đó
chỉ theo dõi thư mục app/.
"""

import os

import uvicorn

if __name__ == "__main__":
    reload = os.environ.get("RELOAD") == "1"
    port = int(os.environ.get("PORT", "8002"))
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=reload,
        reload_dirs=["app"] if reload else None,
    )
