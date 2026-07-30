#!/usr/bin/env python3
"""Serve Smart-Canteen frontend/dist + proxy /api → backend (Jetson-friendly).

Ubuntu 20.04 / glibc cũ trên Jetson AGX thường không chạy được Vite/Rollup
native arm64 mới. Dùng bản build sẵn + FastAPI static thay cho `npm run dev`.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "frontend" / "dist"
API = os.environ.get("CANTEEN_API", "http://127.0.0.1:8002").rstrip("/")
PORT = int(os.environ.get("CANTEEN_UI_PORT", "5180"))

app = FastAPI(title="Smart-Canteen UI")


@app.api_route(
    "/api/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
)
async def proxy_api(path: str, request: Request):
    body = await request.body()
    url = f"{API}/api/{path}"
    if request.url.query:
        url += f"?{request.url.query}"
    req = urllib.request.Request(url, data=body if body else None, method=request.method)
    for k, v in request.headers.items():
        if k.lower() not in ("host", "content-length"):
            req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            content = resp.read()
            headers = {
                k: v
                for k, v in resp.headers.items()
                if k.lower()
                not in ("transfer-encoding", "content-encoding", "content-length")
            }
            return Response(
                content=content,
                status_code=resp.status,
                headers=headers,
                media_type=resp.headers.get("content-type"),
            )
    except urllib.error.HTTPError as e:
        return Response(e.read(), status_code=e.code)
    except Exception as e:
        return Response(str(e), status_code=502)


@app.get("/")
def index():
    return FileResponse(DIST / "index.html")


if DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(DIST), html=True), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
