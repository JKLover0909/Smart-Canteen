from __future__ import annotations

from datetime import date
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import kiosk as kiosk_svc
from .data import DISHES, SHIFTS
from .forecast import compute_ingredients, dashboard_stats, history_summary, predict_day
from .vision import store

app = FastAPI(
    title="Smart-Canteen API",
    description="FoodsAI — Demo dự đoán định lượng suất ăn nhà ăn thông minh",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"service": "Smart-Canteen", "status": "ok", "version": "0.1.0-demo"}


@app.get("/api/health")
def health():
    return {"status": "healthy"}


@app.get("/api/dishes")
def get_dishes():
    return {"dishes": DISHES}


@app.get("/api/shifts")
def get_shifts():
    return {"shifts": SHIFTS}


@app.get("/api/dashboard")
def get_dashboard():
    return dashboard_stats()


@app.get("/api/forecast")
def get_forecast(
    target_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    weather: Optional[str] = Query(None, description="nắng | mưa | âm u"),
):
    parsed = date.fromisoformat(target_date) if target_date else None
    forecast = predict_day(parsed, weather)
    ingredients = compute_ingredients(forecast)
    return {**forecast, "ingredients": ingredients}


@app.get("/api/history")
def get_history(days: int = Query(30, ge=7, le=90)):
    return {"days": days, "records": history_summary(days)}


# ── Kiosk AI: quét khay -> nhận diện món -> định lượng -> tính tiền ──────────


class RecalcItem(BaseModel):
    group: str
    grams: float | None = None
    count: int | None = None
    pieces: int | None = None
    bowls: int | None = None
    area_cm2: float = 0.0
    n_regions: int = 1
    confidence: float = 1.0
    outlines: list = []


class RecalcBody(BaseModel):
    items: list[RecalcItem]


class ConfirmBody(BaseModel):
    transaction_id: str
    total_vnd: int | None = None


@app.get("/api/kiosk/menu")
def kiosk_menu():
    return kiosk_svc.get_menu()


@app.get("/api/kiosk/status")
def kiosk_status():
    return kiosk_svc.model_status()


@app.post("/api/kiosk/scan")
async def kiosk_scan(
    file: UploadFile = File(..., description="Ảnh khay từ camera RGB"),
    canteen_id: str = Form("NA1"),
    shift: str = Form("Trưa"),
    worker_id: str | None = Form(None),
    conf: float = Form(0.45),
    persist: bool = Form(True),
):
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Ảnh trống")
    try:
        return kiosk_svc.scan(
            raw, canteen_id=canteen_id, shift=shift,
            worker_id=worker_id, conf=conf, persist=persist,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from e


@app.post("/api/kiosk/recalculate")
def kiosk_recalc(body: RecalcBody):
    return kiosk_svc.recalculate([i.model_dump() for i in body.items])


@app.post("/api/kiosk/confirm")
def kiosk_confirm(body: ConfirmBody):
    return kiosk_svc.confirm(body.transaction_id, body.total_vnd)


@app.get("/api/kiosk/transactions")
def kiosk_transactions(limit: int = Query(20, ge=1, le=200)):
    return {"stats": store.stats(), "transactions": store.recent(limit)}
