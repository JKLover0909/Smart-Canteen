from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

DISHES = [
    {"id": 1, "name": "Phở bò", "category": "Món nước", "price": 35000, "popularity": 0.22},
    {"id": 2, "name": "Cơm gà", "category": "Món cơm", "price": 30000, "popularity": 0.18},
    {"id": 3, "name": "Bún chả", "category": "Món nước", "price": 32000, "popularity": 0.15},
    {"id": 4, "name": "Cơm sườn", "category": "Món cơm", "price": 28000, "popularity": 0.14},
    {"id": 5, "name": "Mì xào bò", "category": "Món mì", "price": 30000, "popularity": 0.12},
    {"id": 6, "name": "Cháo sườn", "category": "Món nước", "price": 25000, "popularity": 0.08},
    {"id": 7, "name": "Salad gà", "category": "Healthy", "price": 35000, "popularity": 0.06},
    {"id": 8, "name": "Cơm chay", "category": "Chay", "price": 25000, "popularity": 0.05},
]

SHIFTS = [
    {"id": "breakfast", "name": "Sáng", "start": "06:30", "end": "08:30", "base_diners": 420},
    {"id": "lunch", "name": "Trưa", "start": "11:00", "end": "13:00", "base_diners": 850},
    {"id": "dinner", "name": "Tối", "start": "17:00", "end": "19:00", "base_diners": 620},
]

INGREDIENTS = {
    1: [{"name": "Bánh phở", "unit": "kg", "per_serving": 0.18}, {"name": "Thịt bò", "unit": "kg", "per_serving": 0.12}, {"name": "Hành lá", "unit": "kg", "per_serving": 0.02}],
    2: [{"name": "Gạo", "unit": "kg", "per_serving": 0.15}, {"name": "Thịt gà", "unit": "kg", "per_serving": 0.14}, {"name": "Rau", "unit": "kg", "per_serving": 0.05}],
    3: [{"name": "Bún", "unit": "kg", "per_serving": 0.16}, {"name": "Thịt nướng", "unit": "kg", "per_serving": 0.13}, {"name": "Rau sống", "unit": "kg", "per_serving": 0.06}],
    4: [{"name": "Gạo", "unit": "kg", "per_serving": 0.15}, {"name": "Sườn heo", "unit": "kg", "per_serving": 0.16}, {"name": "Trứng", "unit": "quả", "per_serving": 0.5}],
    5: [{"name": "Mì", "unit": "kg", "per_serving": 0.14}, {"name": "Thịt bò", "unit": "kg", "per_serving": 0.12}, {"name": "Rau củ", "unit": "kg", "per_serving": 0.08}],
    6: [{"name": "Gạo", "unit": "kg", "per_serving": 0.08}, {"name": "Sườn heo", "unit": "kg", "per_serving": 0.10}, {"name": "Hành lá", "unit": "kg", "per_serving": 0.01}],
    7: [{"name": "Gà luộc", "unit": "kg", "per_serving": 0.12}, {"name": "Rau xà lách", "unit": "kg", "per_serving": 0.08}, {"name": "Sốt", "unit": "lít", "per_serving": 0.03}],
    8: [{"name": "Gạo", "unit": "kg", "per_serving": 0.14}, {"name": "Đậu phụ", "unit": "kg", "per_serving": 0.08}, {"name": "Rau củ", "unit": "kg", "per_serving": 0.10}],
}


def _shift_multiplier(shift_id: str, weekday: int) -> float:
    lunch_boost = 1.0 if shift_id == "lunch" else 0.85 if shift_id == "dinner" else 0.55
    weekday_factor = 0.75 if weekday >= 5 else 1.0
    monday_boost = 1.08 if weekday == 0 and shift_id == "lunch" else 1.0
    return lunch_boost * weekday_factor * monday_boost


def generate_historical(days: int = 90, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    today = date.today()
    rows = []

    for offset in range(days, 0, -1):
        d = today - timedelta(days=offset)
        weekday = d.weekday()
        is_holiday = d.month == 1 and d.day <= 3
        weather = rng.choice(["nắng", "mưa", "âm u"], p=[0.55, 0.25, 0.20])
        weather_factor = 0.88 if weather == "mưa" else 1.0

        for shift in SHIFTS:
            total_diners = int(
                shift["base_diners"]
                * _shift_multiplier(shift["id"], weekday)
                * weather_factor
                * (0.7 if is_holiday else 1.0)
                * rng.normal(1.0, 0.06)
            )
            total_diners = max(80, total_diners)

            weights = np.array([dish["popularity"] for dish in DISHES])
            if weekday == 4:
                weights[7] *= 1.35
            weights = weights / weights.sum()
            allocations = rng.multinomial(total_diners, weights)

            for dish, servings in zip(DISHES, allocations):
                noise = int(rng.normal(0, max(2, servings * 0.04)))
                actual = max(0, servings + noise)
                rows.append(
                    {
                        "date": d.isoformat(),
                        "weekday": weekday,
                        "shift": shift["id"],
                        "shift_name": shift["name"],
                        "dish_id": dish["id"],
                        "dish_name": dish["name"],
                        "category": dish["category"],
                        "servings": actual,
                        "weather": weather,
                        "registered": int(actual * rng.uniform(0.82, 0.95)),
                        "is_holiday": is_holiday,
                    }
                )

    return pd.DataFrame(rows)


HISTORY = generate_historical()
