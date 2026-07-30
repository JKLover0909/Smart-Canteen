#!/usr/bin/env bash
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "==> Starting Smart-Canteen backend..."
cd "$ROOT/backend"
if [ ! -f models/canteen-seg.pt ]; then
  echo "!! Thiếu models/canteen-seg.pt — Meiko sẽ không nhận diện được."
  echo "   Chạy: python -m scripts.build_dataset && python -m scripts.train"
  echo "   rồi copy runs/canteen-seg/weights/best.pt -> models/canteen-seg.pt"
fi
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt -q
fi
.venv/bin/python run.py &
BACKEND_PID=$!

echo "==> Starting Smart-Canteen frontend..."
cd "$ROOT/frontend"
if [ ! -d node_modules ]; then
  npm install -q
fi
npm run dev &
FRONTEND_PID=$!

echo ""
echo "✅ Smart-Canteen Demo"
echo "   Frontend: http://localhost:5180"
echo "   Backend:  http://localhost:8002/docs"
echo ""
echo "Press Ctrl+C to stop"

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT
wait
