#!/usr/bin/env bash
# Render free tier assigns the HTTP port via $PORT at runtime.
set -euo pipefail
PORT="${PORT:-8000}"
cd "$(dirname "$0")"
exec python -m uvicorn backend.app.main:app --host 0.0.0.0 --port "$PORT"
