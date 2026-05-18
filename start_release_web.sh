#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
  echo "[ERROR] Missing .venv. Run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export FLAGS_use_mkldnn="${FLAGS_use_mkldnn:-0}"
export OCR_MAX_PAGES="${OCR_MAX_PAGES:-all}"
export OCR_FORCE_LOCAL="${OCR_FORCE_LOCAL:-1}"
export OCR_FALLBACK_ON_ZERO_RECORDS="${OCR_FALLBACK_ON_ZERO_RECORDS:-1}"
export APP_HOST="${APP_HOST:-127.0.0.1}"
export APP_PORT="${APP_PORT:-8010}"

".venv/bin/python" -m uvicorn integrated_test_app:app --host "$APP_HOST" --port "$APP_PORT"
