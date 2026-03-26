#!/usr/bin/env bash
# Run all local processes: video worker + report worker + Streamlit (default :8501).
# Same roles as Render (render.yaml). Requires .env with AWS_BUCKET (and AWS keys as needed).
#
# Usage:
#   chmod +x scripts/run_local_stack.sh
#   ./scripts/run_local_stack.sh
#
# Optional:
#   STREAMLIT_PORT=8501 PYTHON=python3 ./scripts/run_local_stack.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
STREAMLIT_ADDR="${STREAMLIT_ADDR:-127.0.0.1}"

if [[ ! -f "$ROOT/app.py" ]]; then
  echo "ERROR: run from repo root (app.py not found)." >&2
  exit 1
fi

if ! command -v "$PY" &>/dev/null; then
  echo "ERROR: Python not found: $PY (set PYTHON=...)" >&2
  exit 1
fi

cleanup() {
  local ec=$?
  echo ""
  echo "Stopping workers..."
  for pid in "${WORKER_PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
  exit "$ec"
}
trap cleanup EXIT INT TERM

WORKER_PIDS=()

echo "=== [1/3] Video worker: $PY src/worker.py ==="
"$PY" src/worker.py &
WORKER_PIDS+=($!)
sleep 1

echo "=== [2/3] Report worker: $PY src/report_worker.py ==="
"$PY" src/report_worker.py &
WORKER_PIDS+=($!)
sleep 1

echo "=== [3/3] Streamlit: http://${STREAMLIT_ADDR}:${STREAMLIT_PORT} (Ctrl+C stops all) ==="
"$PY" -m streamlit run app.py --server.port "$STREAMLIT_PORT" --server.address "$STREAMLIT_ADDR"
