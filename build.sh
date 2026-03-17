#!/bin/bash
set -e

echo "==> Upgrading pip..."
pip install --upgrade pip

echo "==> Clearing pip cache..."
pip cache purge || true

echo "==> Installing dependencies from worker_requirements.txt..."
pip install --no-cache-dir -r worker_requirements.txt

echo "==> Verifying MediaPipe installation..."
python -c "import mediapipe as mp; print(f'MediaPipe version: {mp.__version__}'); print(f'Has solutions: {hasattr(mp, \"solutions\")}')"

echo "==> Build complete!"



