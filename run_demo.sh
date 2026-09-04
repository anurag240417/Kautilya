#!/usr/bin/env bash
# ChainTrace Demo Runner
# Single command to build frontend and start the investigation server.
#
# Usage:
#   ./run_demo.sh              # Build frontend + start server
#   ./run_demo.sh --skip-build # Start server only (use existing frontend build)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
FRONTEND_DIR="${PROJECT_ROOT}/frontend"
BACKEND_HOST="${CHAINTRACE_HOST:-127.0.0.1}"
BACKEND_PORT="${CHAINTRACE_PORT:-8000}"

echo "═══════════════════════════════════════════════════════════════"
echo "  ChainTrace — Investigation Platform Demo"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Check Python virtual environment
if [ ! -d "${PROJECT_ROOT}/.venv" ]; then
    echo "⚠  Virtual environment not found. Creating..."
    python3 -m venv "${PROJECT_ROOT}/.venv"
    echo "✓  Virtual environment created."
fi

# Activate venv
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/.venv/bin/activate"

# Install backend dependencies if needed
if ! python -c "import backend" 2>/dev/null; then
    echo "⚠  Installing backend dependencies..."
    pip install -e "${PROJECT_ROOT}" --quiet
    echo "✓  Backend dependencies installed."
fi

# Build frontend unless --skip-build
if [[ "${1:-}" != "--skip-build" ]]; then
    echo ""
    echo "▸ Building frontend..."
    if [ ! -d "${FRONTEND_DIR}/node_modules" ]; then
        echo "  Installing npm dependencies..."
        (cd "${FRONTEND_DIR}" && npm install --silent)
    fi
    (cd "${FRONTEND_DIR}" && npm run build --silent)
    echo "✓  Frontend built → ${FRONTEND_DIR}/dist/"
else
    echo "▸ Skipping frontend build (--skip-build)"
    if [ ! -d "${FRONTEND_DIR}/dist" ]; then
        echo "⚠  Warning: frontend/dist/ not found. UI may not load."
    fi
fi

echo ""
echo "▸ Starting ChainTrace Investigation API..."
echo "  Host: ${BACKEND_HOST}"
echo "  Port: ${BACKEND_PORT}"
echo "  UI:   http://${BACKEND_HOST}:${BACKEND_PORT}/"
echo "  API:  http://${BACKEND_HOST}:${BACKEND_PORT}/api/v1/"
echo ""
echo "  Press Ctrl+C to stop."
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Start the server
exec python -m backend.main
