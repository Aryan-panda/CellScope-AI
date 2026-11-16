#!/usr/bin/env bash
# ==============================================================================
# CellScope AI — Full-Stack Launcher (Linux / macOS)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "======================================================================"
echo "             CellScope AI — Full-Stack Launcher"
echo "======================================================================"

# Start backend in background
echo "[INFO] Starting backend..."
bash "$SCRIPT_DIR/start_backend.sh" &
BACKEND_PID=$!

# Trap exit to kill backend when frontend exits
cleanup() {
    echo ""
    echo "[INFO] Shutting down backend (PID: $BACKEND_PID)..."
    kill $BACKEND_PID 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# Start frontend in foreground
echo "[INFO] Starting frontend..."
bash "$SCRIPT_DIR/start_frontend.sh"
