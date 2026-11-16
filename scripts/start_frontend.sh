#!/usr/bin/env bash
# ==============================================================================
# CellScope AI — Frontend Launcher (Linux / macOS)
# ==============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$SCRIPT_DIR/../frontend"

cd "$FRONTEND_DIR"

echo "======================================================================"
echo "             CellScope AI — React + Vite Frontend"
echo "======================================================================"

# Check npm
if ! command -v npm &> /dev/null; then
    echo "[ERROR] npm was not found in PATH."
    exit 1
fi

if [ ! -d "node_modules" ]; then
    echo "[INFO] Installing npm dependencies..."
    npm install
fi

echo ""
echo "======================================================================"
echo "[INFO] Starting Vite dev server on http://localhost:5173"
echo "======================================================================"
echo ""

exec npm run dev
