#!/usr/bin/env bash
# ==============================================================================
# CellScope AI — Backend Launcher (Linux / macOS)
# ==============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/../backend"

cd "$BACKEND_DIR"

echo "======================================================================"
echo "             CellScope AI — FastAPI Backend Server"
echo "======================================================================"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 was not found in PATH."
    exit 1
fi

# Virtual Environment
if [ ! -d "venv" ]; then
    echo "[INFO] Creating Python virtual environment (venv)..."
    python3 -m venv venv
fi

echo "[INFO] Activating virtual environment..."
source venv/bin/activate

echo "[INFO] Checking / installing Python dependencies..."
pip install -r requirement.txt

echo ""
echo "======================================================================"
echo "[INFO] Starting Uvicorn on http://localhost:8000"
echo "[INFO] Swagger Docs: http://localhost:8000/docs"
echo "======================================================================"
echo ""

exec uvicorn main:app --host 0.0.0.0 --port 8000 --reload
