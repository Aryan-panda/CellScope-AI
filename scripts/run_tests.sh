#!/usr/bin/env bash
# ==============================================================================
# CellScope AI — Test Suite Runner (Linux / macOS)
# ==============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/../backend"

cd "$BACKEND_DIR"

echo "======================================================================"
echo "             CellScope AI — Automated Test Suite"
echo "======================================================================"

if [ -f "venv/bin/activate" ]; then
    echo "[INFO] Activating virtual environment..."
    source venv/bin/activate
fi

echo "[INFO] Running pytest test suite..."
python3 -m pytest tests/ -v -s || {
    echo ""
    echo "[INFO] Running tests directly via Python..."
    python3 tests/test_instance_segmentation.py
    python3 tests/test_uncertainty_pipeline.py
    python3 tests/test_api_endpoints.py
}
