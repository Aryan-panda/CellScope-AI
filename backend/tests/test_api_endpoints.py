"""
backend/tests/test_api_endpoints.py
===================================
Integration test suite for the FastAPI service endpoints:
- GET  /api/health
- GET  /api/presets
- POST /api/export
- POST /api/segment (validation and error handling)
"""

import sys
from pathlib import Path
import base64

TEST_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TEST_DIR.parent
SRC_DIR = BACKEND_DIR / "src"

for p in (str(BACKEND_DIR), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi.testclient import TestClient
import main


def get_client():
    return TestClient(main.app)


def test_health_endpoint():
    """Verify /api/health returns 200, valid status, device, and Dice score."""
    with TestClient(main.app) as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "device" in data
        assert "dice_score" in data
        assert data["dice_score"] == "88.47%"
        print("  [PASS] /api/health endpoint verified.")


def test_presets_endpoint():
    """Verify /api/presets returns 200 and list of benchmark preset objects."""
    with TestClient(main.app) as client:
        response = client.get("/api/presets")
        assert response.status_code == 200
        presets = response.json()
        assert isinstance(presets, list)
        assert len(presets) > 0
        first_preset = presets[0]
        assert "id" in first_preset
        assert "display_name" in first_preset
        assert "description" in first_preset
        assert "thumbnail_b64" in first_preset
        print(f"  [PASS] /api/presets returned {len(presets)} presets.")


def test_export_endpoint_csv():
    """Verify /api/export returns streaming CSV content with attachment header."""
    with TestClient(main.app) as client:
        dummy_csv = "id,area,perimeter\n1,100,40.0"
        b64_data = base64.b64encode(dummy_csv.encode("utf-8")).decode("utf-8")

        payload = {
            "export_type": "csv",
            "data_b64": b64_data,
            "filename": "test_morphology.csv",
        }
        response = client.post("/api/export", json=payload)
        assert response.status_code == 200
        assert "text/csv" in response.headers.get("content-type", "")
        assert "test_morphology.csv" in response.headers.get("content-disposition", "")
        assert response.content.decode("utf-8") == dummy_csv
        print("  [PASS] /api/export CSV export verified.")


def test_segment_validation_error():
    """Verify /api/segment fails with 400 when neither file nor preset_id is passed."""
    with TestClient(main.app) as client:
        response = client.post(
            "/api/segment",
            data={
                "threshold": 0.50,
                "algorithm": "watershed",
                "min_area": 20,
                "min_distance": 5,
                "overlay_opacity": 0.40,
            },
        )
        assert response.status_code == 400
        assert "Exactly one of 'file' or 'preset_id'" in response.json()["detail"]
        print("  [PASS] /api/segment parameter validation verified.")


if __name__ == "__main__":
    print("=== Running FastAPI API Endpoints Test Suite ===")
    test_health_endpoint()
    test_presets_endpoint()
    test_export_endpoint_csv()
    test_segment_validation_error()
    print("\n>>> ALL API ENDPOINT TESTS PASSED SUCCESSFULLY! <<<")
