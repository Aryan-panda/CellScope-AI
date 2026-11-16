@echo off
title CellScope AI — Test Suite Runner
cd /d "%~dp0..\backend"

echo ======================================================================
echo             CellScope AI — Automated Test Suite
echo ======================================================================
echo.

REM Activate virtual environment if present
if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment...
    call venv\Scripts\activate.bat
)

echo [INFO] Running pytest test suite...
python -m pytest tests/ -v -s

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ======================================================================
    echo [SUCCESS] All verification tests passed!
    echo ======================================================================
) else (
    echo.
    echo [WARNING] Pytest encountered errors or pytest is not yet installed.
    echo Running tests directly via Python...
    echo.
    python tests/test_instance_segmentation.py
    python tests/test_uncertainty_pipeline.py
    python tests/test_api_endpoints.py
)

pause
