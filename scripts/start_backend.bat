@echo off
title CellScope AI — Backend Server (FastAPI)
cd /d "%~dp0..\backend"

echo ======================================================================
echo             CellScope AI — FastAPI Backend Server
echo ======================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python was not found in your system PATH!
    echo Please install Python 3.10+ from https://www.python.org/
    pause
    exit /b 1
)

REM Check if virtual environment exists, else create it
if not exist "venv\" (
    echo [INFO] Creating Python virtual environment (venv)...
    python -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

echo [INFO] Checking / installing Python dependencies from requirement.txt...
pip install -r requirement.txt
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install backend dependencies.
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo [INFO] Starting Uvicorn server on http://localhost:8000 ...
echo [INFO] Interactive Swagger Docs: http://localhost:8000/docs
echo ======================================================================
echo.

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
pause
