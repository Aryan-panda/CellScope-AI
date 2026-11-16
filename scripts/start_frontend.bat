@echo off
title CellScope AI — Frontend Application (React + Vite)
cd /d "%~dp0..\frontend"

echo ======================================================================
echo             CellScope AI — React + Vite Frontend
echo ======================================================================
echo.

REM Check if Node.js / npm is installed
npm --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] npm was not found in your system PATH!
    echo Please install Node.js 18+ from https://nodejs.org/
    pause
    exit /b 1
)

REM Check if node_modules exists, else install
if not exist "node_modules\" (
    echo [INFO] Installing Node.js packages (npm install)...
    call npm install
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to install npm packages.
        pause
        exit /b 1
    )
)

echo.
echo ======================================================================
echo [INFO] Starting Vite development server on http://localhost:5173 ...
echo ======================================================================
echo.

call npm run dev
pause
