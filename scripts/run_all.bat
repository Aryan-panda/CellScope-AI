@echo off
title CellScope AI — Full-Stack Launcher
cd /d "%~dp0"

echo ======================================================================
echo              CellScope AI — Full-Stack Launcher
echo ======================================================================
echo.
echo Launching Backend (FastAPI on Port 8000) in a new window...
start "CellScope AI - Backend (Port 8000)" cmd /k "call "%~dp0start_backend.bat""

echo.
echo Launching Frontend (Vite on Port 5173) in a new window...
start "CellScope AI - Frontend (Port 5173)" cmd /k "call "%~dp0start_frontend.bat""

echo.
echo ======================================================================
echo [SUCCESS] Both servers are launching!
echo  - Backend API: http://localhost:8000 (Docs: http://localhost:8000/docs)
echo  - Frontend UI: http://localhost:5173
echo ======================================================================
echo.
echo You can minimize this window or close it.
pause
