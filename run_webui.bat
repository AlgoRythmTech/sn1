@echo off
echo.
echo ========================
echo   🌟 Supernova Web UI   
echo ========================
echo.
echo Starting Supernova Web UI...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.8 or later.
    pause
    exit /b 1
)

REM Run the web UI launcher
python run_webui.py

echo.
echo Web UI stopped.
pause
