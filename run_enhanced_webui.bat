@echo off
echo.
echo ================================
echo   🌟 Supernova Enhanced Web UI   
echo ================================
echo.
echo 🛠️  Advanced Tools Included:
echo    🧮 Math solving (SymPy + Wolfram)
echo    💻 Code execution (Python + Safety)  
echo    📊 Data plotting (Matplotlib)
echo    🔍 Web search integration
echo    🛡️  Safety filtering
echo.
echo Starting Enhanced Supernova Web UI...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.8 or later.
    pause
    exit /b 1
)

REM Run the enhanced web UI launcher
python run_enhanced_webui.py

echo.
echo Enhanced Web UI stopped.
pause
