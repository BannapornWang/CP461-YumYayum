@echo off
chcp 65001 > nul
echo ========================================================
echo   YumYayum (CP461) - Super-Resolution Web Application
echo ========================================================
echo.

if not exist ".venv312\Scripts\python.exe" (
    echo [ERROR] Virtual environment .venv312 not found!
    echo Please setup environment first or run setup script.
    pause
    exit /b 1
)

echo [INFO] Activating .venv312 and starting FastAPI server...
echo [INFO] Web App will open at: http://localhost:8000
echo.

start "" "http://localhost:8000"
.\.venv312\Scripts\python.exe -m uvicorn webapp.app:app --host 127.0.0.1 --port 8000 --reload

pause
