@echo off
chcp 65001 > nul
echo ========================================================
echo   YumYayum (CP461) - Super-Resolution Web Application
echo ========================================================
echo.

set "PYTHON_EXE="
if exist ".venv312\Scripts\python.exe" (
    set "PYTHON_EXE=.venv312\Scripts\python.exe"
) else if exist "C:\Program Files\Python312\python.exe" (
    set "PYTHON_EXE=C:\Program Files\Python312\python.exe"
) else (
    where python.exe >nul 2>nul
    if %errorlevel%==0 set "PYTHON_EXE=python.exe"
)

if "%PYTHON_EXE%"=="" (
    echo [ERROR] Python executable not found!
    echo Please setup .venv312 or install Python 3.12 first.
    pause
    exit /b 1
)

echo [INFO] Using Python: %PYTHON_EXE%
echo [INFO] Web App will open at: http://127.0.0.1:8000
echo.

start "" "http://127.0.0.1:8000"
"%PYTHON_EXE%" -m uvicorn webapp.app:app --host 127.0.0.1 --port 8000 --reload

pause
