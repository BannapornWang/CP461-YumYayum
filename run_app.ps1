# YumYayum (CP461) - Super-Resolution Web Application Launcher
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  YumYayum (CP461) - Super-Resolution Web Application" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

$pythonExe = ".\.venv312\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    Write-Host "[ERROR] Virtual environment .venv312 not found!" -ForegroundColor Red
    Write-Host "Please ensure the virtual environment is set up." -ForegroundColor Yellow
    exit 1
}

Write-Host "[INFO] Starting FastAPI server on http://localhost:8000 ..." -ForegroundColor Green
Start-Process "http://localhost:8000"

& $pythonExe -m uvicorn webapp.app:app --host 127.0.0.1 --port 8000 --reload
