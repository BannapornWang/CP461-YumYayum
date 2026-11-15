# YumYayum (CP461) - Super-Resolution Web Application Launcher
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  YumYayum (CP461) - Super-Resolution Web Application" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

$candidates = @(
    ".\.venv312\Scripts\python.exe",
    "C:\Program Files\Python312\python.exe"
)

$pythonExe = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $pythonExe) {
    $cmd = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($cmd) { $pythonExe = $cmd.Source }
}

if (-not $pythonExe) {
    Write-Host "[ERROR] Python executable not found! Checked .venv312, C:\Program Files\Python312\python.exe, and python.exe in PATH." -ForegroundColor Red
    exit 1
}

Write-Host "[INFO] Using Python: $pythonExe" -ForegroundColor Gray
Write-Host "[INFO] Starting FastAPI server on http://127.0.0.1:8000 ..." -ForegroundColor Green
Start-Process "http://127.0.0.1:8000"

& $pythonExe -m uvicorn webapp.app:app --host 127.0.0.1 --port 8000 --reload
