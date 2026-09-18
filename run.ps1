# Mineazy WMS - Windows launcher.  Seeds (first run) then starts the API.
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$venvPy = Join-Path $root ".venv\Scripts\python.exe"
$py = if (Test-Path $venvPy) { $venvPy } else { "python" }

Push-Location $root
if (-not (Test-Path "wms.db")) {
    Write-Host "==> Seeding demo database" -ForegroundColor Cyan
    & $py -m wms.scripts.seed
}
Write-Host "==> API on http://127.0.0.1:8000  (Swagger: /docs)" -ForegroundColor Cyan
Write-Host "    Console:  $py -m wms.console        (add --demo for a headless tour)"
& $py -m uvicorn wms.api.main:app --host 127.0.0.1 --port 8000
Pop-Location
