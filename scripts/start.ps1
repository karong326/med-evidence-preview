param(
    [string]$Python = '',
    [ValidateRange(1,65535)][int]$Port = 8001,
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $Python) {
    $localPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $localPython) { $Python = $localPython }
    else { $Python = (Get-Command python -ErrorAction Stop).Source }
}
Push-Location $projectRoot
try {
    # The selected checkout always supplies the source, even after a directory rename.
    $previousPythonPath = $env:PYTHONPATH
    $env:PYTHONPATH = Join-Path $projectRoot 'src'
    & $Python (Join-Path $PSScriptRoot 'check_environment.py') --cuda
    if ($LASTEXITCODE -ne 0) { throw 'Environment check failed; fix the listed settings first.' }
    if ($CheckOnly) { return }
    $listener = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
    if ($listener) { throw "Port $Port is already in use; no process was stopped." }
    & $Python -m med_evidence_agentic_rag.main --port $Port
    if ($LASTEXITCODE -ne 0) { throw "Service exited with code $LASTEXITCODE" }
} finally {
    $env:PYTHONPATH = $previousPythonPath
    Pop-Location
}
