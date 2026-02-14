param(
    [switch]$InstallDirectML
)

$ErrorActionPreference = "Stop"

$backendDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvDir = Join-Path $backendDir ".venv312"
$usingSharedVenv = $false

# Python venv creation/execution on UNC paths can fail on Windows.
if ($backendDir.StartsWith("\\")) {
    $venvDir = Join-Path $env:LOCALAPPDATA "Whisper4Windows\venvs\backend312"
    $usingSharedVenv = $true
}

$pythonExe = Join-Path $venvDir "Scripts\python.exe"

Write-Host "== Whisper4Windows backend bootstrap (Python 3.12) =="
Write-Host "Backend dir: $backendDir"
Write-Host "Venv dir:    $venvDir"
if ($usingSharedVenv) {
    Write-Host "Note: UNC workspace detected; using shared local venv."
}

$pyLauncher = Get-Command py -ErrorAction SilentlyContinue
if (-not $pyLauncher) {
    throw "Python launcher 'py' not found. Install Python 3.12 and ensure py.exe is on PATH."
}

if (-not (Test-Path $pythonExe)) {
    $parent = Split-Path -Parent $venvDir
    if (-not (Test-Path $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    Write-Host "Creating .venv312 with Python 3.12..."
    & py -3.12 -m venv $venvDir
}

if (-not (Test-Path $pythonExe)) {
    throw "Failed to create Python 3.12 virtual environment at $venvDir"
}

Write-Host "Installing backend dependencies..."
& $pythonExe -m pip install --upgrade pip
& $pythonExe -m pip install -r (Join-Path $backendDir "requirements.txt")

if ($InstallDirectML) {
    Write-Host "Installing torch-directml..."
    & $pythonExe -m pip install torch-directml
}

Write-Host "Done. Backend can be started with:"
Write-Host "  $pythonExe main.py"
