@echo off
setlocal

echo ========================================
echo   Whisper4Windows - Quick Start
echo ========================================
echo.

set "ROOT_DIR=%~dp0"
set "BACKEND_DIR=%ROOT_DIR%backend"
set "BACKEND_VENV=%BACKEND_DIR%\.venv312"
set "BACKEND_PY=%BACKEND_VENV%\Scripts\python.exe"
if "%ROOT_DIR:~0,2%"=="\\" (
    echo [setup] UNC workspace detected. Using shared local backend venv.
    set "BACKEND_VENV=%LOCALAPPDATA%\Whisper4Windows\venvs\backend312"
    set "BACKEND_PY=%BACKEND_VENV%\Scripts\python.exe"
)

if not exist "%BACKEND_PY%" (
    echo [setup] Python 3.12 backend environment not found. Creating .venv312...
    if not exist "%BACKEND_VENV%" mkdir "%BACKEND_VENV%"
    py -3.12 -m venv "%BACKEND_VENV%"
    if errorlevel 1 (
        echo ERROR: Failed to create Python 3.12 venv. Ensure Python 3.12 is installed: py -0p
        exit /b 1
    )
    "%BACKEND_PY%" -m pip install --upgrade pip
    "%BACKEND_PY%" -m pip install -r "%BACKEND_DIR%\requirements.txt"
    if errorlevel 1 (
        echo ERROR: Failed to install backend dependencies.
        exit /b 1
    )
)

set "CUDA_PATHS=%BACKEND_VENV%\Lib\site-packages\nvidia\cublas\bin;%BACKEND_VENV%\Lib\site-packages\nvidia\cudnn\bin;C:\Program Files\NVIDIA\CUDNN\v9.13\bin\13.0"

REM Start Backend (Python 3.12 venv)
echo [1/2] Starting Python Backend...
start "Whisper4Windows Backend" powershell -NoExit -Command "$env:PATH += ';%CUDA_PATHS%'; Set-Location '%BACKEND_DIR%'; & '%BACKEND_PY%' main.py"
timeout /t 3 /nobreak >nul

REM Start Frontend (with console for logging)
echo [2/2] Starting Frontend App...
start "Whisper4Windows Frontend" cmd /k "%ROOT_DIR%frontend\src-tauri\target\release\app.exe"

echo.
echo ========================================
echo   App Started!
echo ========================================
echo.
echo Backend: Check "Whisper4Windows Backend" window
echo   Look for: "CUDA is available!" for GPU mode
echo Frontend: System tray icon (bottom-right)
echo   Look for: CPU/GPU selector buttons
echo.
echo Press any key to close this window...
pause >nul
