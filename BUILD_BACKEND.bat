@echo off
setlocal

echo ============================================
echo Building Whisper4Windows Backend Executable
echo ============================================
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
)

cd /d "%BACKEND_DIR%"

echo Installing backend build dependencies...
"%BACKEND_PY%" -m pip install --upgrade pip
"%BACKEND_PY%" -m pip install -r requirements.txt

REM Install PyInstaller if not already installed
echo Installing PyInstaller...
"%BACKEND_PY%" -m pip install pyinstaller

REM Build the backend executable
echo.
echo Building backend executable...
"%BACKEND_PY%" build_backend.py

REM Check if build was successful
if exist "dist\whisper-backend.exe" (
    echo.
    echo ============================================
    echo Build successful!
    echo Backend executable: backend\dist\whisper-backend.exe
    echo ============================================

    REM Copy to Tauri binaries folder
    if not exist "..\frontend\src-tauri\binaries" mkdir "..\frontend\src-tauri\binaries"
    copy /Y "dist\whisper-backend.exe" "..\frontend\src-tauri\binaries\whisper-backend-x86_64-pc-windows-msvc.exe"
    echo Copied to Tauri binaries folder
) else (
    echo.
    echo ============================================
    echo Build failed! Check the output above for errors.
    echo ============================================
)

echo.
pause
