"""
PyInstaller build script for Whisper4Windows backend
This creates a standalone executable that includes Python and all dependencies
"""

import PyInstaller.__main__
import os
import sys
from pathlib import Path

if sys.version_info < (3, 12):
    print("ERROR: Backend build now targets Python 3.12+.")
    print("Use: py -3.12 -m venv backend\\.venv312")
    print("Then run BUILD_BACKEND.bat again.")
    sys.exit(1)

# Get the directory where this script is located
backend_dir = os.path.dirname(os.path.abspath(__file__))

# CUDA libraries are now downloaded on-demand by gpu_manager.py
# We no longer bundle them to keep installer small (~200MB instead of ~1.3GB)
print("INFO: CUDA libraries will be downloaded on first run if GPU is detected")
print("INFO: This keeps the installer size small (~200MB vs ~1.3GB)")

binary_includes = []

PyInstaller.__main__.run([
    'main.py',
    '--name=whisper-backend',
    '--onefile',
    '--console',  # Show console for debugging
    '--icon=NONE',

    # Include necessary data files
    '--add-data=requirements.txt;.',
    '--add-data=models/manifest.json;models',

    # Include NVIDIA CUDA DLLs
    *binary_includes,

    # Hidden imports that PyInstaller might miss
    '--hidden-import=faster_whisper',
    '--hidden-import=ctranslate2',
    '--hidden-import=sounddevice',
    '--hidden-import=numpy',
    '--hidden-import=uvicorn',
    '--hidden-import=fastapi',
    '--hidden-import=pydantic',
    '--hidden-import=onnxruntime',
    '--hidden-import=asr.selector',
    '--hidden-import=asr.faster_whisper_backend',
    '--hidden-import=asr.onnx_backend',
    '--hidden-import=asr.parakeet_transformers_backend',
    '--hidden-import=models.registry',
    '--hidden-import=models.downloader',
    '--hidden-import=models.hf_client',
    '--hidden-import=models.storage',
    '--hidden-import=models.performance',
    '--hidden-import=transformers',
    '--hidden-import=torch',

    # Exclude unnecessary packages to reduce size
    '--exclude-module=matplotlib',
    '--exclude-module=tkinter',

    # Output directory
    f'--distpath={os.path.join(backend_dir, "dist")}',
    f'--workpath={os.path.join(backend_dir, "build")}',
    f'--specpath={backend_dir}',
])
