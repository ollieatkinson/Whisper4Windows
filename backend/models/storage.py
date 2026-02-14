"""
Shared model storage helpers.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Iterable


def _default_data_root() -> Path:
    if getattr(sys, "frozen", False):
        appdata = Path(os.getenv("APPDATA") or os.path.expanduser("~"))
        return appdata / "Whisper4Windows"

    if os.name == "nt":
        appdata = Path(os.getenv("APPDATA") or os.path.expanduser("~"))
        return appdata / "Whisper4Windows"

    return Path.home() / ".local" / "share" / "Whisper4Windows"


def get_data_root() -> Path:
    override = os.getenv("W4W_DATA_DIR", "").strip()
    root = Path(override).expanduser() if override else _default_data_root()
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_models_root() -> Path:
    override = os.getenv("W4W_MODELS_DIR", "").strip()
    root = Path(override).expanduser() if override else get_data_root() / "models"
    root.mkdir(parents=True, exist_ok=True)
    return root


def sanitize_model_id(model_id: str) -> str:
    if not model_id:
        raise ValueError("model_id cannot be empty")
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "--", model_id.strip())
    return cleaned.strip("-")


def get_model_dir(model_id: str) -> Path:
    path = get_models_root() / sanitize_model_id(model_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_model_lock_path(model_id: str) -> Path:
    return get_model_dir(model_id) / ".download.lock"


def get_model_marker_path(model_id: str) -> Path:
    return get_model_dir(model_id) / ".download.complete.json"


def has_required_files(model_id: str, file_paths: Iterable[str]) -> bool:
    model_dir = get_model_dir(model_id)
    for rel_path in file_paths:
        path = model_dir / rel_path
        if not path.exists():
            return False
    return True
