"""
Whisper4Windows ASR compatibility facade.
Keeps the historical WhisperEngine API while routing to pluggable backends.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

from asr.faster_whisper_backend import FasterWhisperBackend
from asr.selector import create_backend
from models.downloader import get_download_manager
from models.registry import get_model_registry
from models.storage import has_required_files

logger = logging.getLogger(__name__)


def _as_bool(raw: str, default: bool) -> bool:
    value = (raw or "").strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    return default


class WhisperEngine:
    """
    Compatibility engine used by `backend/main.py`.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "auto",
        model_id: Optional[str] = None,
    ) -> None:
        self.registry = get_model_registry()
        self.download_manager = get_download_manager()
        self.model_spec = self.registry.resolve_model_spec(model_id=model_id, model_size=model_size)

        self.model_id = str(self.model_spec["model_id"])
        self.model_size = str(self.model_spec["legacy_model_size"])  # legacy field consumed by existing UI calls
        self.device = device
        self.compute_type = compute_type
        self._original_device = device
        self._last_error = ""
        self.is_loaded = False

        self.backend = create_backend(
            model_spec=self.model_spec,
            requested_device=device,
            compute_type=compute_type,
            track_id=self.registry.track_id,
        )

    @property
    def family(self) -> str:
        return str(self.model_spec.get("family", ""))

    @property
    def runtime(self) -> str:
        return str(self.model_spec.get("runtime", ""))

    @property
    def supports_translate(self) -> bool:
        return bool(self.model_spec.get("supports_translate", False))

    @property
    def supported_devices(self) -> list[str]:
        raw = self.model_spec.get("supported_devices", [])
        if isinstance(raw, list):
            return [str(item) for item in raw]
        return []

    def backend_info(self) -> dict:
        info = self.backend.backend_info().to_dict()
        if self._last_error and not info.get("fallback_reason"):
            info["fallback_reason"] = self._last_error
        return info

    def is_model_downloaded(self, model_size: str = None, model_id: str = None) -> bool:
        spec = self.registry.resolve_model_spec(model_id=model_id, model_size=model_size)
        runtime = str(spec.get("runtime", "")).lower()
        source = str(spec.get("download", {}).get("source", "")).lower()

        if runtime == "faster_whisper":
            return FasterWhisperBackend.is_model_downloaded_static(str(spec.get("backend_model_name", "")))

        if source == "huggingface":
            required_files = self.registry.required_download_files(spec["model_id"])
            file_paths = [str(item.get("path", "")).strip() for item in required_files if str(item.get("path", "")).strip()]
            if not file_paths:
                return False
            return has_required_files(spec["model_id"], file_paths)

        return self.backend.is_model_downloaded()

    def _ensure_external_artifacts(self) -> bool:
        source = str(self.model_spec.get("download", {}).get("source", "")).lower()
        if source != "huggingface":
            return True

        if self.is_model_downloaded(model_id=self.model_id):
            return True

        auto_download = _as_bool(os.getenv("W4W_AUTO_DOWNLOAD_HF_MODELS", "0"), False)
        if not auto_download:
            self._last_error = (
                f"model artifacts for '{self.model_id}' are not downloaded. "
                "Use POST /models/download first, or set W4W_AUTO_DOWNLOAD_HF_MODELS=1."
            )
            return False

        status = self.download_manager.ensure_downloaded(model_id=self.model_id, include_large_files=True)
        if status.get("state") != "completed":
            self._last_error = str(status.get("message", "model download failed"))
            return False
        return True

    def load_model(self) -> bool:
        if self.is_loaded:
            return True

        self._last_error = ""

        if not self._ensure_external_artifacts():
            logger.error(self._last_error)
            return False

        success = self.backend.load_model()
        self.is_loaded = bool(success)
        info = self.backend.backend_info()
        if info.effective_device:
            self.device = info.effective_device
        if info.fallback_reason:
            self._last_error = info.fallback_reason
        return self.is_loaded

    def transcribe_audio(
        self,
        audio_data: np.ndarray,
        language: Optional[str] = None,
        task: str = "transcribe",
    ) -> dict:
        if not self.is_loaded and not self.load_model():
            return {
                "success": False,
                "error": self._last_error or "failed to load model",
                "text": "",
            }

        result = self.backend.transcribe_audio(audio_data=audio_data, language=language, task=task)
        payload = result.to_dict()
        payload.setdefault("text", "")
        payload.setdefault("success", False)
        return payload

    def transcribe_file(
        self,
        audio_file: str,
        language: Optional[str] = None,
        task: str = "transcribe",
    ) -> dict:
        try:
            import soundfile as sf

            audio_data, sample_rate = sf.read(audio_file)
            if sample_rate != 16000:
                from scipy import signal

                num_samples = int(len(audio_data) * 16000 / sample_rate)
                audio_data = signal.resample(audio_data, num_samples)
            return self.transcribe_audio(audio_data=audio_data, language=language, task=task)
        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
                "text": "",
            }


_whisper_engine = None


def get_whisper_engine(model_size: str = "base", model_id: Optional[str] = None) -> WhisperEngine:
    global _whisper_engine
    if (
        _whisper_engine is None
        or _whisper_engine.model_size != model_size
        or (model_id is not None and _whisper_engine.model_id != model_id)
    ):
        _whisper_engine = WhisperEngine(model_size=model_size, model_id=model_id)
    return _whisper_engine
