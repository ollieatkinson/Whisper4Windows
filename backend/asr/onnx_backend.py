"""
ONNX Runtime backend scaffold for DirectML/CPU execution.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from .provider_resolver import select_onnx_provider
from .types import BackendInfo, TranscriptionResult

logger = logging.getLogger(__name__)


class OnnxBackend:
    def __init__(
        self,
        model_id: str,
        model_dir: Path,
        requested_device: str = "auto",
        track_id: str = "ralph-wiggum",
    ) -> None:
        self.model_id = model_id
        self.model_dir = model_dir
        self.requested_device = requested_device
        self.track_id = track_id
        self.is_loaded = False
        self.fallback_reason = ""
        self.provider = ""
        self.effective_device = requested_device
        self.providers_to_use: list[str] = []
        self.session = None
        self.model_path = None

    def _find_model_path(self) -> Optional[Path]:
        candidates = [
            self.model_dir / "model.onnx",
            self.model_dir / "encoder.onnx",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def is_model_downloaded(self) -> bool:
        return self._find_model_path() is not None

    def load_model(self) -> bool:
        try:
            import onnxruntime as ort
        except Exception as exc:
            logger.error("onnxruntime unavailable: %s", exc)
            self.fallback_reason = f"onnxruntime unavailable: {exc}"
            return False

        model_path = self._find_model_path()
        if model_path is None:
            self.fallback_reason = (
                "no ONNX model file found (expected model.onnx or encoder.onnx in model directory)"
            )
            return False

        available_providers = ort.get_available_providers()
        selection = select_onnx_provider(self.requested_device, available_providers)
        self.providers_to_use = selection.providers
        self.provider = selection.provider
        self.effective_device = selection.effective_device or "cpu"
        self.fallback_reason = selection.fallback_reason

        if not self.providers_to_use:
            if not self.fallback_reason:
                self.fallback_reason = "no usable ONNX Runtime provider"
            return False

        try:
            self.session = ort.InferenceSession(str(model_path), providers=self.providers_to_use)
            self.model_path = model_path
            self.is_loaded = True
            return True
        except Exception as exc:
            self.fallback_reason = f"failed to create ONNX session: {exc}"
            logger.error("failed to create ONNX session: %s", exc)
            return False

    def transcribe_audio(
        self,
        audio_data: np.ndarray,
        language: Optional[str] = None,
        task: str = "transcribe",
    ) -> TranscriptionResult:
        if not self.is_loaded and not self.load_model():
            return TranscriptionResult(success=False, error=self.fallback_reason or "failed to load ONNX model")

        if task == "translate":
            return TranscriptionResult(success=False, error="translate task is not supported for ONNX CTC models")

        # Full feature extraction + CTC decode is model-layout-specific and will be implemented in T3/T4.
        return TranscriptionResult(
            success=False,
            error=(
                "ONNX backend loaded but transcription pipeline is not configured for this artifact layout yet. "
                "Provide an ONNX CTC artifact set compatible with the upcoming decoder pipeline."
            ),
        )

    def backend_info(self) -> BackendInfo:
        return BackendInfo(
            runtime="onnx",
            provider=self.provider,
            requested_device=self.requested_device,
            effective_device=self.effective_device,
            fallback_reason=self.fallback_reason,
            model_id=self.model_id,
            track_id=self.track_id,
        )
