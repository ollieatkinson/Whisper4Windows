"""
ASR backend protocol.
"""

from __future__ import annotations

from typing import Optional, Protocol

import numpy as np

from .types import BackendInfo, TranscriptionResult


class ASRBackend(Protocol):
    model_id: str
    is_loaded: bool

    def is_model_downloaded(self) -> bool:
        ...

    def load_model(self) -> bool:
        ...

    def transcribe_audio(
        self,
        audio_data: np.ndarray,
        language: Optional[str] = None,
        task: str = "transcribe",
    ) -> TranscriptionResult:
        ...

    def backend_info(self) -> BackendInfo:
        ...
