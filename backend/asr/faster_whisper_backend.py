"""
faster-whisper backend adapter.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from models.storage import get_models_root

from .types import BackendInfo, TranscriptionResult

logger = logging.getLogger(__name__)


def setup_cuda_paths() -> None:
    """Add CUDA library paths to PATH for bundled and optional runtime installs."""
    cuda_paths: list[Path] = []

    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).parent
        cuda_paths.append(exe_dir)
        cuda_paths.append(Path(sys._MEIPASS))
        cuda_paths.extend(
            [
                Path(sys._MEIPASS) / "nvidia" / "cublas" / "bin",
                Path(sys._MEIPASS) / "nvidia" / "cudnn" / "bin",
                Path(sys._MEIPASS) / "nvidia" / "cufft" / "bin",
                Path(sys._MEIPASS) / "nvidia" / "curand" / "bin",
                Path(sys._MEIPASS) / "nvidia" / "cusolver" / "bin",
                Path(sys._MEIPASS) / "nvidia" / "cusparse" / "bin",
                Path(sys._MEIPASS) / "nvidia" / "cuda_runtime" / "bin",
                Path(sys._MEIPASS) / "nvidia" / "cuda_nvrtc" / "bin",
            ]
        )

    appdata = Path(os.getenv("APPDATA") or os.path.expanduser("~"))
    gpu_libs_dir = appdata / "Whisper4Windows" / "gpu_libs"
    if gpu_libs_dir.exists():
        cuda_paths.extend(
            [
                gpu_libs_dir / "nvidia" / "cublas" / "bin",
                gpu_libs_dir / "nvidia" / "cudnn" / "bin",
                gpu_libs_dir / "nvidia" / "cufft" / "bin",
                gpu_libs_dir / "nvidia" / "curand" / "bin",
                gpu_libs_dir / "nvidia" / "cusolver" / "bin",
                gpu_libs_dir / "nvidia" / "cusparse" / "bin",
                gpu_libs_dir / "nvidia" / "cuda_runtime" / "bin",
                gpu_libs_dir / "nvidia" / "cuda_nvrtc" / "bin",
            ]
        )

    current_path = os.environ.get("PATH", "")
    for cuda_path in cuda_paths:
        if not cuda_path.exists():
            continue
        if str(cuda_path) not in current_path:
            current_path = str(cuda_path) + os.pathsep + current_path
        try:
            os.add_dll_directory(str(cuda_path))
        except Exception:
            pass

    os.environ["PATH"] = current_path


setup_cuda_paths()

try:
    from faster_whisper import WhisperModel

    FASTER_WHISPER_AVAILABLE = True
except Exception as exc:
    logger.warning("faster-whisper not available: %s", exc)
    WhisperModel = None  # type: ignore
    FASTER_WHISPER_AVAILABLE = False


class FasterWhisperBackend:
    def __init__(
        self,
        model_id: str,
        model_name: str,
        requested_device: str = "auto",
        compute_type: str = "auto",
        track_id: str = "ralph-wiggum",
    ) -> None:
        self.model_id = model_id
        self.model_name = model_name
        self.requested_device = requested_device
        self.compute_type = compute_type
        self.model = None
        self.is_loaded = False
        self.track_id = track_id
        self.fallback_reason = ""
        self._cuda_detected = False
        self.effective_device = requested_device
        self.models_root = get_models_root()

        if requested_device == "auto":
            self.effective_device = self._detect_device()
        else:
            self.effective_device = requested_device

        if compute_type == "auto":
            self.compute_type = self._detect_compute_type()

    @staticmethod
    def _model_cache_dir(model_name: str) -> Path:
        return get_models_root() / f"models--Systran--faster-whisper-{model_name}"

    @staticmethod
    def is_model_downloaded_static(model_name: str) -> bool:
        model_path = FasterWhisperBackend._model_cache_dir(model_name=model_name)
        if not model_path.exists():
            return False
        snapshots_dir = model_path / "snapshots"
        return snapshots_dir.exists() and any(snapshots_dir.iterdir())

    def is_model_downloaded(self) -> bool:
        return self.is_model_downloaded_static(self.model_name)

    def _detect_device(self) -> str:
        try:
            import ctranslate2

            cuda_count = ctranslate2.get_cuda_device_count()
            if cuda_count > 0:
                self._cuda_detected = True
                return "cuda"
        except Exception as exc:
            logger.warning("CUDA check failed: %s", exc)

        self._cuda_detected = False
        return "cpu"

    def _detect_compute_type(self) -> str:
        if self.effective_device == "cuda":
            return "float16"
        return "int8"

    def _cuda_compute_type_fallbacks(self) -> list[str]:
        return ["float16", "int8_float16", "int8"]

    def load_model(self) -> bool:
        if not FASTER_WHISPER_AVAILABLE:
            logger.error("faster-whisper is not installed")
            return False

        if self.is_loaded:
            return True

        try:
            if self.effective_device == "cuda":
                for compute in self._cuda_compute_type_fallbacks():
                    try:
                        self.model = WhisperModel(
                            self.model_name,
                            device=self.effective_device,
                            compute_type=compute,
                            download_root=str(self.models_root),
                        )
                        self.compute_type = compute
                        self.is_loaded = True
                        return True
                    except Exception as exc:
                        logger.warning("CUDA load failed for %s: %s", compute, exc)
                        continue

                self.effective_device = "cpu"
                self.compute_type = "int8"
                self.fallback_reason = "all CUDA compute types failed, fell back to CPU"

            self.model = WhisperModel(
                self.model_name,
                device=self.effective_device,
                compute_type=self.compute_type,
                download_root=str(self.models_root),
            )
            self.is_loaded = True
            return True
        except Exception as exc:
            logger.error("Failed to load faster-whisper model: %s", exc)
            return False

    def transcribe_audio(
        self,
        audio_data: np.ndarray,
        language: Optional[str] = None,
        task: str = "transcribe",
    ) -> TranscriptionResult:
        if not self.is_loaded and not self.load_model():
            return TranscriptionResult(success=False, error="failed to load model")

        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        if len(audio_data.shape) > 1:
            audio_data = audio_data.flatten()

        try:
            segments, info = self.model.transcribe(  # type: ignore[union-attr]
                audio_data,
                language=language,
                task=task,
                beam_size=1,
                best_of=1,
                temperature=0.0,
                vad_filter=False,
                condition_on_previous_text=False,
            )

            parsed_segments: list[dict] = []
            full_text = ""
            for segment in segments:
                parsed_segments.append(
                    {
                        "start": float(segment.start),
                        "end": float(segment.end),
                        "text": segment.text.strip(),
                    }
                )
                full_text += segment.text

            return TranscriptionResult(
                success=True,
                text=full_text.strip(),
                segments=parsed_segments,
                language=str(getattr(info, "language", language or "")),
                language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
                duration=float(getattr(info, "duration", 0.0) or 0.0),
            )
        except Exception as exc:
            error_str = str(exc)
            if "cuda" in error_str.lower() or "cublas64_12.dll" in error_str.lower():
                self.effective_device = "cpu"
                self.compute_type = "int8"
                self.is_loaded = False
                self.fallback_reason = "runtime CUDA error during transcription, switched to CPU"
                if self.load_model():
                    return self.transcribe_audio(audio_data=audio_data, language=language, task=task)
            return TranscriptionResult(success=False, error=error_str)

    def backend_info(self) -> BackendInfo:
        provider = "CUDAExecutionProvider" if self.effective_device == "cuda" else "CPUExecutionProvider"
        return BackendInfo(
            runtime="faster_whisper",
            provider=provider,
            requested_device=self.requested_device,
            effective_device=self.effective_device,
            fallback_reason=self.fallback_reason,
            model_id=self.model_id,
            track_id=self.track_id,
        )
