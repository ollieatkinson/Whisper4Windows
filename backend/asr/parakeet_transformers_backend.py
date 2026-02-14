"""
Parakeet backend using Hugging Face Transformers.
"""

from __future__ import annotations

import logging
import io
import sys
import contextlib
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .types import BackendInfo, TranscriptionResult

logger = logging.getLogger(__name__)


class ParakeetTransformersBackend:
    def __init__(
        self,
        model_id: str,
        model_dir: Path,
        model_spec: dict[str, Any],
        requested_device: str = "auto",
        track_id: str = "ralph-wiggum",
    ) -> None:
        self.model_id = model_id
        self.model_dir = model_dir
        self.model_spec = model_spec
        self.requested_device = requested_device
        self.track_id = track_id

        self.is_loaded = False
        self.fallback_reason = ""
        self.provider = ""
        self.effective_device = requested_device
        self._torch_device = "cpu"
        self._dtype = None
        self.processor = None
        self.model = None
        self.sample_rate = 16000

    def is_model_downloaded(self) -> bool:
        required = [
            self.model_dir / "config.json",
            self.model_dir / "preprocessor_config.json",
            self.model_dir / "tokenizer.json",
            self.model_dir / "model.safetensors",
        ]
        return all(path.exists() for path in required)

    def _resolve_device(self) -> tuple[object, str, str]:
        try:
            import torch
        except Exception as exc:
            self.fallback_reason = f"torch unavailable: {exc}"
            return "cpu", "CPUExecutionProvider", "cpu"

        requested = (self.requested_device or "auto").strip().lower()

        if requested == "cuda":
            if torch.cuda.is_available():
                return "cuda", "CUDAExecutionProvider", "cuda"
            self.fallback_reason = "cuda requested but not available; falling back to CPU"
            return "cpu", "CPUExecutionProvider", "cpu"

        if requested == "directml":
            try:
                import torch_directml  # type: ignore

                dml_device = torch_directml.device()
                return dml_device, "DmlExecutionProvider", "directml"
            except Exception:
                self.fallback_reason = "directml requested but torch-directml unavailable; falling back to CPU"
                return "cpu", "CPUExecutionProvider", "cpu"

        if requested == "cpu":
            return "cpu", "CPUExecutionProvider", "cpu"

        # auto
        if torch.cuda.is_available():
            self.fallback_reason = "auto selected CUDA"
            return "cuda", "CUDAExecutionProvider", "cuda"

        try:
            import torch_directml  # type: ignore

            dml_device = torch_directml.device()
            self.fallback_reason = "auto selected DirectML"
            return dml_device, "DmlExecutionProvider", "directml"
        except Exception:
            pass

        self.fallback_reason = "auto selected CPU"
        return "cpu", "CPUExecutionProvider", "cpu"

    def load_model(self) -> bool:
        for stream_name in ("stdout", "stderr"):
            stream = getattr(sys, stream_name, None)
            try:
                if stream is not None and hasattr(stream, "reconfigure"):
                    stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

        try:
            import torch
            from transformers import AutoModelForCTC, AutoProcessor
        except Exception as exc:
            self.fallback_reason = f"transformers backend dependencies unavailable: {exc}"
            logger.error(self.fallback_reason)
            return False

        torch_device, provider, effective_device = self._resolve_device()
        self._torch_device = torch_device
        self.provider = provider
        self.effective_device = effective_device

        repo_id = str(self.model_spec.get("download", {}).get("repo_id", "")).strip()
        load_source: str
        local_only = False
        if self.is_model_downloaded():
            load_source = str(self.model_dir)
            local_only = True
        elif repo_id:
            load_source = repo_id
            local_only = False
            if not self.fallback_reason:
                self.fallback_reason = "local artifacts missing; loading from Hugging Face cache"
        else:
            self.fallback_reason = "no local artifacts and no repo_id configured"
            return False

        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.processor = AutoProcessor.from_pretrained(load_source, local_files_only=local_only)
                if hasattr(self.processor, "feature_extractor"):
                    self.sample_rate = int(getattr(self.processor.feature_extractor, "sampling_rate", 16000) or 16000)
        except Exception as exc:
            self.fallback_reason = f"failed to load processor: {exc}"
            logger.error(self.fallback_reason)
            return False

        try:
            # DirectML does not support bfloat16; forcing float32 avoids native crashes.
            if self.effective_device in ("cpu", "directml"):
                dtype = torch.float32
            else:
                dtype = "auto"
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.model = AutoModelForCTC.from_pretrained(
                    load_source,
                    local_files_only=local_only,
                    dtype=dtype,
                )
            self.model.to(self._torch_device)
            self.model.eval()
            first_param = next(self.model.parameters(), None)
            self._dtype = first_param.dtype if first_param is not None else torch.float32
        except Exception as exc:
            self.fallback_reason = f"failed to load model weights: {exc}"
            logger.error(self.fallback_reason)
            return False

        self.is_loaded = True
        return True

    def transcribe_audio(
        self,
        audio_data: np.ndarray,
        language: Optional[str] = None,
        task: str = "transcribe",
    ) -> TranscriptionResult:
        if task == "translate":
            return TranscriptionResult(success=False, error="Parakeet CTC does not support translate task")

        if not self.is_loaded and not self.load_model():
            return TranscriptionResult(success=False, error=self.fallback_reason or "failed to load Parakeet backend")

        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        if len(audio_data.shape) > 1:
            audio_data = audio_data.flatten()

        try:
            import torch
        except Exception as exc:
            return TranscriptionResult(success=False, error=f"torch unavailable during inference: {exc}")

        try:
            inputs = self.processor(  # type: ignore[operator]
                audio_data,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
            )
            # DirectML path in current torch-directml builds is unstable with parakeet attention masks.
            # Omitting the mask avoids bool->conv type issues while preserving correct decoding behavior.
            if self.effective_device == "directml":
                inputs.pop("attention_mask", None)
            inputs = {key: value.to(self._torch_device) for key, value in inputs.items()}

            with torch.no_grad():
                # CTC decoding path; avoids generate() behaviors that are unstable on DirectML.
                outputs = self.model(**inputs)  # type: ignore[operator]
                logits = outputs.logits
                predicted_ids = torch.argmax(logits, dim=-1)
                decoded = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)  # type: ignore[operator]
                text = decoded[0] if decoded else ""

            text = str(text).strip()
            duration = float(len(audio_data) / max(self.sample_rate, 1))
            segments = [{"start": 0.0, "end": duration, "text": text}] if text else []

            return TranscriptionResult(
                success=True,
                text=text,
                segments=segments,
                language=language or "en",
                language_probability=1.0 if text else 0.0,
                duration=duration,
            )
        except Exception as exc:
            return TranscriptionResult(success=False, error=f"Parakeet inference failed: {exc}")

    def backend_info(self) -> BackendInfo:
        return BackendInfo(
            runtime="parakeet_transformers",
            provider=self.provider,
            requested_device=self.requested_device,
            effective_device=self.effective_device,
            fallback_reason=self.fallback_reason,
            model_id=self.model_id,
            track_id=self.track_id,
        )
