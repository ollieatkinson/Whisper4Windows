"""
ASR backend selection.
"""

from __future__ import annotations

from typing import Any

from models.storage import get_model_dir

from .faster_whisper_backend import FasterWhisperBackend
from .onnx_backend import OnnxBackend
from .parakeet_transformers_backend import ParakeetTransformersBackend


def create_backend(
    model_spec: dict[str, Any],
    requested_device: str,
    compute_type: str = "auto",
    track_id: str = "ralph-wiggum",
):
    runtime = str(model_spec.get("runtime", "")).strip().lower()
    model_id = str(model_spec.get("model_id", "")).strip()
    backend_model_name = str(model_spec.get("backend_model_name", model_id)).strip()

    if runtime == "faster_whisper":
        return FasterWhisperBackend(
            model_id=model_id,
            model_name=backend_model_name,
            requested_device=requested_device,
            compute_type=compute_type,
            track_id=track_id,
        )

    if runtime == "onnx":
        return OnnxBackend(
            model_id=model_id,
            model_dir=get_model_dir(model_id),
            requested_device=requested_device,
            track_id=track_id,
        )

    if runtime == "parakeet_transformers":
        return ParakeetTransformersBackend(
            model_id=model_id,
            model_dir=get_model_dir(model_id),
            model_spec=model_spec,
            requested_device=requested_device,
            track_id=track_id,
        )

    raise ValueError(f"unsupported runtime '{runtime}' for model_id '{model_id}'")
