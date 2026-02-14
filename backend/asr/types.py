"""
Shared ASR backend types.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class BackendInfo:
    runtime: str
    provider: str
    requested_device: str
    effective_device: str
    fallback_reason: str = ""
    model_id: str = ""
    track_id: str = "ralph-wiggum"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TranscriptionResult:
    success: bool
    text: str = ""
    segments: list[dict[str, Any]] = field(default_factory=list)
    language: str = ""
    language_probability: float = 0.0
    duration: float = 0.0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if not self.error:
            payload.pop("error", None)
        return payload
