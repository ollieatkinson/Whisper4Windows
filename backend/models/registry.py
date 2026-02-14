"""
Capability-aware model registry backed by JSON manifest.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from typing import Any


def _as_bool(raw: str, default: bool) -> bool:
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    return default


class ModelRegistry:
    def __init__(self, manifest_path: Path | None = None) -> None:
        self.manifest_path = manifest_path or (Path(__file__).resolve().parent / "manifest.json")
        self._manifest: dict[str, Any] = {}
        self._models_by_id: dict[str, dict[str, Any]] = {}
        self._aliases_to_id: dict[str, str] = {}
        self._load_manifest()

    def _load_manifest(self) -> None:
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        models = payload.get("models", [])
        if not isinstance(models, list):
            raise ValueError("manifest.models must be a list")

        models_by_id: dict[str, dict[str, Any]] = {}
        aliases_to_id: dict[str, str] = {}

        for item in models:
            model_id = str(item.get("model_id", "")).strip()
            if not model_id:
                raise ValueError("manifest model entry missing model_id")
            if model_id in models_by_id:
                raise ValueError(f"duplicate model_id in manifest: {model_id}")
            models_by_id[model_id] = item

            aliases = item.get("aliases", [])
            if not isinstance(aliases, list):
                raise ValueError(f"aliases must be a list for model {model_id}")
            for alias in aliases:
                alias_key = str(alias).strip().lower()
                if alias_key:
                    aliases_to_id[alias_key] = model_id

            aliases_to_id[model_id.lower()] = model_id

        self._manifest = payload
        self._models_by_id = models_by_id
        self._aliases_to_id = aliases_to_id

    @property
    def version(self) -> int:
        return int(self._manifest.get("version", 0))

    @property
    def track_id(self) -> str:
        # User-requested rollout identifier carried through all phases.
        return str(self._manifest.get("track_id", "ralph-wiggum"))

    @property
    def default_model_id(self) -> str:
        configured = str(self._manifest.get("default_model_id", "")).strip()
        if configured and configured in self._models_by_id and self.is_model_enabled(configured):
            return configured

        for model_id, spec in self._models_by_id.items():
            if spec.get("family") == "whisper" and self.is_model_enabled(model_id):
                return model_id

        if self._models_by_id:
            return next(iter(self._models_by_id.keys()))
        raise RuntimeError("model registry is empty")

    def list_models(self, include_disabled: bool = False) -> list[dict[str, Any]]:
        models = list(self._models_by_id.values())
        if include_disabled:
            return models
        return [item for item in models if self.is_model_enabled(item["model_id"])]

    def is_model_enabled(self, model_id: str) -> bool:
        spec = self._models_by_id.get(model_id)
        if not spec:
            return False

        runtime = str(spec.get("runtime", "")).strip().lower()
        family = str(spec.get("family", "")).strip().lower()

        if runtime == "onnx" and not _as_bool(os.getenv("ENABLE_ONNX_BACKEND", "1"), True):
            return False
        if family == "parakeet" and not _as_bool(os.getenv("ENABLE_PARAKEET_MODELS", "1"), True):
            return False
        return True

    def get_model(self, model_id: str, allow_disabled: bool = False) -> dict[str, Any]:
        spec = self._models_by_id.get(model_id)
        if not spec:
            raise ValueError(f"unknown model_id: {model_id}")
        if not allow_disabled and not self.is_model_enabled(model_id):
            raise ValueError(f"model_id is disabled by feature flags: {model_id}")
        return spec

    def resolve_model_id(self, model_id: str | None = None, model_size: str | None = None) -> str:
        if model_id:
            key = model_id.strip().lower()
            resolved = self._aliases_to_id.get(key)
            if not resolved:
                raise ValueError(f"unknown model_id: {model_id}")
            if not self.is_model_enabled(resolved):
                raise ValueError(f"model_id is disabled by feature flags: {resolved}")
            return resolved

        if model_size:
            key = model_size.strip().lower()
            resolved = self._aliases_to_id.get(key)
            if resolved and self.is_model_enabled(resolved):
                return resolved

        return self.default_model_id

    def resolve_model_spec(self, model_id: str | None = None, model_size: str | None = None) -> dict[str, Any]:
        resolved_id = self.resolve_model_id(model_id=model_id, model_size=model_size)
        spec = dict(self.get_model(resolved_id))
        spec["model_id"] = resolved_id
        spec["legacy_model_size"] = self.get_preferred_legacy_size(resolved_id)
        return spec

    def get_preferred_legacy_size(self, model_id: str) -> str:
        spec = self.get_model(model_id, allow_disabled=True)
        aliases = spec.get("aliases", [])
        if isinstance(aliases, list) and aliases:
            return str(aliases[0])
        return str(spec.get("backend_model_name", model_id))

    def required_download_files(self, model_id: str) -> list[dict[str, Any]]:
        spec = self.get_model(model_id, allow_disabled=True)
        download = spec.get("download", {})
        files = download.get("files", [])
        if not isinstance(files, list):
            return []

        required: list[dict[str, Any]] = []
        for item in files:
            if not isinstance(item, dict):
                continue
            if bool(item.get("required", True)):
                required.append(item)
        return required


_registry_singleton: ModelRegistry | None = None
_registry_lock = Lock()


def get_model_registry() -> ModelRegistry:
    global _registry_singleton
    if _registry_singleton is not None:
        return _registry_singleton

    with _registry_lock:
        if _registry_singleton is None:
            _registry_singleton = ModelRegistry()
    return _registry_singleton
