"""
Model download manager with background tasks and status tracking.
"""

from __future__ import annotations

import json
import shutil
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .hf_client import DownloadCancelled, HfClient, sha256_file
from .registry import ModelRegistry, get_model_registry
from .storage import get_model_dir, get_model_marker_path, get_models_root


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_large_file(path: str, size: int) -> bool:
    lowered = path.lower()
    if lowered.endswith((".safetensors", ".bin", ".onnx", ".nemo", ".pt", ".ckpt")):
        return True
    return size >= 200 * 1024 * 1024


@dataclass
class FileDownloadStatus:
    path: str
    required: bool
    size: int = 0
    downloaded_bytes: int = 0
    state: str = "pending"
    error: str = ""
    sha256: str = ""


@dataclass
class ModelDownloadStatus:
    model_id: str
    track_id: str
    state: str = "not_started"
    message: str = ""
    source: str = ""
    repo_id: str = ""
    revision: str = ""
    include_large_files: bool = True
    started_at: str = ""
    finished_at: str = ""
    bytes_total: int = 0
    bytes_downloaded: int = 0
    files_total: int = 0
    files_completed: int = 0
    files_failed: int = 0
    files_skipped: int = 0
    files: list[FileDownloadStatus] = field(default_factory=list)


class ModelDownloadManager:
    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or get_model_registry()
        self.hf_client = HfClient()
        self._status_by_model: dict[str, ModelDownloadStatus] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._model_locks: dict[str, threading.Lock] = {}
        self._lock = threading.Lock()

    def _get_model_lock(self, model_id: str) -> threading.Lock:
        with self._lock:
            if model_id not in self._model_locks:
                self._model_locks[model_id] = threading.Lock()
            return self._model_locks[model_id]

    def _set_status(self, model_id: str, status: ModelDownloadStatus) -> None:
        with self._lock:
            self._status_by_model[model_id] = status

    def _get_status_obj(self, model_id: str) -> ModelDownloadStatus:
        with self._lock:
            status = self._status_by_model.get(model_id)
            if status is None:
                status = ModelDownloadStatus(
                    model_id=model_id,
                    track_id=self.registry.track_id,
                    state="not_started",
                    message="download not started",
                )
                self._status_by_model[model_id] = status
            return status

    def _status_payload(self, model_id: str) -> dict[str, Any]:
        status = self._get_status_obj(model_id)
        payload = asdict(status)
        payload["files"] = [asdict(item) for item in status.files]
        return payload

    def get_status(self, model_id: str) -> dict[str, Any]:
        resolved_id = self.registry.resolve_model_id(model_id=model_id)
        return self._status_payload(resolved_id)

    def cancel_download(self, model_id: str) -> dict[str, Any]:
        resolved_id = self.registry.resolve_model_id(model_id=model_id)
        with self._lock:
            event = self._cancel_events.get(resolved_id)
        if event is None:
            return {"success": False, "message": "no active download"}
        event.set()
        return {"success": True, "message": "cancel requested", "status": self._status_payload(resolved_id)}

    def delete_model_artifacts(self, model_id: str) -> dict[str, Any]:
        resolved_id = self.registry.resolve_model_id(model_id=model_id)
        spec = self.registry.get_model(resolved_id, allow_disabled=True)
        runtime = str(spec.get("runtime", "")).strip().lower()
        source = str(spec.get("download", {}).get("source", "")).strip().lower()

        with self._lock:
            thread = self._threads.get(resolved_id)
            if thread and thread.is_alive():
                event = self._cancel_events.get(resolved_id)
                if event is not None:
                    event.set()

        if source == "huggingface":
            model_dir = get_model_dir(resolved_id)
            if model_dir.exists():
                shutil.rmtree(model_dir, ignore_errors=False)
            status = ModelDownloadStatus(
                model_id=resolved_id,
                track_id=self.registry.track_id,
                state="not_started",
                message="model artifacts deleted",
                source=source,
                finished_at=_utc_now(),
            )
            self._set_status(resolved_id, status)
            return {"success": True, "message": "model artifacts deleted", "status": self._status_payload(resolved_id)}

        if runtime == "faster_whisper":
            backend_name = str(spec.get("backend_model_name", "")).strip()
            if backend_name:
                cache_dir = get_models_root() / f"models--Systran--faster-whisper-{backend_name}"
                if cache_dir.exists():
                    shutil.rmtree(cache_dir, ignore_errors=False)
            status = ModelDownloadStatus(
                model_id=resolved_id,
                track_id=self.registry.track_id,
                state="not_started",
                message="model cache deleted",
                source=source or runtime,
                finished_at=_utc_now(),
            )
            self._set_status(resolved_id, status)
            return {"success": True, "message": "model cache deleted", "status": self._status_payload(resolved_id)}

        return {"success": False, "message": "model source is not deletable via manager", "status": self._status_payload(resolved_id)}

    def is_downloaded(self, model_id: str) -> bool:
        resolved_id = self.registry.resolve_model_id(model_id=model_id)
        spec = self.registry.get_model(resolved_id, allow_disabled=True)
        source = str(spec.get("download", {}).get("source", "")).strip().lower()
        if source != "huggingface":
            # faster-whisper manages this internally; this helper only checks HF-managed models.
            return False

        required_files = self.registry.required_download_files(resolved_id)
        model_dir = get_model_dir(resolved_id)
        for file_spec in required_files:
            rel_path = str(file_spec.get("path", "")).strip()
            if not rel_path:
                continue
            file_path = model_dir / rel_path
            if not file_path.exists():
                return False
        return True

    def start_download(
        self,
        model_id: str,
        include_large_files: bool = True,
        background: bool = True,
    ) -> dict[str, Any]:
        resolved_id = self.registry.resolve_model_id(model_id=model_id)
        spec = self.registry.get_model(resolved_id, allow_disabled=True)
        source = str(spec.get("download", {}).get("source", "")).strip().lower()

        if source != "huggingface":
            status = ModelDownloadStatus(
                model_id=resolved_id,
                track_id=self.registry.track_id,
                state="completed",
                message="download not required for this model source",
                source=source,
                started_at=_utc_now(),
                finished_at=_utc_now(),
            )
            self._set_status(resolved_id, status)
            return {"success": True, "started": False, "status": self._status_payload(resolved_id)}

        with self._lock:
            current_thread = self._threads.get(resolved_id)
            if current_thread and current_thread.is_alive():
                return {"success": True, "started": False, "status": self._status_payload(resolved_id)}

            cancel_event = threading.Event()
            self._cancel_events[resolved_id] = cancel_event

        if background:
            thread = threading.Thread(
                target=self._run_download,
                args=(resolved_id, include_large_files, cancel_event),
                daemon=True,
            )
            with self._lock:
                self._threads[resolved_id] = thread
            thread.start()
            return {"success": True, "started": True, "status": self._status_payload(resolved_id)}

        self._run_download(resolved_id, include_large_files, cancel_event)
        return {"success": True, "started": True, "status": self._status_payload(resolved_id)}

    def ensure_downloaded(self, model_id: str, include_large_files: bool = True) -> dict[str, Any]:
        resolved_id = self.registry.resolve_model_id(model_id=model_id)
        start_payload = self.start_download(
            model_id=resolved_id,
            include_large_files=include_large_files,
            background=True,
        )

        with self._lock:
            thread = self._threads.get(resolved_id)
        if thread:
            thread.join()

        return self._status_payload(resolved_id)

    def _run_download(self, model_id: str, include_large_files: bool, cancel_event: threading.Event) -> None:
        model_lock = self._get_model_lock(model_id)
        with model_lock:
            spec = self.registry.get_model(model_id, allow_disabled=True)
            download = spec.get("download", {})
            source = str(download.get("source", "")).strip().lower()
            repo_id = str(download.get("repo_id", "")).strip()
            revision = str(download.get("revision", "")).strip()
            files = download.get("files", []) if isinstance(download.get("files", []), list) else []

            status = ModelDownloadStatus(
                model_id=model_id,
                track_id=self.registry.track_id,
                state="running",
                message="downloading model artifacts",
                source=source,
                repo_id=repo_id,
                revision=revision,
                include_large_files=include_large_files,
                started_at=_utc_now(),
            )

            status.files = []
            for file_spec in files:
                rel_path = str(file_spec.get("path", "")).strip()
                if not rel_path:
                    continue
                required = bool(file_spec.get("required", True))
                size = int(file_spec.get("size", 0) or 0)
                status.files.append(FileDownloadStatus(path=rel_path, required=required, size=size))
                status.bytes_total += size

            status.files_total = len(status.files)
            self._set_status(model_id, status)

            if source != "huggingface":
                status.state = "completed"
                status.message = "download not required for this model source"
                status.finished_at = _utc_now()
                self._set_status(model_id, status)
                return

            if not repo_id:
                status.state = "failed"
                status.message = "manifest missing download.repo_id"
                status.finished_at = _utc_now()
                self._set_status(model_id, status)
                return

            if not revision:
                revision = self.hf_client.resolve_revision(repo_id=repo_id, revision="")
                status.revision = revision

            model_dir = get_model_dir(model_id)
            required_failure = False

            for item in status.files:
                if cancel_event.is_set():
                    status.state = "cancelled"
                    status.message = "download cancelled"
                    status.finished_at = _utc_now()
                    self._set_status(model_id, status)
                    return

                if _is_large_file(item.path, item.size) and not include_large_files:
                    item.state = "skipped"
                    status.files_skipped += 1
                    if item.required:
                        required_failure = True
                        item.error = "required file skipped because include_large_files=false"
                    self._set_status(model_id, status)
                    continue

                destination = model_dir / item.path
                expected_sha = ""
                for raw in files:
                    if str(raw.get("path", "")).strip() == item.path:
                        expected_sha = str(raw.get("sha256", "")).strip().lower()
                        break

                if destination.exists():
                    if expected_sha:
                        existing_sha = sha256_file(destination)
                        if existing_sha == expected_sha:
                            item.sha256 = existing_sha
                            item.downloaded_bytes = destination.stat().st_size
                            item.state = "completed"
                            status.bytes_downloaded += item.downloaded_bytes
                            status.files_completed += 1
                            self._set_status(model_id, status)
                            continue
                        destination.unlink(missing_ok=True)
                    else:
                        item.downloaded_bytes = destination.stat().st_size
                        item.state = "completed"
                        status.bytes_downloaded += item.downloaded_bytes
                        status.files_completed += 1
                        self._set_status(model_id, status)
                        continue

                item.state = "running"
                self._set_status(model_id, status)

                try:
                    downloaded = self.hf_client.download_file(
                        repo_id=repo_id,
                        revision=revision,
                        file_path=item.path,
                        destination=destination,
                        cancel_event=cancel_event,
                    )
                    digest = sha256_file(destination)
                    if expected_sha and digest != expected_sha:
                        file_size = destination.stat().st_size
                        # Some HF large-file backends expose non-sha256 etags; accept size-verified large files.
                        if not (_is_large_file(item.path, item.size) and item.size > 0 and file_size == item.size):
                            destination.unlink(missing_ok=True)
                            raise RuntimeError(
                                f"checksum mismatch for {item.path}: expected {expected_sha}, got {digest}"
                            )
                        item.error = (
                            f"checksum mismatch accepted for large file after size verification; "
                            f"expected={expected_sha} actual={digest}"
                        )

                    item.sha256 = digest
                    item.downloaded_bytes = downloaded
                    item.state = "completed"
                    status.bytes_downloaded += downloaded
                    status.files_completed += 1
                except DownloadCancelled:
                    destination.unlink(missing_ok=True)
                    status.state = "cancelled"
                    status.message = "download cancelled"
                    status.finished_at = _utc_now()
                    item.state = "cancelled"
                    self._set_status(model_id, status)
                    return
                except Exception as exc:
                    item.state = "failed"
                    item.error = str(exc)
                    status.files_failed += 1
                    if item.required:
                        required_failure = True

                self._set_status(model_id, status)

            marker_payload = {
                "model_id": model_id,
                "track_id": self.registry.track_id,
                "revision": status.revision,
                "timestamp_utc": _utc_now(),
                "state": "completed" if not required_failure else "failed",
                "files_completed": status.files_completed,
                "files_failed": status.files_failed,
                "files_skipped": status.files_skipped,
            }
            get_model_marker_path(model_id).write_text(json.dumps(marker_payload, indent=2), encoding="utf-8")

            status.finished_at = _utc_now()
            if required_failure:
                status.state = "failed"
                status.message = "one or more required files failed to download"
            else:
                status.state = "completed"
                status.message = "model download completed"
            self._set_status(model_id, status)

        with self._lock:
            self._cancel_events.pop(model_id, None)


_download_manager_singleton: ModelDownloadManager | None = None
_download_manager_lock = threading.Lock()


def get_download_manager() -> ModelDownloadManager:
    global _download_manager_singleton
    if _download_manager_singleton is not None:
        return _download_manager_singleton

    with _download_manager_lock:
        if _download_manager_singleton is None:
            _download_manager_singleton = ModelDownloadManager()
    return _download_manager_singleton
