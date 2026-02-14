"""
Minimal Hugging Face HTTP client for deterministic artifact downloads.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from threading import Event
from urllib import parse, request


HF_BASE_URL = "https://huggingface.co"
HF_API_BASE_URL = "https://huggingface.co/api"


class DownloadCancelled(Exception):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class HfClient:
    def __init__(self, timeout: int = 120) -> None:
        self.timeout = timeout
        self.user_agent = "Whisper4Windows-hf-client/1.0"

    def _json_get(self, url: str) -> dict | list:
        req = request.Request(url, headers={"User-Agent": self.user_agent})
        with request.urlopen(req, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def model_info(self, repo_id: str) -> dict:
        encoded_repo = parse.quote(repo_id, safe="/")
        payload = self._json_get(f"{HF_API_BASE_URL}/models/{encoded_repo}")
        if not isinstance(payload, dict):
            raise RuntimeError("unexpected model_info response payload type")
        return payload

    def resolve_revision(self, repo_id: str, revision: str | None = None) -> str:
        if revision:
            return revision
        info = self.model_info(repo_id)
        resolved = str(info.get("sha", "")).strip()
        return resolved or "main"

    def resolve_url(self, repo_id: str, revision: str, file_path: str) -> str:
        encoded_parts = [parse.quote(part, safe="") for part in file_path.split("/")]
        encoded_path = "/".join(encoded_parts)
        return f"{HF_BASE_URL}/{repo_id}/resolve/{revision}/{encoded_path}"

    def download_file(
        self,
        repo_id: str,
        revision: str,
        file_path: str,
        destination: Path,
        cancel_event: Event | None = None,
    ) -> int:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_path = destination.parent / f"{destination.name}.partial"
        url = self.resolve_url(repo_id=repo_id, revision=revision, file_path=file_path)

        req = request.Request(url, headers={"User-Agent": self.user_agent})
        bytes_written = 0

        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                with temp_path.open("wb") as handle:
                    while True:
                        if cancel_event is not None and cancel_event.is_set():
                            raise DownloadCancelled(f"download cancelled: {repo_id}/{file_path}")
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        bytes_written += len(chunk)
            temp_path.replace(destination)
            return bytes_written
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
