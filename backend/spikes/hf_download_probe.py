#!/usr/bin/env python3
"""
Probe Hugging Face model download feasibility for ASR artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, parse, request


HF_BASE_URL = "https://huggingface.co"
HF_API_BASE_URL = "https://huggingface.co/api"

DEFAULT_REPOS = [
    "nvidia/parakeet-ctc-0.6b",
    "nvidia/parakeet-ctc-1.1b",
]

DEFAULT_REQUIRED_FILES = [
    "config.json",
    "preprocessor_config.json",
    "tokenizer.json",
]

LARGE_FILE_SUFFIXES = (".safetensors", ".bin", ".onnx", ".nemo", ".pt", ".ckpt")


@dataclass
class FileProbe:
    path: str
    size: int
    oid: str
    resolve_url: str
    head_ok: bool = False
    head_status: int = 0
    head_error: str = ""
    etag: str = ""
    content_length: int = 0
    download_attempted: bool = False
    download_ok: bool = False
    download_error: str = ""
    download_path: str = ""
    download_size: int = 0
    sha256: str = ""
    skipped_reason: str = ""


@dataclass
class RepoProbe:
    repo_id: str
    requested_revision: str
    resolved_revision: str = ""
    model_api_ok: bool = False
    model_api_error: str = ""
    tree_api_ok: bool = False
    tree_api_error: str = ""
    discovered_files: int = 0
    file_probes: list[FileProbe] = field(default_factory=list)


@dataclass
class ProbeResult:
    timestamp_utc: str
    python: str
    platform: str
    machine: str
    release: str
    output_root: str
    include_large_files: bool
    max_small_file_mb: int
    hf_api_reachable: bool = False
    hf_api_error: str = ""
    repos: list[RepoProbe] = field(default_factory=list)
    total_head_successes: int = 0
    total_download_successes: int = 0


def _json_get(url: str, timeout: int) -> Any:
    req = request.Request(url, headers={"User-Agent": "Whisper4Windows-hf-probe/1.0"})
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _head(url: str, timeout: int) -> tuple[bool, int, dict[str, str], str]:
    req = request.Request(url, method="HEAD", headers={"User-Agent": "Whisper4Windows-hf-probe/1.0"})
    try:
        with request.urlopen(req, timeout=timeout) as response:
            headers = {key.lower(): value for key, value in response.headers.items()}
            return True, response.getcode(), headers, ""
    except error.HTTPError as exc:
        return False, exc.code, {}, str(exc)
    except Exception as exc:
        return False, 0, {}, str(exc)


def _is_large_file(path: str) -> bool:
    lowered = path.lower()
    return lowered.endswith(LARGE_FILE_SUFFIXES)


def _should_download(path: str, size: int, include_large_files: bool, max_small_file_mb: int) -> tuple[bool, str]:
    if _is_large_file(path) and not include_large_files:
        return False, "large_file_skipped"

    max_bytes = max_small_file_mb * 1024 * 1024
    if size > max_bytes and not include_large_files:
        return False, f"size_over_limit_{max_small_file_mb}mb"

    return True, ""


def _compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_download(url: str, destination: Path, timeout: int) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.parent / f"{destination.name}.partial"

    try:
        req = request.Request(url, headers={"User-Agent": "Whisper4Windows-hf-probe/1.0"})
        with request.urlopen(req, timeout=timeout) as response:
            with temp_path.open("wb") as handle:
                bytes_written = 0
                while True:
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


def _to_model_api_url(repo_id: str) -> str:
    encoded_repo = parse.quote(repo_id, safe="/")
    return f"{HF_API_BASE_URL}/models/{encoded_repo}"


def _to_tree_api_url(repo_id: str, revision: str) -> str:
    encoded_repo = parse.quote(repo_id, safe="/")
    encoded_rev = parse.quote(revision, safe="")
    return f"{HF_API_BASE_URL}/models/{encoded_repo}/tree/{encoded_rev}?recursive=1&expand=1"


def _to_resolve_url(repo_id: str, revision: str, file_path: str) -> str:
    encoded_segments = [parse.quote(segment, safe="") for segment in file_path.split("/")]
    encoded_path = "/".join(encoded_segments)
    return f"{HF_BASE_URL}/{repo_id}/resolve/{revision}/{encoded_path}"


def _select_probe_files(tree_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_path = {entry.get("path", ""): entry for entry in tree_entries if entry.get("type") == "file"}
    selected: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    for required in DEFAULT_REQUIRED_FILES:
        entry = by_path.get(required)
        if entry:
            selected.append(entry)
            seen_paths.add(required)

    # Probe at least one large artifact for each repo without forcing a full download.
    for entry in tree_entries:
        path = entry.get("path", "")
        if path in seen_paths or entry.get("type") != "file":
            continue
        if _is_large_file(path):
            selected.append(entry)
            seen_paths.add(path)
            break

    return selected


def run_probe(
    repos: list[str],
    output_root: Path,
    requested_revision: str,
    include_large_files: bool,
    max_small_file_mb: int,
    timeout: int,
) -> ProbeResult:
    result = ProbeResult(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        python=sys.version.replace("\n", " "),
        platform=platform.system(),
        machine=platform.machine(),
        release=platform.release(),
        output_root=str(output_root),
        include_large_files=include_large_files,
        max_small_file_mb=max_small_file_mb,
    )

    try:
        _json_get(f"{HF_API_BASE_URL}/models?search=parakeet&limit=1", timeout=timeout)
        result.hf_api_reachable = True
    except Exception as exc:
        result.hf_api_error = str(exc)
        return result

    for repo_id in repos:
        repo = RepoProbe(repo_id=repo_id, requested_revision=requested_revision or "auto(head-sha)")
        result.repos.append(repo)

        try:
            model_info = _json_get(_to_model_api_url(repo_id), timeout=timeout)
            repo.model_api_ok = True
        except Exception as exc:
            repo.model_api_error = str(exc)
            continue

        resolved_revision = requested_revision or str(model_info.get("sha", "")).strip() or "main"
        repo.resolved_revision = resolved_revision

        try:
            tree_entries = _json_get(_to_tree_api_url(repo_id, resolved_revision), timeout=timeout)
            if not isinstance(tree_entries, list):
                raise RuntimeError("tree API response was not a list")
            repo.tree_api_ok = True
            repo.discovered_files = len(tree_entries)
        except Exception as exc:
            repo.tree_api_error = str(exc)
            continue

        selected_entries = _select_probe_files(tree_entries)
        for entry in selected_entries:
            path = str(entry.get("path", ""))
            size = int(entry.get("size", 0))
            oid = str(entry.get("oid", ""))
            resolve_url = _to_resolve_url(repo_id, resolved_revision, path)

            file_probe = FileProbe(path=path, size=size, oid=oid, resolve_url=resolve_url)

            head_ok, status, headers, head_error = _head(resolve_url, timeout=timeout)
            file_probe.head_ok = head_ok
            file_probe.head_status = status
            file_probe.head_error = head_error
            file_probe.etag = headers.get("etag", "")

            content_length_raw = headers.get("content-length", "")
            if content_length_raw.isdigit():
                file_probe.content_length = int(content_length_raw)

            if file_probe.head_ok:
                result.total_head_successes += 1

            should_download, skipped_reason = _should_download(
                path=path,
                size=size,
                include_large_files=include_large_files,
                max_small_file_mb=max_small_file_mb,
            )

            if not should_download:
                file_probe.skipped_reason = skipped_reason
                repo.file_probes.append(file_probe)
                continue

            destination = output_root / repo_id.replace("/", "--") / resolved_revision / path
            file_probe.download_attempted = True
            file_probe.download_path = str(destination)

            try:
                bytes_written = _atomic_download(resolve_url, destination, timeout=timeout)
                file_probe.download_ok = True
                file_probe.download_size = bytes_written
                file_probe.sha256 = _compute_sha256(destination)
                result.total_download_successes += 1
            except Exception as exc:
                file_probe.download_error = str(exc)

            repo.file_probes.append(file_probe)

    return result


def _write_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe Hugging Face download feasibility for ASR artifacts.")
    parser.add_argument(
        "--repo",
        action="append",
        default=[],
        help="Model repo id to probe (repeatable). Defaults to two Parakeet repos.",
    )
    parser.add_argument(
        "--revision",
        default="",
        help="Pinned revision to test. If omitted, uses model head SHA from HF API.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent / "artifacts" / "hf-cache"),
        help="Directory where downloaded files will be written.",
    )
    parser.add_argument(
        "--include-large-files",
        action="store_true",
        help="Allow downloading large files (e.g., .safetensors/.nemo).",
    )
    parser.add_argument(
        "--max-small-file-mb",
        type=int,
        default=20,
        help="Maximum size for auto-downloaded files when --include-large-files is not set.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output JSON path.",
    )
    args = parser.parse_args()

    repos = args.repo or list(DEFAULT_REPOS)
    output_root = Path(args.out_dir).resolve()

    result = run_probe(
        repos=repos,
        output_root=output_root,
        requested_revision=args.revision,
        include_large_files=args.include_large_files,
        max_small_file_mb=args.max_small_file_mb,
        timeout=args.timeout,
    )

    payload = asdict(result)
    payload["repos"] = []
    for repo in result.repos:
        repo_payload = asdict(repo)
        repo_payload["file_probes"] = [asdict(item) for item in repo.file_probes]
        payload["repos"].append(repo_payload)

    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output:
        _write_output(Path(args.output).resolve(), payload)
        print(f"\nSaved probe result to: {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
