#!/usr/bin/env python3
"""
Probe local Parakeet ONNX artifact readiness for offline transcription integration.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CANDIDATE_MODELS = [
    "parakeet-ctc-0.6b",
    "parakeet-ctc-1.1b",
]

MODEL_FILES = [
    "model.onnx",
    "encoder.onnx",
]

TOKENIZER_FILES = [
    "tokens.txt",
    "vocab.txt",
    "tokenizer.json",
    "spm.model",
]

CONFIG_FILES = [
    "config.json",
    "preprocessor_config.json",
]


@dataclass
class ArtifactStatus:
    model_id: str
    artifact_dir: str
    exists: bool
    model_files: list[str] = field(default_factory=list)
    tokenizer_files: list[str] = field(default_factory=list)
    config_files: list[str] = field(default_factory=list)
    load_test_ok: bool = False
    load_test_error: str = ""
    input_signatures: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ProbeResult:
    timestamp_utc: str
    python: str
    platform: str
    machine: str
    release: str
    artifact_root: str
    fixtures_dir: str
    fixture_files: list[str] = field(default_factory=list)
    onnxruntime_installed: bool = False
    onnxruntime_version: str = ""
    onnx_asr_installed: bool = False
    onnx_asr_version: str = ""
    artifacts: list[ArtifactStatus] = field(default_factory=list)
    has_any_artifact: bool = False
    has_any_loadable_model: bool = False


def _find_files(directory: Path, names: list[str]) -> list[str]:
    found: list[str] = []
    for name in names:
        path = directory / name
        if path.exists():
            found.append(path.name)
    return found


def _load_model_if_possible(model_dir: Path) -> tuple[bool, str, list[dict[str, Any]]]:
    candidates = [model_dir / "model.onnx", model_dir / "encoder.onnx"]
    onnx_files = [path for path in candidates if path.exists()]
    if not onnx_files:
        return False, "no ONNX model file found", []

    try:
        import onnxruntime as ort
    except Exception as exc:
        return False, f"onnxruntime unavailable: {exc}", []

    model_path = onnx_files[0]
    try:
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        signatures = []
        for inp in session.get_inputs():
            signatures.append(
                {
                    "name": inp.name,
                    "type": inp.type,
                    "shape": inp.shape,
                }
            )
        return True, "", signatures
    except Exception as exc:
        return False, str(exc), []


def run_probe(artifact_root: Path, fixtures_dir: Path) -> ProbeResult:
    result = ProbeResult(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        python=sys.version.replace("\n", " "),
        platform=platform.system(),
        machine=platform.machine(),
        release=platform.release(),
        artifact_root=str(artifact_root),
        fixtures_dir=str(fixtures_dir),
    )

    if fixtures_dir.exists():
        result.fixture_files = sorted([path.name for path in fixtures_dir.glob("*.wav")])

    try:
        import onnxruntime as ort

        result.onnxruntime_installed = True
        result.onnxruntime_version = ort.__version__
    except Exception:
        pass

    try:
        import onnx_asr  # type: ignore

        result.onnx_asr_installed = True
        result.onnx_asr_version = getattr(onnx_asr, "__version__", "unknown")
    except Exception:
        pass

    for model_id in CANDIDATE_MODELS:
        model_dir = artifact_root / model_id
        status = ArtifactStatus(
            model_id=model_id,
            artifact_dir=str(model_dir),
            exists=model_dir.exists(),
        )
        if model_dir.exists():
            status.model_files = _find_files(model_dir, MODEL_FILES)
            status.tokenizer_files = _find_files(model_dir, TOKENIZER_FILES)
            status.config_files = _find_files(model_dir, CONFIG_FILES)

            ok, error, signatures = _load_model_if_possible(model_dir)
            status.load_test_ok = ok
            status.load_test_error = error
            status.input_signatures = signatures

        result.artifacts.append(status)

    result.has_any_artifact = any(item.exists for item in result.artifacts)
    result.has_any_loadable_model = any(item.load_test_ok for item in result.artifacts)
    return result


def _write_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe local Parakeet ONNX artifact readiness.")
    parser.add_argument(
        "--artifact-root",
        default=str(Path(__file__).resolve().parent / "artifacts"),
        help="Root folder containing model subdirectories.",
    )
    parser.add_argument(
        "--fixtures-dir",
        default=str(Path(__file__).resolve().parent / "fixtures"),
        help="Directory containing WAV fixtures.",
    )
    parser.add_argument("--output", default="", help="Optional output JSON path.")
    args = parser.parse_args()

    artifact_root = Path(args.artifact_root).resolve()
    fixtures_dir = Path(args.fixtures_dir).resolve()

    result = run_probe(artifact_root=artifact_root, fixtures_dir=fixtures_dir)
    payload = asdict(result)
    payload["artifacts"] = [asdict(item) for item in result.artifacts]

    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output:
        _write_output(Path(args.output).resolve(), payload)
        print(f"\nSaved probe result to: {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
