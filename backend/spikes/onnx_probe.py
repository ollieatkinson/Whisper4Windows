#!/usr/bin/env python3
"""
Probe ONNX Runtime provider availability and run a tiny identity inference test.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ProviderTest:
    requested_provider: str
    available: bool
    ok: bool = False
    selected_provider: str = ""
    error: str = ""


@dataclass
class ProbeResult:
    timestamp_utc: str
    python: str
    platform: str
    machine: str
    release: str
    onnxruntime_installed: bool = False
    onnxruntime_version: str = ""
    onnx_installed: bool = False
    onnx_version: str = ""
    numpy_installed: bool = False
    numpy_version: str = ""
    providers: list[str] = field(default_factory=list)
    provider_tests: list[ProviderTest] = field(default_factory=list)
    inference_probe_ok: bool = False
    inference_probe_error: str = ""
    directml_provider_detected: bool = False
    cuda_provider_detected: bool = False


def _write_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _build_identity_onnx(model_path: Path) -> None:
    import onnx
    from onnx import TensorProto, checker, helper

    node = helper.make_node("Identity", inputs=["input"], outputs=["output"])
    graph = helper.make_graph(
        nodes=[node],
        name="identity_graph",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
    )
    model = helper.make_model(graph, producer_name="onnx_probe")
    checker.check_model(model)
    onnx.save(model, str(model_path))


def run_probe() -> ProbeResult:
    result = ProbeResult(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        python=sys.version.replace("\n", " "),
        platform=platform.system(),
        machine=platform.machine(),
        release=platform.release(),
    )

    try:
        import onnxruntime as ort

        result.onnxruntime_installed = True
        result.onnxruntime_version = ort.__version__
        result.providers = ort.get_available_providers()
    except Exception as exc:
        result.inference_probe_error = f"onnxruntime import failed: {exc}"
        return result

    try:
        import onnx

        result.onnx_installed = True
        result.onnx_version = onnx.__version__
    except Exception:
        pass

    try:
        import numpy as np

        result.numpy_installed = True
        result.numpy_version = np.__version__
    except Exception:
        pass

    result.directml_provider_detected = any(
        provider in result.providers for provider in ("DmlExecutionProvider", "DirectMLExecutionProvider")
    )
    result.cuda_provider_detected = "CUDAExecutionProvider" in result.providers

    if not (result.onnx_installed and result.numpy_installed):
        missing = []
        if not result.onnx_installed:
            missing.append("onnx")
        if not result.numpy_installed:
            missing.append("numpy")
        result.inference_probe_error = f"identity inference skipped, missing dependencies: {', '.join(missing)}"
        return result

    import numpy as np
    import onnxruntime as ort

    with tempfile.TemporaryDirectory(prefix="onnx_probe_") as tmp_dir:
        model_path = Path(tmp_dir) / "identity.onnx"
        _build_identity_onnx(model_path)

        candidates = [
            "CUDAExecutionProvider",
            "DmlExecutionProvider",
            "DirectMLExecutionProvider",
            "CPUExecutionProvider",
        ]

        for candidate in candidates:
            available = candidate in result.providers
            test = ProviderTest(requested_provider=candidate, available=available)
            if not available:
                result.provider_tests.append(test)
                continue

            providers_to_try = [candidate]
            if candidate != "CPUExecutionProvider" and "CPUExecutionProvider" in result.providers:
                providers_to_try.append("CPUExecutionProvider")

            try:
                session = ort.InferenceSession(str(model_path), providers=providers_to_try)
                x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
                y = session.run(None, {"input": x})[0]
                test.ok = bool(np.allclose(x, y))
                active = session.get_providers()
                test.selected_provider = active[0] if active else ""
            except Exception as exc:
                test.error = str(exc)

            result.provider_tests.append(test)

    result.inference_probe_ok = any(item.ok for item in result.provider_tests)
    if not result.inference_probe_ok and not result.inference_probe_error:
        result.inference_probe_error = "no provider passed the identity inference test"

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe ONNX Runtime providers and tiny inference.")
    parser.add_argument(
        "--output",
        default="",
        help="Optional output JSON path.",
    )
    args = parser.parse_args()

    result = run_probe()
    payload = asdict(result)
    payload["provider_tests"] = [asdict(item) for item in result.provider_tests]

    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output:
        _write_output(Path(args.output).resolve(), payload)
        print(f"\nSaved probe result to: {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
