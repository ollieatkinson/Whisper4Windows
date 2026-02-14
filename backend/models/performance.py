"""
Heuristic model performance estimation for current hardware.
"""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
from typing import Any

import gpu_manager
from asr.provider_resolver import DIRECTML_PROVIDER_NAMES


_KNOWN_MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "whisper-tiny": {"quality_score": 1, "relative_cost": 1.2, "min_ram_gb": 2.0, "min_vram_gb": 1.0},
    "whisper-base": {"quality_score": 2, "relative_cost": 1.8, "min_ram_gb": 3.0, "min_vram_gb": 2.0},
    "whisper-small": {"quality_score": 3, "relative_cost": 2.8, "min_ram_gb": 4.0, "min_vram_gb": 3.0},
    "whisper-medium": {"quality_score": 4, "relative_cost": 4.5, "min_ram_gb": 6.0, "min_vram_gb": 5.0},
    "whisper-large-v3": {"quality_score": 5, "relative_cost": 6.4, "min_ram_gb": 8.0, "min_vram_gb": 8.0},
    "whisper-large-v3-turbo": {"quality_score": 5, "relative_cost": 5.1, "min_ram_gb": 7.0, "min_vram_gb": 6.0},
    "parakeet-ctc-0.6b": {"quality_score": 4, "relative_cost": 5.8, "min_ram_gb": 10.0, "min_vram_gb": 8.0},
    "parakeet-ctc-1.1b": {"quality_score": 5, "relative_cost": 8.2, "min_ram_gb": 14.0, "min_vram_gb": 12.0},
}

_GPU_VRAM_NAME_HINTS_GB: list[tuple[str, float]] = [
    ("rtx 4090", 24.0),
    ("rtx 4080", 16.0),
    ("rtx 4070 ti", 12.0),
    ("rtx 4070", 12.0),
    ("rtx 3090", 24.0),
    ("rtx 3080", 10.0),
    ("7900 xtx", 24.0),
    ("7900 xt", 20.0),
    ("7800 xt", 16.0),
    ("6950 xt", 16.0),
    ("6900 xt", 16.0),
    ("6800 xt", 16.0),
]


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return default


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(raw)
    except Exception:
        return default


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def _detect_total_ram_gb() -> float:
    # Windows physical memory in KiB.
    mem_kb = ctypes.c_ulonglong(0)
    try:
        ok = ctypes.windll.kernel32.GetPhysicallyInstalledSystemMemory(ctypes.byref(mem_kb))
        if ok and mem_kb.value > 0:
            return float(mem_kb.value) / (1024.0 * 1024.0)
    except Exception:
        pass
    return 0.0


def _load_video_controller_details() -> list[dict[str, Any]]:
    try:
        ps_cmd = (
            "Get-CimInstance Win32_VideoController | "
            "Select-Object Name, AdapterRAM | ConvertTo-Json -Depth 3"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            capture_output=True,
            text=True,
            timeout=8,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
        )
        payload = (result.stdout or "").strip()
        if not payload:
            return []
        parsed = json.loads(payload)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            return [parsed]
    except Exception:
        pass
    return []


def _guess_gpu_vram_gb(vendor: str, gpu_names: list[str]) -> float:
    details = _load_video_controller_details()
    if not details:
        return 0.0

    normalized_vendor = (vendor or "unknown").lower()

    def _matches_vendor(name: str) -> bool:
        lowered = (name or "").lower()
        if normalized_vendor == "nvidia":
            return any(token in lowered for token in ("nvidia", "geforce", "quadro", "rtx"))
        if normalized_vendor == "amd":
            return any(token in lowered for token in ("amd", "radeon", "ati"))
        if normalized_vendor == "intel":
            return "intel" in lowered
        return True

    best_bytes = 0
    for item in details:
        name = str(item.get("Name") or item.get("name") or "")
        adapter_ram = _safe_int(item.get("AdapterRAM", 0))
        if adapter_ram <= 0:
            continue

        # Prefer entries that match detected vendor and known GPU names.
        in_detected_names = any(name.lower() in gpu_name.lower() or gpu_name.lower() in name.lower() for gpu_name in gpu_names)
        if _matches_vendor(name) or in_detected_names:
            best_bytes = max(best_bytes, adapter_ram)

    if best_bytes <= 0:
        for item in details:
            best_bytes = max(best_bytes, _safe_int(item.get("AdapterRAM", 0)))

    inferred_gb = float(best_bytes) / (1024.0 ** 3) if best_bytes > 0 else 0.0

    # Some Windows driver stacks report ~4GB for larger cards (especially non-CUDA stacks).
    if inferred_gb <= 5.0:
        joined_names = " ".join(name.lower() for name in gpu_names)
        for marker, hinted_gb in _GPU_VRAM_NAME_HINTS_GB:
            if marker in joined_names and hinted_gb > inferred_gb:
                inferred_gb = hinted_gb
                break

    return inferred_gb


def _detect_onnx_directml_available() -> bool:
    try:
        import onnxruntime as ort

        providers = [str(name) for name in ort.get_available_providers()]
        return any(name in providers for name in DIRECTML_PROVIDER_NAMES)
    except Exception:
        return False


def _detect_torch_directml_available() -> tuple[bool, str]:
    try:
        import torch_directml  # type: ignore

        _ = torch_directml.device()
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _detect_cuda_available() -> bool:
    # CUDA availability for faster-whisper/ctranslate2.
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def build_hardware_profile() -> dict[str, Any]:
    gpu_names = gpu_manager.get_video_controller_names()
    gpu_vendor = gpu_manager.detect_gpu_vendor(gpu_names)

    torch_directml_available, torch_directml_error = _detect_torch_directml_available()
    profile = {
        "cpu_cores": _safe_int(os.cpu_count(), 0),
        "ram_gb": round(_detect_total_ram_gb(), 1),
        "gpu_vendor": gpu_vendor,
        "gpu_names": gpu_names,
        "gpu_vram_gb": round(_guess_gpu_vram_gb(gpu_vendor, gpu_names), 1),
        "onnx_directml_available": _detect_onnx_directml_available(),
        "torch_directml_available": torch_directml_available,
        "torch_directml_error": torch_directml_error,
        "cuda_available": _detect_cuda_available(),
        "cuda_download_required": gpu_vendor == "nvidia",
    }
    return profile


def _profile_for_model(model_id: str) -> dict[str, Any]:
    known = _KNOWN_MODEL_PROFILES.get(model_id)
    if known:
        return dict(known)

    model_id_lower = model_id.lower()
    if "large" in model_id_lower:
        return {"quality_score": 5, "relative_cost": 6.2, "min_ram_gb": 8.0, "min_vram_gb": 7.0}
    if "medium" in model_id_lower:
        return {"quality_score": 4, "relative_cost": 4.4, "min_ram_gb": 6.0, "min_vram_gb": 5.0}
    if "small" in model_id_lower:
        return {"quality_score": 3, "relative_cost": 2.8, "min_ram_gb": 4.0, "min_vram_gb": 3.0}
    if "base" in model_id_lower:
        return {"quality_score": 2, "relative_cost": 1.8, "min_ram_gb": 3.0, "min_vram_gb": 2.0}
    if "tiny" in model_id_lower:
        return {"quality_score": 1, "relative_cost": 1.2, "min_ram_gb": 2.0, "min_vram_gb": 1.0}
    return {"quality_score": 3, "relative_cost": 3.4, "min_ram_gb": 4.0, "min_vram_gb": 3.0}


def _choose_effective_device(spec: dict[str, Any], requested_device: str, hardware: dict[str, Any]) -> tuple[str, str]:
    runtime = str(spec.get("runtime", "")).strip().lower()
    requested = (requested_device or "auto").strip().lower()

    cuda_ok = bool(hardware.get("cuda_available"))
    dml_torch_ok = bool(hardware.get("torch_directml_available"))
    dml_onnx_ok = bool(hardware.get("onnx_directml_available"))

    def can_use_cuda() -> bool:
        return runtime in ("faster_whisper", "parakeet_transformers", "onnx") and cuda_ok

    def can_use_directml() -> bool:
        if runtime == "onnx":
            return dml_onnx_ok
        if runtime == "parakeet_transformers":
            return dml_torch_ok
        return False

    if requested == "cpu":
        return "cpu", "requested CPU"

    if requested == "cuda":
        if can_use_cuda():
            return "cuda", ""
        return "cpu", "CUDA requested but unavailable; using CPU"

    if requested == "directml":
        if can_use_directml():
            return "directml", ""
        return "cpu", "DirectML requested but unavailable for this runtime; using CPU"

    # auto
    if runtime == "faster_whisper":
        if can_use_cuda():
            return "cuda", "auto selected CUDA"
        return "cpu", "faster-whisper runs on CUDA or CPU; using CPU"

    if runtime == "parakeet_transformers":
        if can_use_cuda():
            return "cuda", "auto selected CUDA"
        if can_use_directml():
            return "directml", "auto selected DirectML"
        return "cpu", "no GPU runtime available; using CPU"

    if runtime == "onnx":
        if can_use_directml():
            return "directml", "auto selected DirectML"
        if can_use_cuda():
            return "cuda", "auto selected CUDA"
        return "cpu", "no accelerator provider available; using CPU"

    return "cpu", "auto selected CPU"


def _compute_hardware_scores(hardware: dict[str, Any]) -> dict[str, float]:
    cpu_cores = _safe_int(hardware.get("cpu_cores"), 0)
    ram_gb = _safe_float(hardware.get("ram_gb"), 0.0)
    vram_gb = _safe_float(hardware.get("gpu_vram_gb"), 0.0)

    cpu_score = 1.2 + min(32.0, cpu_cores) * 0.22 + max(0.0, ram_gb - 8.0) * 0.08
    cpu_score = _clamp(cpu_score, 1.0, 10.0)

    cuda_score = 0.0
    if hardware.get("cuda_available"):
        cuda_score = 2.6 + vram_gb * 0.85

    dml_score = 0.0
    if hardware.get("torch_directml_available") or hardware.get("onnx_directml_available"):
        dml_score = 2.2 + max(vram_gb, 2.0) * 0.65

    return {
        "cpu": _clamp(cpu_score, 1.0, 10.0),
        "cuda": _clamp(cuda_score, 0.0, 12.0),
        "directml": _clamp(dml_score, 0.0, 11.0),
    }


def estimate_model_performance(
    spec: dict[str, Any],
    hardware: dict[str, Any],
    requested_device: str = "auto",
) -> dict[str, Any]:
    model_id = str(spec.get("model_id", "")).strip()
    runtime = str(spec.get("runtime", "")).strip().lower()
    profile = _profile_for_model(model_id)
    quality_score = int(profile.get("quality_score", 3))
    relative_cost = _safe_float(profile.get("relative_cost", 3.0), 3.0)
    min_ram_gb = _safe_float(profile.get("min_ram_gb", 4.0), 4.0)
    min_vram_gb = _safe_float(profile.get("min_vram_gb", 3.0), 3.0)

    effective_device, fallback_reason = _choose_effective_device(spec=spec, requested_device=requested_device, hardware=hardware)
    scores = _compute_hardware_scores(hardware)
    engine_score = _safe_float(scores.get(effective_device, scores.get("cpu", 1.0)), 1.0)

    raw_perf_index = engine_score / max(relative_cost, 0.1)
    device_multiplier = 1.0
    if effective_device == "cuda":
        device_multiplier = 1.65
    elif effective_device == "directml":
        device_multiplier = 1.35
    estimated_realtime_factor = round(_clamp(raw_perf_index * device_multiplier, 0.2, 8.0), 2)

    ram_gb = _safe_float(hardware.get("ram_gb"), 0.0)
    vram_gb = _safe_float(hardware.get("gpu_vram_gb"), 0.0)

    too_large = False
    too_large_reasons: list[str] = []
    if ram_gb > 0 and min_ram_gb > ram_gb * 0.95:
        too_large = True
        too_large_reasons.append(f"needs ~{min_ram_gb:.0f}GB RAM (system has {ram_gb:.0f}GB)")
    if effective_device in ("cuda", "directml") and vram_gb > 0 and min_vram_gb > vram_gb * 0.95:
        too_large = True
        too_large_reasons.append(f"needs ~{min_vram_gb:.0f}GB VRAM (GPU has {vram_gb:.0f}GB)")

    if too_large:
        status = "too_large"
        status_label = "Too large for this device"
        likely_percent = 10
        performance_score = 1
        reason = "; ".join(too_large_reasons) if too_large_reasons else "model likely exceeds available memory"
    elif estimated_realtime_factor < 0.7:
        status = "too_slow"
        status_label = "Likely too slow"
        likely_percent = 30
        performance_score = 1
        reason = "expected below real-time speed on current hardware"
    elif estimated_realtime_factor < 1.2:
        status = "caution"
        status_label = "Usable with caution"
        likely_percent = 55
        performance_score = 2
        reason = "likely near real-time; may feel laggy on long recordings"
    elif estimated_realtime_factor >= 2.5:
        status = "great"
        status_label = "Great fit"
        likely_percent = 90
        performance_score = 5
        reason = "likely comfortably faster than real-time"
    elif estimated_realtime_factor >= 1.8:
        status = "runnable"
        status_label = "Likely performant"
        likely_percent = 80
        performance_score = 4
        reason = "likely faster than real-time in most cases"
    else:
        status = "runnable"
        status_label = "Likely runnable"
        likely_percent = 70
        performance_score = 3
        reason = "expected around real-time performance"

    if fallback_reason:
        reason = f"{reason}. {fallback_reason}" if reason else fallback_reason
        likely_percent = max(15, likely_percent - 10)

    return {
        "model_id": model_id,
        "runtime": runtime,
        "requested_device": (requested_device or "auto").lower(),
        "effective_device": effective_device,
        "quality_score": int(_clamp(float(quality_score), 1.0, 5.0)),
        "performance_score": int(_clamp(float(performance_score), 1.0, 5.0)),
        "estimated_realtime_factor": estimated_realtime_factor,
        "likely_performant_percent": int(_clamp(float(likely_percent), 5.0, 95.0)),
        "status": status,
        "status_label": status_label,
        "status_reason": reason,
        "min_ram_gb": min_ram_gb,
        "min_vram_gb": min_vram_gb,
    }


def summarize_device_profile(hardware: dict[str, Any]) -> str:
    vendor = str(hardware.get("gpu_vendor", "unknown")).upper()
    cores = _safe_int(hardware.get("cpu_cores"), 0)
    ram = _safe_float(hardware.get("ram_gb"), 0.0)
    vram = _safe_float(hardware.get("gpu_vram_gb"), 0.0)
    dml_ok = bool(hardware.get("torch_directml_available") or hardware.get("onnx_directml_available"))
    cuda_ok = bool(hardware.get("cuda_available"))

    accel = "CPU"
    if cuda_ok:
        accel = "CUDA"
    elif dml_ok:
        accel = "DirectML"

    vram_text = f", {vram:.0f}GB VRAM" if vram > 0 else ""
    summary = f"{vendor} | {cores} CPU cores, {ram:.0f}GB RAM{vram_text} | Best path: {accel}"
    if vendor in ("AMD", "INTEL") and not dml_ok:
        dml_error = str(hardware.get("torch_directml_error", "")).strip()
        if dml_error:
            summary += f" (DirectML unavailable: {dml_error})"
    return summary
