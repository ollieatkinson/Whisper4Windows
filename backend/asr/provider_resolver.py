"""
ONNX Runtime provider selection helpers.
"""

from __future__ import annotations

from dataclasses import dataclass


DIRECTML_PROVIDER_NAMES = ("DmlExecutionProvider", "DirectMLExecutionProvider")


@dataclass
class ProviderSelection:
    providers: list[str]
    provider: str
    effective_device: str
    fallback_reason: str = ""


def select_onnx_provider(requested_device: str, available_providers: list[str]) -> ProviderSelection:
    requested = (requested_device or "auto").strip().lower()

    has_cpu = "CPUExecutionProvider" in available_providers
    has_cuda = "CUDAExecutionProvider" in available_providers
    dml = next((name for name in DIRECTML_PROVIDER_NAMES if name in available_providers), "")
    has_directml = bool(dml)

    if requested == "directml":
        if has_directml:
            providers = [dml]
            if has_cpu:
                providers.append("CPUExecutionProvider")
            return ProviderSelection(providers=providers, provider=dml, effective_device="directml")
        if has_cpu:
            return ProviderSelection(
                providers=["CPUExecutionProvider"],
                provider="CPUExecutionProvider",
                effective_device="cpu",
                fallback_reason="directml provider unavailable, fell back to CPU",
            )

    if requested == "cuda":
        if has_cuda:
            providers = ["CUDAExecutionProvider"]
            if has_cpu:
                providers.append("CPUExecutionProvider")
            return ProviderSelection(providers=providers, provider="CUDAExecutionProvider", effective_device="cuda")
        if has_cpu:
            return ProviderSelection(
                providers=["CPUExecutionProvider"],
                provider="CPUExecutionProvider",
                effective_device="cpu",
                fallback_reason="cuda provider unavailable, fell back to CPU",
            )

    if requested == "cpu":
        if has_cpu:
            return ProviderSelection(
                providers=["CPUExecutionProvider"],
                provider="CPUExecutionProvider",
                effective_device="cpu",
            )

    # auto policy: prefer DirectML, then CUDA, then CPU.
    if has_directml:
        providers = [dml]
        if has_cpu:
            providers.append("CPUExecutionProvider")
        return ProviderSelection(providers=providers, provider=dml, effective_device="directml")

    if has_cuda:
        providers = ["CUDAExecutionProvider"]
        if has_cpu:
            providers.append("CPUExecutionProvider")
        return ProviderSelection(
            providers=providers,
            provider="CUDAExecutionProvider",
            effective_device="cuda",
            fallback_reason="directml unavailable, selected CUDA",
        )

    if has_cpu:
        return ProviderSelection(
            providers=["CPUExecutionProvider"],
            provider="CPUExecutionProvider",
            effective_device="cpu",
            fallback_reason="no accelerator provider available, selected CPU",
        )

    return ProviderSelection(
        providers=[],
        provider="",
        effective_device="",
        fallback_reason="no ONNX Runtime providers available",
    )
