from asr.provider_resolver import select_onnx_provider


def test_directml_requested_and_available():
    result = select_onnx_provider(
        requested_device="directml",
        available_providers=["DmlExecutionProvider", "CPUExecutionProvider"],
    )
    assert result.effective_device == "directml"
    assert result.provider == "DmlExecutionProvider"
    assert result.providers[0] == "DmlExecutionProvider"


def test_auto_falls_back_to_cpu_when_no_accelerator():
    result = select_onnx_provider(
        requested_device="auto",
        available_providers=["CPUExecutionProvider"],
    )
    assert result.effective_device == "cpu"
    assert result.provider == "CPUExecutionProvider"
    assert "CPU" in result.fallback_reason
