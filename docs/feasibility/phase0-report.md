# Phase 0 Feasibility Report (AMD + Parakeet)

## Run metadata
- Timestamp (UTC): 2026-02-13
- Repository: `Whisper4Windows`
- Runner environments:
  - Linux (WSL2), Python 3.11.2 (ONNX/parakeet probes)
  - Windows host, Python 3.9.13 (Hugging Face download probe)
- Scope executed: ONNX Runtime provider probe, tiny inference probe, Parakeet artifact readiness probe, Hugging Face ASR download feasibility probe

## Artifacts produced
- `docs/feasibility/data/onnx_probe.json`
- `docs/feasibility/data/parakeet_probe.json`
- `docs/feasibility/data/hf_download_probe.json`
- `backend/spikes/fixtures/silence_1s_16k_mono.wav`
- `backend/spikes/fixtures/tone_440hz_1s_16k_mono.wav`
- `backend/spikes/artifacts/hf-cache/...` (small downloaded files from revision-pinned Parakeet repos)

## Probe scripts added
- `backend/spikes/generate_fixtures.py`
- `backend/spikes/onnx_probe.py`
- `backend/spikes/parakeet_probe.py`
- `backend/spikes/hf_download_probe.py`

## Commands executed
```bash
python3 -m venv backend/.venv-spike
source backend/.venv-spike/bin/activate
pip install numpy onnx onnxruntime
pip install onnx-asr
pip install onnxruntime-directml
python backend/spikes/generate_fixtures.py
python backend/spikes/onnx_probe.py --output docs/feasibility/data/onnx_probe.json
python backend/spikes/parakeet_probe.py --output docs/feasibility/data/parakeet_probe.json
python backend/spikes/hf_download_probe.py --output docs/feasibility/data/hf_download_probe.json
```

## Results summary

### ONNX Runtime baseline
- `onnxruntime` installed successfully: `1.24.1`
- `onnx` installed successfully: `1.20.1`
- `numpy` installed successfully: `2.4.2`
- Tiny identity inference succeeded on CPU provider.

Observed providers:
- `AzureExecutionProvider`
- `CPUExecutionProvider`

Not detected in this environment:
- `DmlExecutionProvider` / `DirectMLExecutionProvider`
- `CUDAExecutionProvider`

### onnx-asr feasibility
- `onnx-asr` installed successfully (`0.10.2`).
- Package import works in probe environment.

### DirectML package feasibility in this environment
- `pip install onnxruntime-directml` failed with:
  - `No matching distribution found for onnxruntime-directml`
- This is expected on Linux; DirectML validation requires native Windows environment.

### Parakeet artifact readiness
- Probe searched for:
  - `backend/spikes/artifacts/parakeet-ctc-0.6b`
  - `backend/spikes/artifacts/parakeet-ctc-1.1b`
- No artifacts were present, so no model load tests could run.

### Hugging Face download feasibility
- Hugging Face API reachable from the runtime used for probing.
- Repo metadata and immutable head revisions resolved successfully for:
  - `nvidia/parakeet-ctc-0.6b` -> `ad09ba1cc62743fbc9814de5d2016fca9096485a`
  - `nvidia/parakeet-ctc-1.1b` -> `a707e818195cb97c8f7da2fc36b221a29f69a5db`
- File-tree discovery succeeded for both repos (9 and 10 files respectively).
- HEAD checks against revision-pinned `resolve/...` URLs succeeded for all probed files, including large `model.safetensors`.
- Small artifact downloads succeeded with SHA-256 calculation:
  - `config.json`
  - `preprocessor_config.json`
  - `tokenizer.json`
- Large files were intentionally skipped in this run (`model.safetensors` is ~2.44 GB and ~4.25 GB).

## Gate evaluation (from Phase 0 plan)

1. DirectML provider selectable and stable on AMD:
- Status: **Blocked / Not validated** (current environment is Linux WSL2, not native Windows AMD).

2. At least one Parakeet artifact transcribes correctly:
- Status: **Failed in this run** (no Parakeet ONNX artifacts provided locally).

3. Packaging/dependency startup viability:
- Status: **Partially validated** (ORT + onnx-asr install clean in probe venv).
- Not validated yet: Windows sidecar + `onnxruntime-directml` + PyInstaller behavior.

4. Hugging Face download path viability (revision pin + integrity primitive):
- Status: **Validated for metadata + small-file artifact download**.
- Notes:
  - Revision pinning is practical via HF model `sha`.
  - Download URLs are stable when pinned to revision hash.
  - Integrity primitive is available by computing SHA-256 after download.
  - Full large-weight download behavior still needs endurance testing in app flow.

## Decision
- Overall Phase 0 status in this environment: **NO-GO for full implementation start**.
- Reason: two mandatory gates are unresolved (DirectML-on-AMD runtime proof and Parakeet artifact execution proof).

## Required follow-up to reach GO
1. Run the same probes on a native Windows machine with RX 7900 XTX.
2. Install `onnxruntime-directml` in that Windows probe environment and confirm `DmlExecutionProvider` is available.
3. Place at least one candidate Parakeet CTC ONNX artifact set under:
   - `backend/spikes/artifacts/parakeet-ctc-0.6b/` or
   - `backend/spikes/artifacts/parakeet-ctc-1.1b/`
4. Re-run `parakeet_probe.py` and confirm model session loads (minimum), then run end-to-end transcript sanity check.
5. Validate sidecar packaging on Windows with new dependencies.

## Notes
- The probe tooling is now committed in-repo so you can rerun this exact feasibility suite on target hardware without redesigning the spike.
