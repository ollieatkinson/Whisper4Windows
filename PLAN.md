# Whisper4Windows: AMD + Parakeet Expansion Plan

Execution track ID carried through implementation and telemetry: `ralph-wiggum`

## 1) Review of the rough plan

The rough plan is directionally correct. The strongest choices are:
- Keep `faster-whisper` as the default path for current users.
- Add AMD support through ONNX Runtime + DirectML instead of forcing CTranslate2.
- Treat Parakeet as a separate model family, not a Whisper variant.

Main gaps to close before implementation:
- No feasibility gate before major refactor (high risk around ONNX artifact quality and provider behavior).
- No model capability contract (translation/language behavior differs by model family).
- No compatibility strategy for current saved settings and existing API fields.
- No packaging/runtime matrix (Python version vs `onnxruntime-directml` vs PyInstaller).
- No rollback plan if Parakeet or DirectML path is unstable.

This plan addresses those gaps.

## 2) Scope and success criteria

### In scope (v1)
- Add a second inference runtime using ONNX Runtime with DirectML support.
- Add initial Parakeet support in offline/batch mode (same record/stop UX).
- Keep current Whisper behavior and defaults stable.
- Add backend observability for runtime/provider/model/fallback reason.

### Out of scope (v1)
- True streaming RNNT/TDT.
- New cloud services.
- Full UI redesign.

### Definition of done
- AMD Windows system can run transcription with `DirectMLExecutionProvider`.
- At least one Parakeet CTC model transcribes end-to-end from current hotkey flow.
- Current Whisper model flow still works on CPU and NVIDIA CUDA.
- MSI build includes all required runtime deps; no manual post-install steps for AMD path.

## 3) Key technical decisions

### D1: Two runtime tracks, one API surface
- Keep `faster-whisper` path for Whisper CPU/CUDA.
- Add ONNX Runtime path for DirectML/CPU and Parakeet.
- API remains stable; internals become runtime-pluggable.

### D2: Add capability-aware model registry
- Model entries declare:
  - `model_id`, `family`, `runtime`, `supported_devices`, `supports_translate`, `languages`, `artifact layout`.
- Runtime selection is based on `model_id` + requested device + available providers.

### D3: Preserve backward compatibility
- Existing `model_size` values (`tiny`, `small`, etc.) remain valid.
- New request field `model_id` is introduced; `model_size` is mapped internally.
- Existing `/health` fields remain; new metadata fields are additive.

### D4: Ship DirectML first, defer optional torch backend
- DirectML path is the primary AMD delivery target.
- Optional AMD PyTorch backend is explicitly post-v1.

### D5: Model delivery is in-app download, not bundled
- Parakeet artifacts are downloaded by the app on demand (first use or explicit prefetch).
- Do not bundle Parakeet models in MSI due to size.
- Pin downloads to Hugging Face revision and verify checksum before activation.

## 4) Phased execution with gates

## Phase 0: Feasibility spike (must pass before refactor)

### Goal
Validate that ONNX Runtime + DirectML + Parakeet artifacts are practical in this repo and packaging workflow.

### Work items
- Add `backend/spikes/onnx_probe.py` to print available providers and run a tiny inference probe.
- Add `backend/spikes/parakeet_probe.py` to load candidate Parakeet ONNX artifacts and transcribe a fixture.
- Add `backend/spikes/fixtures/` with short WAV files.
- Record benchmark output (CPU vs DirectML) and startup latency.

### Deliverables
- `docs/feasibility/phase0-report.md` with:
  - providers detected on AMD/NVIDIA/CPU-only boxes,
  - model load time,
  - transcription correctness checks,
  - known blockers.

### Gate (Go/No-Go)
- Go if:
  - DirectML provider is selectable and stable on AMD,
  - at least one Parakeet artifact produces valid transcript output,
  - packaging can include dependencies without startup failure.
- No-Go fallback:
  - Ship DirectML + Whisper ONNX first,
  - defer Parakeet until artifact/tooling is stabilized.

## Phase 1: Backend runtime abstraction

### Files
- Add `backend/asr/base.py`
- Add `backend/asr/types.py`
- Add `backend/asr/selector.py`
- Add `backend/asr/faster_whisper_backend.py`
- Add `backend/asr/onnx_backend.py` (scaffold only)
- Modify `backend/whisper_engine.py` to become a compatibility facade

### Tasks
- Define `ASRBackend` protocol:
  - `load_model(model_spec)`
  - `transcribe(audio, language, task)`
  - `backend_info()`
- Introduce backend selection object:
  - requested device,
  - effective device/provider,
  - fallback reason.
- Keep `WhisperEngine` public methods intact so `backend/main.py` changes are incremental.

### Acceptance
- Existing Whisper flow unchanged for `tiny/base/small/medium/large-v3`.
- Existing endpoints still return successful responses.

## Phase 2: Model registry + artifact management

### Files
- Add `backend/models/registry.py`
- Add `backend/models/manifest.json`
- Add `backend/models/storage.py`
- Add `backend/models/downloader.py`
- Add `backend/models/hf_client.py`
- Modify `backend/whisper_engine.py` to use shared model storage helpers

### Tasks
- Centralize model metadata and supported runtime/device matrix.
- Implement artifact checks (`exists`, `checksum`, `version`).
- Unify model download location under `%APPDATA%/Whisper4Windows/models`.
- Introduce per-model lock file to prevent concurrent partial downloads.
- Add download metadata in manifest:
  - `source` (`huggingface`)
  - `repo_id`
  - `revision`
  - `files` (name, checksum, size)
- Use atomic download flow (download to temp path, validate checksum, then move into active model path).

### Acceptance
- `GET /model/status` works for both Whisper and Parakeet IDs.
- Legacy model names resolve through alias mapping.
- Parakeet is downloaded by app flow; no model file bundling required.

## Phase 3: ONNX Runtime backend implementation

### Files
- Implement `backend/asr/onnx_backend.py`
- Add `backend/asr/provider_resolver.py`
- Add `backend/asr/decoders/ctc_decoder.py`

### Tasks
- Provider detection via `onnxruntime.get_available_providers()`.
- Device selection policy for ONNX backend:
  - `directml` -> DirectML provider required,
  - `cpu` -> CPU provider,
  - `auto` -> DirectML preferred, else CPU.
- Implement CTC decode pipeline (greedy first; optional beam behind flag).
- Return normalized transcription output format matching current `WhisperEngine` contract.

### Acceptance
- ONNX backend can transcribe via CPU and DirectML.
- Provider and fallback reason included in backend info.

## Phase 4: Parakeet model integration

### Files
- Update `backend/models/manifest.json`
- Update `backend/asr/onnx_backend.py`
- Update `backend/main.py` request validation and response metadata

### Tasks
- Add initial Parakeet CTC entries to registry.
- Add capability flags to avoid unsupported features:
  - translation behavior,
  - language constraints,
  - timestamp support.
- Ensure language/task logic in `/stop` is model-aware.
  - Whisper path can keep `translate` behavior.
  - Parakeet path should not silently run unsupported translate logic.

### Acceptance
- Selecting Parakeet model transcribes successfully.
- Unsupported task combinations return clear user-facing errors.

## Phase 5: API surface and status telemetry

### Files
- Modify `backend/main.py`
- Possibly add `backend/api_schemas.py` for request/response models

### Tasks
- Extend `StartRequest`:
  - keep `model_size` for compatibility,
  - add `model_id` (preferred),
  - add `device` value `directml`.
- Extend `/health` response (additive fields):
  - `runtime`, `provider`, `requested_device`, `effective_device`, `fallback_reason`.
- Update `/gpu/info` to generalized accelerator info while preserving old fields used by UI.
- Add model download APIs:
  - `POST /models/download` (start)
  - `GET /models/download/{model_id}/status`
  - `POST /models/download/{model_id}/cancel`

### Acceptance
- Existing frontend calls continue to work with no immediate JS changes.
- New fields appear in API responses and logs.

## Phase 6: Frontend and settings migration

### Files
- Modify `frontend/dist/index.html`
- Modify `frontend/src-tauri/src/lib.rs`

### Tasks
- Device controls:
  - add `DirectML` option,
  - relabel current `GPU` to `CUDA` for clarity.
- Model picker:
  - group models by family (Whisper / Parakeet).
- Backend status badge:
  - display runtime + provider + model.
- Settings migration:
  - map legacy model values to new model IDs if needed,
  - keep old settings file readable.

### Acceptance
- User can select Parakeet + DirectML from existing Configuration screen.
- Settings persist and restore correctly after restart.

## Phase 7: Packaging and installer updates

### Files
- Modify `backend/requirements.txt`
- Modify `backend/build_backend.py`
- Modify `BUILD.md`
- Modify `INSTALLATION.md`

### Tasks
- Add ONNX runtime dependencies and required hidden imports.
- Verify PyInstaller onefile startup with ONNX deps on clean machine.
- Update build docs and installer notes for AMD/DirectML path.
- Explicitly document supported Python version if dependency matrix requires pinning.
- Keep installer model-free for Parakeet (bundle runtime only, never the model artifacts).

### Acceptance
- Built sidecar launches successfully and exposes `/health`.
- MSI install works on systems without Python preinstalled.
- Fresh install can download Parakeet from app UI/API and transcribe after first download.

## Phase 8: Tests, quality gates, and rollout

### Files
- Add `backend/tests/test_provider_selection.py`
- Add `backend/tests/test_model_registry.py`
- Add `backend/tests/test_request_compat.py`
- Add `backend/tests/test_integration_cpu_smoke.py`

### Tasks
- Unit tests for provider/routing logic.
- Unit tests for registry aliasing and capability checks.
- CPU integration smoke test from WAV fixture.
- Add lightweight manual test protocol for AMD and NVIDIA hardware runs.

### Acceptance
- CI passes CPU test suite.
- Manual hardware checklist completed before release tag.

## 5) Current checkpoint and Windows handoff

### Implementation status (backend pass-go update on 2026-02-13)
- Completed foundations in code:
  - Phase 1: backend abstraction scaffold (`backend/asr/*`) and compatibility facade in `backend/whisper_engine.py`
  - Phase 2: model registry/manifest/storage/downloader (`backend/models/*`) with Hugging Face revision pin + checksum verification
  - Phase 5: API surface additions (`model_id`, additive `/health` metadata, model download endpoints)
  - Parakeet runtime working path: `parakeet_transformers` backend (`transformers` + `torch`) with fixture WAV execution
  - Tracking identifier carried through API and registry: `ralph-wiggum`
- Working now:
  - Existing Whisper CPU/CUDA path still loads/transcribes
  - `GET /models`, `POST /models/download`, `GET /models/download/{model_id}/status`, `POST /models/download/{model_id}/cancel`
  - `POST /transcribe/file` for backend fixture/integration transcription
  - `/health` now reports runtime/provider/requested/effective device/fallback + `track_id`
- Remaining:
  - ONNX decode path is scaffolded; Parakeet ONNX end-to-end transcription still pending Phase 3/4 completion
  - DirectML hardware validation and packaging matrix still pending

### Checkpoint status (after mixed Linux/Windows spike work on 2026-02-13)
- Completed:
  - `backend/spikes/generate_fixtures.py`
  - `backend/spikes/onnx_probe.py`
  - `backend/spikes/parakeet_probe.py`
  - `backend/spikes/hf_download_probe.py`
  - `docs/feasibility/data/onnx_probe.json`
  - `docs/feasibility/data/parakeet_probe.json`
  - `docs/feasibility/data/hf_download_probe.json`
  - `docs/feasibility/phase0-report.md`
- Current gate state: `NO-GO` (DirectML on AMD and Parakeet execution on target hardware not yet validated; HF download feasibility is now validated for metadata + small-file download path).

### Resume on Windows (exact commands)
1. Create and activate spike environment:
```bash
py -3.11 -m venv backend\\.venv-spike
backend\\.venv-spike\\Scripts\\activate
pip install --upgrade pip
pip install numpy onnx onnxruntime onnxruntime-directml onnx-asr
```
2. Run probe suite:
```bash
python backend/spikes/generate_fixtures.py
python backend/spikes/onnx_probe.py --output docs/feasibility/data/onnx_probe.windows.json
python backend/spikes/parakeet_probe.py --output docs/feasibility/data/parakeet_probe.windows.json
python backend/spikes/hf_download_probe.py --output docs/feasibility/data/hf_download_probe.windows.json
```
3. Add Parakeet artifacts for local load test under one of:
- `backend/spikes/artifacts/parakeet-ctc-0.6b/`
- `backend/spikes/artifacts/parakeet-ctc-1.1b/`
4. Re-run `parakeet_probe.py` after artifacts are present.
5. Update `docs/feasibility/phase0-report.md` with Windows findings and final Go/No-Go.

### Phase 0 exit criteria before T1+ starts
- ONNX probe reports `DmlExecutionProvider` or `DirectMLExecutionProvider` available on AMD Windows.
- DirectML provider can successfully run the tiny inference probe.
- At least one Parakeet ONNX artifact set can be loaded with ONNX Runtime.
- Windows sidecar packaging sanity check (imports/startup) passes.

## 6) Detailed ticket breakdown

1. `T0` Feasibility spike and report.
2. `T1` Introduce ASR backend protocol and selector.
3. `T2` Add model registry and download/checksum framework.
4. `T3` Implement ONNX backend with DirectML/CPU provider logic.
5. `T4` Integrate Parakeet CTC models and capability-aware task handling.
6. `T5` Extend backend API schemas while preserving compatibility.
7. `T6` Update frontend controls and status display.
8. `T7` Update build packaging and dependency matrix.
9. `T8` Add tests and run final regression matrix.

## 7) Regression matrix (must pass before merge)

1. CPU-only machine:
- Whisper `small` in `auto` and `cpu` modes.
- Parakeet model in `cpu` mode.

2. NVIDIA machine:
- Whisper `small` in `auto` resolves to CUDA runtime.
- Explicit `cuda` works.
- ONNX backend in `cpu` mode still works.

3. AMD machine (RX 7900 XTX target):
- `auto` resolves to DirectML when Parakeet selected.
- Explicit `directml` works.
- Failure fallback to CPU is explicit and visible in `/health`.

4. Persistence:
- Settings survive restart for model/device/language.
- Legacy settings file values still load.

5. Packaging:
- Sidecar starts on first launch.
- Model download and first transcription work on fresh install.

## 8) Rollout strategy

### Release sequencing
- `vNext-alpha1`: backend refactor + DirectML plumbing behind feature flag.
- `vNext-alpha2`: Parakeet CTC exposed in UI.
- `vNext-rc1`: packaging/docs/tests complete.
- `vNext`: general availability.

### Feature flags
- `ENABLE_ONNX_BACKEND`
- `ENABLE_PARAKEET_MODELS`

Default in early phases: off for stable builds, on for dev builds.

## 9) Risks and mitigations

1. Dependency compatibility risk (Python/ORT/PyInstaller).
- Mitigation: lock versions after Phase 0 and validate onefile launch early.

2. Parakeet artifact inconsistency risk.
- Mitigation: registry with checksums + strict artifact contract + fallback path.

3. UX confusion around GPU/CUDA/DirectML.
- Mitigation: explicit labels and provider shown in status indicator.

4. Behavior regression for existing users.
- Mitigation: compatibility facade + legacy request fields + targeted regression tests.

## 10) Review questions before execution

1. Should Parakeet v1 be released as experimental toggle or fully visible by default?
2. Do we require multilingual Parakeet in v1, or allow English-first model set?
3. Is changing backend Python version acceptable if required by ONNX dependency support?
4. Which Hugging Face repo + revision should be pinned for each initial Parakeet model ID?
5. Do we want to keep NVIDIA library downloader as-is, or start deprecating it once ONNX path is stable?
