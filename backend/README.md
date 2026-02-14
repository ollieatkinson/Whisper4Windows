# Whisper4Windows Backend

Python FastAPI sidecar for local speech-to-text.

## Run

```bash
cd backend
py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
.\.venv312\Scripts\python.exe main.py
```

Or bootstrap automatically:

```powershell
cd backend
.\bootstrap_env.ps1
.\.venv312\Scripts\python.exe main.py
```

DirectML note:
- For AMD/Intel GPU acceleration with `torch-directml`, use Python `3.10+` (recommended `3.12`).
- Python `3.9` can fail importing `torch-directml` with `'staticmethod' object is not callable`, which forces CPU fallback.

Server URLs:
- `http://127.0.0.1:8000`
- `http://127.0.0.1:8000/docs`

## Model Selection

`StartRequest` keeps backward compatibility and now supports both:
- `model_size` (legacy alias, e.g. `small`)
- `model_id` (preferred, e.g. `whisper-small`, `parakeet-ctc-0.6b`)

If both are provided, `model_id` wins.

## New Metadata (Track + Runtime)

All runtime metadata now carries `track_id = "ralph-wiggum"` from the model manifest.

`GET /health` includes additive fields:
- `model_id`
- `runtime`
- `provider`
- `requested_device`
- `effective_device`
- `fallback_reason`
- `track_id`

## Download APIs

- `POST /models/download`
- `GET /models/download/{model_id}/status`
- `POST /models/download/{model_id}/cancel`
- `GET /models`
- `GET /model/status`
- `POST /transcribe/file` (backend integration and fixture testing)

The Hugging Face download path supports revision-pinned artifacts with checksum verification.

## Notes

- Whisper models use `faster-whisper` and continue to auto-download/load as before.
- Parakeet CTC models run through `transformers` (`AutoModelForCTC` + `AutoProcessor`).
- ONNX runtime scaffolding remains in place for future ONNX-specific model layouts.
