"""
Whisper4Windows Backend Server
FastAPI server for local speech-to-text processing
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import gpu_manager
from audio_capture import AudioCapture
from models.downloader import get_download_manager
from models.performance import build_hardware_profile, estimate_model_performance, summarize_device_profile
from models.registry import get_model_registry
from models.storage import get_model_dir, get_models_root
from whisper_engine import WhisperEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Shared singletons
model_registry = get_model_registry()
download_manager = get_download_manager()

# Global runtime state
audio_capture: Optional[AudioCapture] = None
whisper_engine: Optional[WhisperEngine] = None
is_recording = False
transcription_task: Optional[asyncio.Task] = None
last_transcribed_text = ""
is_model_loading = False
model_loading_info = {"model": "", "model_id": "", "status": ""}
current_language: Optional[str] = None


class StartRequest(BaseModel):
    model_size: str = "small"  # compatibility alias
    model_id: Optional[str] = None  # preferred identifier
    language: Optional[str] = None
    device: str = "auto"  # auto, cpu, cuda, directml
    device_index: Optional[int] = None


class StopRequest(BaseModel):
    pass


class ModelDownloadRequest(BaseModel):
    model_id: Optional[str] = None
    model_size: Optional[str] = None
    include_large_files: bool = True
    background: bool = True


class TranscribeFileRequest(BaseModel):
    file_path: str
    model_size: str = "small"
    model_id: Optional[str] = None
    language: Optional[str] = None
    device: str = "auto"
    task: Optional[str] = None


class TranscriptionResponse(BaseModel):
    success: bool
    text: str = ""
    is_final: bool = False
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    backend: str
    model: str
    recording: bool
    model_id: str = ""
    runtime: str = ""
    provider: str = ""
    requested_device: str = ""
    effective_device: str = ""
    fallback_reason: str = ""
    track_id: str = ""


def _resolve_model_spec(model_id: Optional[str], model_size: Optional[str]) -> dict:
    try:
        return model_registry.resolve_model_spec(model_id=model_id, model_size=model_size)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _should_download_before_load(engine: WhisperEngine) -> bool:
    source = str(engine.model_spec.get("download", {}).get("source", "")).lower()
    if engine.runtime == "onnx":
        # ONNX model download should be explicit while decoder integration is incomplete.
        return False
    return source == "huggingface" and not engine.is_model_downloaded(model_id=engine.model_id)


def _resolve_downloaded_model_path(spec: dict, model_id: str, is_downloaded: bool) -> str:
    if not is_downloaded:
        return ""

    runtime = str(spec.get("runtime", "")).strip().lower()
    source = str(spec.get("download", {}).get("source", "")).strip().lower()

    if runtime == "faster_whisper":
        backend_name = str(spec.get("backend_model_name", "")).strip()
        if not backend_name:
            return ""
        return str(get_models_root() / f"models--Systran--faster-whisper-{backend_name}")

    if source == "huggingface":
        return str(get_model_dir(model_id))

    return ""


async def _ensure_model_artifacts_if_needed(engine: WhisperEngine) -> dict:
    if not _should_download_before_load(engine):
        return {"state": "completed", "message": "already present"}

    global is_model_loading, model_loading_info
    is_model_loading = True
    model_loading_info = {
        "model": engine.model_size,
        "model_id": engine.model_id,
        "status": "Downloading model artifacts...",
    }

    loop = asyncio.get_event_loop()

    def _run_download():
        return download_manager.ensure_downloaded(model_id=engine.model_id, include_large_files=True)

    try:
        status = await loop.run_in_executor(None, _run_download)
        return status
    finally:
        is_model_loading = False
        model_loading_info = {"model": "", "model_id": "", "status": ""}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("Whisper4Windows Backend Starting")
    logger.info("Track ID: %s", model_registry.track_id)
    logger.info("=" * 60)
    logger.info("Server: http://127.0.0.1:8000")
    logger.info("API Docs: http://127.0.0.1:8000/docs")
    logger.info("Health Check: http://127.0.0.1:8000/health")
    logger.info("=" * 60)

    logger.info("Checking GPU libraries...")
    gpu_info = gpu_manager.get_gpu_info()
    if gpu_info["gpu_available"]:
        logger.info("NVIDIA GPU detected")
        if gpu_info["libs_installed"]:
            logger.info("GPU libraries installed")
        else:
            logger.warning("GPU detected but libraries not installed")
    else:
        vendor = gpu_info.get("gpu_vendor", "unknown")
        if vendor in ("amd", "intel"):
            logger.info("%s GPU detected - DirectML path available (no CUDA download required)", vendor.upper())
        else:
            logger.info("No NVIDIA GPU detected - CPU/DirectML paths available")

    yield

    logger.info("Shutting down backend...")
    global audio_capture
    if is_recording and audio_capture:
        try:
            audio_capture.stop_recording()
        except Exception:
            pass


app = FastAPI(
    title="Whisper4Windows Backend",
    description="Local speech-to-text processing server",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "app": "Whisper4Windows Backend",
        "version": "1.1.0",
        "status": "running",
        "track_id": model_registry.track_id,
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    backend = "not_loaded"
    model = "not_loaded"
    model_id = ""
    runtime = ""
    provider = ""
    requested_device = ""
    effective_device = ""
    fallback_reason = ""

    if whisper_engine:
        model = whisper_engine.model_size
        model_id = whisper_engine.model_id
        info = whisper_engine.backend_info()
        runtime = str(info.get("runtime", ""))
        provider = str(info.get("provider", ""))
        requested_device = str(info.get("requested_device", ""))
        effective_device = str(info.get("effective_device", whisper_engine.device))
        fallback_reason = str(info.get("fallback_reason", ""))
        backend = effective_device or backend

    return HealthResponse(
        status="ok",
        backend=backend,
        model=model,
        recording=is_recording,
        model_id=model_id,
        runtime=runtime,
        provider=provider,
        requested_device=requested_device,
        effective_device=effective_device,
        fallback_reason=fallback_reason,
        track_id=model_registry.track_id,
    )


@app.get("/models")
async def list_models():
    return {
        "success": True,
        "track_id": model_registry.track_id,
        "default_model_id": model_registry.default_model_id,
        "models": model_registry.list_models(include_disabled=False),
    }


@app.get("/models/performance")
async def model_performance(device: str = "auto"):
    normalized_device = (device or "auto").strip().lower()
    if normalized_device not in ("auto", "cpu", "cuda", "directml"):
        normalized_device = "auto"

    hardware = build_hardware_profile()
    estimates: dict[str, dict] = {}
    for spec in model_registry.list_models(include_disabled=False):
        model_id = str(spec.get("model_id", "")).strip()
        if not model_id:
            continue
        estimates[model_id] = estimate_model_performance(
            spec=spec,
            hardware=hardware,
            requested_device=normalized_device,
        )

    return {
        "success": True,
        "track_id": model_registry.track_id,
        "requested_device": normalized_device,
        "hardware": {
            **hardware,
            "summary": summarize_device_profile(hardware),
        },
        "models": estimates,
    }


@app.get("/model/status")
async def get_model_status(model_size: str = "small", model_id: Optional[str] = None):
    global is_model_loading, model_loading_info, whisper_engine

    spec = _resolve_model_spec(model_id=model_id, model_size=model_size)
    resolved_model_id = spec["model_id"]
    legacy_size = spec["legacy_model_size"]

    if is_model_loading:
        return {
            "success": True,
            "is_loading": True,
            "loading_model": model_loading_info.get("model", ""),
            "loading_model_id": model_loading_info.get("model_id", ""),
            "status": model_loading_info.get("status", "Downloading..."),
            "track_id": model_registry.track_id,
        }

    temp_engine = WhisperEngine(model_size=legacy_size, model_id=resolved_model_id)
    is_downloaded = temp_engine.is_model_downloaded(model_id=resolved_model_id)
    is_loaded = (
        whisper_engine is not None
        and whisper_engine.model_id == resolved_model_id
        and whisper_engine.is_loaded
    )

    download_status = download_manager.get_status(resolved_model_id)
    downloaded_path = _resolve_downloaded_model_path(
        spec=spec,
        model_id=resolved_model_id,
        is_downloaded=is_downloaded,
    )
    return {
        "success": True,
        "is_loading": False,
        "is_downloaded": is_downloaded,
        "is_loaded": is_loaded,
        "model": legacy_size,
        "model_id": resolved_model_id,
        "family": spec.get("family"),
        "runtime": spec.get("runtime"),
        "downloaded_path": downloaded_path,
        "download_status": download_status,
        "track_id": model_registry.track_id,
    }


@app.post("/models/download")
async def start_model_download(request: ModelDownloadRequest):
    spec = _resolve_model_spec(model_id=request.model_id, model_size=request.model_size)
    payload = download_manager.start_download(
        model_id=spec["model_id"],
        include_large_files=request.include_large_files,
        background=request.background,
    )
    payload["track_id"] = model_registry.track_id
    payload["model_id"] = spec["model_id"]
    payload["model"] = spec["legacy_model_size"]
    return payload


@app.get("/models/download/{model_id}/status")
async def get_model_download_status(model_id: str):
    try:
        status = download_manager.get_status(model_id=model_id)
        return {"success": True, "status": status, "track_id": model_registry.track_id}
    except Exception as exc:
        return {"success": False, "error": str(exc), "track_id": model_registry.track_id}


@app.post("/models/download/{model_id}/cancel")
async def cancel_model_download(model_id: str):
    try:
        payload = download_manager.cancel_download(model_id=model_id)
        payload["track_id"] = model_registry.track_id
        return payload
    except Exception as exc:
        return {"success": False, "error": str(exc), "track_id": model_registry.track_id}


@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    try:
        payload = download_manager.delete_model_artifacts(model_id=model_id)
        payload["track_id"] = model_registry.track_id
        return payload
    except Exception as exc:
        return {"success": False, "error": str(exc), "track_id": model_registry.track_id}


@app.post("/load_model")
async def load_model(request: StartRequest):
    global whisper_engine, is_model_loading, model_loading_info, current_language

    spec = _resolve_model_spec(model_id=request.model_id, model_size=request.model_size)
    resolved_model_id = spec["model_id"]
    legacy_size = spec["legacy_model_size"]

    logger.info("Loading model: %s (%s) on %s", legacy_size, resolved_model_id, request.device)
    current_language = None if request.language in (None, "auto") else request.language

    if (
        whisper_engine is not None
        and whisper_engine.model_id == resolved_model_id
        and whisper_engine._original_device == request.device
        and whisper_engine.is_loaded
    ):
        return {
            "status": "success",
            "message": "Model already loaded",
            "model": whisper_engine.model_size,
            "model_id": whisper_engine.model_id,
            "device": whisper_engine.device,
            "track_id": model_registry.track_id,
        }

    whisper_engine = WhisperEngine(
        model_size=legacy_size,
        model_id=resolved_model_id,
        device=request.device,
    )

    artifact_status = await _ensure_model_artifacts_if_needed(whisper_engine)
    if artifact_status.get("state") not in ("completed", "not_started"):
        return {
            "status": "error",
            "message": artifact_status.get("message", "Failed to download model artifacts"),
            "download_status": artifact_status,
            "track_id": model_registry.track_id,
        }

    is_model_loading = True
    model_loading_info = {
        "model": whisper_engine.model_size,
        "model_id": whisper_engine.model_id,
        "status": "Loading model...",
    }

    try:
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(None, whisper_engine.load_model)
    finally:
        is_model_loading = False
        model_loading_info = {"model": "", "model_id": "", "status": ""}

    if not success:
        backend_info = whisper_engine.backend_info()
        return {
            "status": "error",
            "message": backend_info.get("fallback_reason") or "Failed to load model",
            "backend_info": backend_info,
            "track_id": model_registry.track_id,
        }

    return {
        "status": "success",
        "message": "Model loaded successfully",
        "model": whisper_engine.model_size,
        "model_id": whisper_engine.model_id,
        "device": whisper_engine.device,
        "runtime": whisper_engine.runtime,
        "track_id": model_registry.track_id,
    }


@app.post("/start")
async def start_recording(request: StartRequest):
    global audio_capture, whisper_engine, is_recording, current_language

    if is_recording:
        return {"status": "error", "message": "Already recording"}

    spec = _resolve_model_spec(model_id=request.model_id, model_size=request.model_size)
    resolved_model_id = spec["model_id"]
    legacy_size = spec["legacy_model_size"]

    current_language = None if request.language in (None, "auto") else request.language
    logger.info("Starting recording; model=%s (%s), device=%s", legacy_size, resolved_model_id, request.device)

    if (
        whisper_engine is None
        or whisper_engine.model_id != resolved_model_id
        or whisper_engine._original_device != request.device
    ):
        whisper_engine = WhisperEngine(
            model_size=legacy_size,
            model_id=resolved_model_id,
            device=request.device,
        )

    audio_capture = AudioCapture()
    audio_capture.clear_queue()

    device_index = request.device_index if request.device_index is not None else None
    audio_capture.start_recording(device_index=device_index)
    await asyncio.sleep(0.1)

    is_recording = True
    return {
        "status": "started",
        "message": "Recording... Press Alt+T when done",
        "model": whisper_engine.model_size,
        "model_id": whisper_engine.model_id,
        "device": whisper_engine.device,
        "track_id": model_registry.track_id,
    }


@app.post("/stop")
async def stop_recording():
    global is_recording, audio_capture, whisper_engine, is_model_loading, model_loading_info

    if not is_recording:
        return {"status": "error", "message": "Not recording"}

    is_recording = False
    loop = asyncio.get_event_loop()
    audio_data = await loop.run_in_executor(None, audio_capture.stop_recording)
    if audio_data is None or len(audio_data) == 0:
        return {"status": "success", "text": "", "message": "No audio recorded"}

    if whisper_engine is None:
        return {"status": "error", "message": "No model selected"}

    artifact_status = await _ensure_model_artifacts_if_needed(whisper_engine)
    if artifact_status.get("state") not in ("completed", "not_started"):
        return {
            "status": "error",
            "message": artifact_status.get("message", "Failed to download model artifacts"),
            "download_status": artifact_status,
        }

    if not whisper_engine.is_loaded:
        is_model_loading = True
        model_loading_info = {
            "model": whisper_engine.model_size,
            "model_id": whisper_engine.model_id,
            "status": "Loading model...",
        }
        try:
            success = await loop.run_in_executor(None, whisper_engine.load_model)
            if not success:
                backend_info = whisper_engine.backend_info()
                return {
                    "status": "error",
                    "message": backend_info.get("fallback_reason") or "Failed to load model",
                    "backend_info": backend_info,
                }
        finally:
            is_model_loading = False
            model_loading_info = {"model": "", "model_id": "", "status": ""}

    if current_language == "en" and whisper_engine.supports_translate:
        task = "translate"
        model_language = None
    else:
        task = "transcribe"
        model_language = None if current_language in (None, "auto") else current_language

    transcription_start = time.time()
    result = await loop.run_in_executor(
        None,
        whisper_engine.transcribe_audio,
        audio_data,
        model_language,
        task,
    )
    transcription_time = time.time() - transcription_start

    if not result.get("success"):
        return {
            "status": "error",
            "message": result.get("error", "Transcription failed"),
            "backend_info": whisper_engine.backend_info(),
        }

    final_text = str(result.get("text", "")).strip()
    return {
        "status": "success",
        "text": final_text,
        "language": result.get("language", "en"),
        "duration": len(audio_data) / 16000,
        "transcription_time": transcription_time,
        "model": whisper_engine.model_size,
        "model_id": whisper_engine.model_id,
        "device": whisper_engine.device,
        "runtime": whisper_engine.runtime,
        "backend_info": whisper_engine.backend_info(),
        "track_id": model_registry.track_id,
    }


@app.post("/transcribe/file")
async def transcribe_file(request: TranscribeFileRequest):
    global whisper_engine, is_model_loading, model_loading_info

    spec = _resolve_model_spec(model_id=request.model_id, model_size=request.model_size)
    resolved_model_id = spec["model_id"]
    legacy_size = spec["legacy_model_size"]

    if (
        whisper_engine is None
        or whisper_engine.model_id != resolved_model_id
        or whisper_engine._original_device != request.device
    ):
        whisper_engine = WhisperEngine(
            model_size=legacy_size,
            model_id=resolved_model_id,
            device=request.device,
        )

    artifact_status = await _ensure_model_artifacts_if_needed(whisper_engine)
    if artifact_status.get("state") not in ("completed", "not_started"):
        return {
            "status": "error",
            "message": artifact_status.get("message", "Failed to download model artifacts"),
            "download_status": artifact_status,
            "track_id": model_registry.track_id,
        }

    if not whisper_engine.is_loaded:
        is_model_loading = True
        model_loading_info = {
            "model": whisper_engine.model_size,
            "model_id": whisper_engine.model_id,
            "status": "Loading model...",
        }
        try:
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(None, whisper_engine.load_model)
            if not success:
                backend_info = whisper_engine.backend_info()
                return {
                    "status": "error",
                    "message": backend_info.get("fallback_reason") or "Failed to load model",
                    "backend_info": backend_info,
                    "track_id": model_registry.track_id,
                }
        finally:
            is_model_loading = False
            model_loading_info = {"model": "", "model_id": "", "status": ""}

    language = None if request.language in (None, "auto") else request.language
    requested_task = (request.task or "").strip().lower()
    if requested_task not in ("", "transcribe", "translate"):
        return {
            "status": "error",
            "message": f"Unsupported task '{request.task}'. Use 'transcribe' or 'translate'.",
            "track_id": model_registry.track_id,
        }

    if requested_task == "translate":
        if not whisper_engine.supports_translate:
            return {
                "status": "error",
                "message": f"Model '{whisper_engine.model_id}' does not support translation",
                "track_id": model_registry.track_id,
            }
        task = "translate"
        language = None
    elif requested_task == "transcribe":
        task = "transcribe"
    else:
        # Compatibility behavior: language=en implies translate when the model supports it.
        if language == "en" and whisper_engine.supports_translate:
            task = "translate"
            language = None
        else:
            task = "transcribe"

    loop = asyncio.get_event_loop()
    start_time = time.time()
    result = await loop.run_in_executor(
        None,
        whisper_engine.transcribe_file,
        request.file_path,
        language,
        task,
    )
    transcription_time = time.time() - start_time

    if not result.get("success"):
        return {
            "status": "error",
            "message": result.get("error", "Transcription failed"),
            "backend_info": whisper_engine.backend_info(),
            "track_id": model_registry.track_id,
        }

    return {
        "status": "success",
        "text": str(result.get("text", "")).strip(),
        "language": result.get("language", "en"),
        "duration": result.get("duration", 0.0),
        "transcription_time": transcription_time,
        "model": whisper_engine.model_size,
        "model_id": whisper_engine.model_id,
        "device": whisper_engine.device,
        "runtime": whisper_engine.runtime,
        "backend_info": whisper_engine.backend_info(),
        "track_id": model_registry.track_id,
    }


@app.post("/transcribe/upload")
async def transcribe_upload(
    file: UploadFile = File(...),
    model_size: str = Form("small"),
    model_id: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    device: str = Form("auto"),
    task: Optional[str] = Form(None),
):
    original_name = str(file.filename or "").strip().lower()
    if not original_name.endswith(".wav"):
        return {
            "status": "error",
            "message": "Only .wav uploads are supported for manual file transcription.",
            "track_id": model_registry.track_id,
        }

    temp_path = ""
    try:
        payload = await file.read()
        if not payload:
            return {
                "status": "error",
                "message": "Uploaded file is empty.",
                "track_id": model_registry.track_id,
            }

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as handle:
            handle.write(payload)
            temp_path = handle.name

        request = TranscribeFileRequest(
            file_path=temp_path,
            model_size=model_size,
            model_id=model_id,
            language=language,
            device=device,
            task=task,
        )
        return await transcribe_file(request)
    except Exception as exc:
        return {"status": "error", "message": str(exc), "track_id": model_registry.track_id}
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except Exception:
                pass


@app.post("/cancel")
async def cancel_recording():
    global is_recording, audio_capture

    if not is_recording:
        return {"status": "error", "message": "Not recording"}

    is_recording = False
    if audio_capture:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, audio_capture.stop_recording)

    return {
        "status": "success",
        "message": "Recording canceled",
        "track_id": model_registry.track_id,
    }


@app.get("/audio_level")
async def get_audio_level():
    global is_recording, audio_capture

    if not is_recording or not audio_capture:
        return {"level": 0.0, "recording": False}

    if audio_capture.audio_queue.empty():
        return {"level": 0.0, "recording": True}

    queue_size = audio_capture.audio_queue.qsize()
    chunks = []
    temp_chunks = []

    for _ in range(min(5, queue_size)):
        try:
            chunk = audio_capture.audio_queue.get_nowait()
            temp_chunks.append(chunk)
            chunks.append(chunk)
        except Exception:
            break

    for chunk in temp_chunks:
        audio_capture.audio_queue.put(chunk)

    if not chunks:
        return {"level": 0.0, "recording": True}

    audio_data = np.concatenate(chunks, axis=0)
    rms = np.sqrt(np.mean(audio_data**2))
    normalized_level = min(1.0, rms * 3.0)
    return {"level": float(normalized_level), "recording": True, "queue_size": queue_size}


@app.get("/devices")
async def list_devices():
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        input_devices = []
        output_devices = []

        for i, dev in enumerate(devices):
            device_info = {
                "id": i,
                "name": dev["name"],
                "channels": dev["max_input_channels"] if dev["max_input_channels"] > 0 else dev["max_output_channels"],
                "sample_rate": int(dev["default_samplerate"]),
            }

            if dev["max_input_channels"] > 0:
                input_devices.append(device_info)
            if dev["max_output_channels"] > 0:
                output_devices.append(device_info)

        return {"success": True, "inputs": input_devices, "outputs": output_devices}
    except Exception as exc:
        logger.error("Error listing devices: %s", exc)
        return {"success": False, "error": str(exc), "inputs": [], "outputs": []}


@app.get("/gpu/info")
async def get_gpu_info():
    try:
        info = gpu_manager.get_gpu_info()
        return {"success": True, **info, "track_id": model_registry.track_id}
    except Exception as exc:
        logger.error("Error getting GPU info: %s", exc)
        return {"success": False, "error": str(exc), "track_id": model_registry.track_id}


@app.post("/gpu/install")
async def install_gpu_libs():
    try:
        info = gpu_manager.get_gpu_info()
        if not info.get("cuda_download_required", False):
            vendor = str(info.get("gpu_vendor", "unknown")).lower()
            if vendor in ("amd", "intel"):
                message = f"{vendor.upper()} GPU detected. CUDA download is not needed; use DirectML."
            else:
                message = "No NVIDIA GPU detected. CUDA download is not needed on this system."
            return {
                "success": False,
                "error": message,
                "gpu_vendor": vendor,
                "track_id": model_registry.track_id,
            }

        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(None, gpu_manager.install_gpu_libs)
        if success:
            return {
                "success": True,
                "message": "NVIDIA CUDA libraries installed successfully. Restart may be required.",
                "track_id": model_registry.track_id,
            }
        return {
            "success": False,
            "error": "NVIDIA CUDA installation failed. Check logs for details.",
            "track_id": model_registry.track_id,
        }
    except Exception as exc:
        logger.error("GPU installation error: %s", exc)
        return {"success": False, "error": str(exc), "track_id": model_registry.track_id}


@app.post("/gpu/uninstall")
async def uninstall_gpu_libs():
    try:
        success = gpu_manager.uninstall_gpu_libs()
        if success:
            return {"success": True, "message": "GPU libraries removed successfully", "track_id": model_registry.track_id}
        return {"success": False, "error": "Failed to remove GPU libraries", "track_id": model_registry.track_id}
    except Exception as exc:
        logger.error("Error uninstalling GPU libs: %s", exc)
        return {"success": False, "error": str(exc), "track_id": model_registry.track_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
