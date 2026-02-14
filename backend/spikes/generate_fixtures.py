#!/usr/bin/env python3
"""
Generate deterministic WAV fixtures for feasibility probes.
"""

from __future__ import annotations

import argparse
import math
import struct
import wave
from pathlib import Path

SAMPLE_RATE = 16_000


def _write_wav_mono_int16(path: Path, samples: list[int], sample_rate: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        frames = b"".join(struct.pack("<h", sample) for sample in samples)
        wav.writeframes(frames)


def _silence(duration_seconds: float) -> list[int]:
    count = int(SAMPLE_RATE * duration_seconds)
    return [0] * count


def _sine_tone(duration_seconds: float, frequency_hz: float, amplitude: float = 0.25) -> list[int]:
    count = int(SAMPLE_RATE * duration_seconds)
    max_int16 = 32767
    samples: list[int] = []
    for idx in range(count):
        t = idx / SAMPLE_RATE
        value = math.sin(2.0 * math.pi * frequency_hz * t) * amplitude
        samples.append(int(value * max_int16))
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate WAV fixtures for ASR probe scripts.")
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent / "fixtures"),
        help="Directory where WAV fixtures will be written.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    silence_path = out_dir / "silence_1s_16k_mono.wav"
    tone_path = out_dir / "tone_440hz_1s_16k_mono.wav"

    _write_wav_mono_int16(silence_path, _silence(1.0))
    _write_wav_mono_int16(tone_path, _sine_tone(1.0, 440.0))

    print(f"Wrote fixture: {silence_path}")
    print(f"Wrote fixture: {tone_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
