from pathlib import Path

from models.registry import ModelRegistry


def test_registry_resolves_legacy_alias():
    registry = ModelRegistry(manifest_path=Path(__file__).resolve().parents[1] / "models" / "manifest.json")
    spec = registry.resolve_model_spec(model_size="small")
    assert spec["model_id"] == "whisper-small"
    assert spec["legacy_model_size"] == "small"


def test_registry_track_id_is_persisted():
    registry = ModelRegistry(manifest_path=Path(__file__).resolve().parents[1] / "models" / "manifest.json")
    assert registry.track_id == "ralph-wiggum"


def test_parakeet_runtime_is_transformers():
    registry = ModelRegistry(manifest_path=Path(__file__).resolve().parents[1] / "models" / "manifest.json")
    spec = registry.resolve_model_spec(model_id="parakeet-ctc-0.6b")
    assert spec["runtime"] == "parakeet_transformers"
