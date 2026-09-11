from __future__ import annotations

from src.utils.config import load_config


def test_defaults_apply_when_nothing_is_set(monkeypatch):
    for name in (
        "LLM_PROVIDER", "WHISPER_MODEL_SIZE", "CONFIDENCE_THRESHOLD",
        "LOW_CONFIDENCE_HALT_RATIO", "DB_PATH", "MAX_RETRIES_PER_NODE",
        "LLM_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)

    config = load_config()
    assert config.llm_provider == "openai"
    assert config.whisper_model_size == "tiny"
    assert config.max_retries_per_node == 3


def test_empty_secret_reads_as_none_not_empty_string(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    assert load_config().openai_api_key is None


def test_numeric_values_are_coerced(monkeypatch):
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.75")
    monkeypatch.setenv("MAX_RETRIES_PER_NODE", "5")
    config = load_config()
    assert config.confidence_threshold == 0.75
    assert config.max_retries_per_node == 5


def test_config_is_immutable(monkeypatch):
    import dataclasses
    import pytest

    config = load_config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.llm_provider = "groq"
