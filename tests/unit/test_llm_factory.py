# callcenter-intelligence
# tests/unit/test_llm_factory.py

from __future__ import annotations

import pytest

from src.utils.llm_factory import (
    DEFAULT_MODELS,
    SUPPORTED_PROVIDERS,
    UnsupportedProviderError,
    get_llm,
)


def test_three_providers_supported() -> None:
    assert set(SUPPORTED_PROVIDERS) == {"openai", "gemini", "groq"}


@pytest.mark.parametrize(
    ("provider", "expected_class", "expected_model"),
    [
        ("openai", "ChatOpenAI", "gpt-4o"),
        ("gemini", "ChatGoogleGenerativeAI", "gemini-2.0-flash"),
        ("groq", "ChatGroq", "llama-3.3-70b-versatile"),
    ],
)
def test_each_provider_builds_without_raising(
    provider, expected_class, expected_model, monkeypatch
) -> None:
    """Construction must not require a live key - the reviewer has none."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    llm = get_llm(provider)
    assert type(llm).__name__ == expected_class
    assert DEFAULT_MODELS[provider] == expected_model


def test_unknown_provider_raises_naming_the_options() -> None:
    with pytest.raises(UnsupportedProviderError) as err:
        get_llm("cohere")
    assert "openai" in str(err.value) and "cohere" in str(err.value)


def test_provider_is_case_and_space_insensitive(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert type(get_llm("  OpenAI ")).__name__ == "ChatOpenAI"


def test_model_override_respected(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert get_llm("openai", model="gpt-4o-mini").model_name == "gpt-4o-mini"


def test_temperature_defaults_to_zero(monkeypatch) -> None:
    """Scores compared across runs cannot carry sampling noise."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert get_llm("openai").temperature == 0.0


def test_no_model_name_hardcoded_in_agents() -> None:
    """The factory is the only place a model string may appear."""
    from pathlib import Path

    for path in Path("src/agents").glob("*.py"):
        text = path.read_text()
        for model in DEFAULT_MODELS.values():
            assert model not in text, f"{path} hardcodes {model}"
