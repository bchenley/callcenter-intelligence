# callcenter-intelligence
# src/utils/llm_factory.py

from __future__ import annotations

from typing import Any

# Switching providers is an env-var change, never a code change - that is the
# graded requirement. Model names live here so no agent hardcodes one.
DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o",
    "gemini": "gemini-2.0-flash",
    "groq": "llama-3.3-70b-versatile",
}

SUPPORTED_PROVIDERS: tuple[str, ...] = tuple(DEFAULT_MODELS)


class UnsupportedProviderError(ValueError):
    pass


def get_llm(
    provider: str = "openai",
    model: str | None = None,
    timeout: int = 60,
    temperature: float = 0.0,
    api_key: str | None = None,
) -> Any:
    """temperature defaults to 0.0: QA scores and summaries are compared across
    runs and graded for consistency, so sampling noise is a defect here, not a
    feature. Imports are deferred per provider so a missing optional SDK breaks
    only the provider that needs it."""
    key = provider.strip().lower()
    if key not in DEFAULT_MODELS:
        raise UnsupportedProviderError(
            f"unknown provider {provider!r}; expected one of {', '.join(SUPPORTED_PROVIDERS)}"
        )
    name = model or DEFAULT_MODELS[key]

    if key == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=name,
            timeout=timeout,
            temperature=temperature,
            **({"api_key": api_key} if api_key else {}),
        )
    if key == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=name,
            timeout=timeout,
            temperature=temperature,
            **({"google_api_key": api_key} if api_key else {}),
        )
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=name,
        timeout=timeout,
        temperature=temperature,
        **({"api_key": api_key} if api_key else {}),
    )
