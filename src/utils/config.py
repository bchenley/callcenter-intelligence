# callcenter-intelligence
# src/utils/config.py

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    llm_provider: str
    openai_api_key: str | None
    google_api_key: str | None
    groq_api_key: str | None
    llm_timeout_seconds: int
    max_retries_per_node: int
    whisper_model_size: str
    confidence_threshold: float
    low_confidence_halt_ratio: float
    db_path: str
    db_encryption_key: str | None


def _env_raw(name: str, default: str = "") -> str:
    """Read an env var. Docker --env-file does not strip inline comments;
    python-dotenv does. Cut at ' #' so the same .env works in both places."""
    raw = os.getenv(name, default)
    if " #" in raw:
        raw = raw.split(" #", 1)[0]
    return raw.strip()


def _env_str(name: str, default: str) -> str:
    return _env_raw(name, default)


def _env_opt(name: str) -> str | None:
    # An unset var and `KEY=` both mean absent; "" would slip past a truthiness
    # check downstream and surface as a 401 instead of "key not configured".
    return _env_raw(name) or None


def _env_int(name: str, default: int) -> int:
    raw = _env_raw(name)
    return int(raw) if raw else default


def _env_float(name: str, default: float) -> float:
    raw = _env_raw(name)
    return float(raw) if raw else default


def load_config(env_file: str | None = None) -> Config:
    # override=False so real environment variables beat the .env file, which is
    # how secrets arrive in Docker and on HuggingFace Spaces.
    load_dotenv(dotenv_path=env_file, override=False)
    return Config(
        llm_provider=_env_str("LLM_PROVIDER", "openai").lower(),
        openai_api_key=_env_opt("OPENAI_API_KEY"),
        google_api_key=_env_opt("GOOGLE_API_KEY"),
        groq_api_key=_env_opt("GROQ_API_KEY"),
        llm_timeout_seconds=_env_int("LLM_TIMEOUT_SECONDS", 60),
        max_retries_per_node=_env_int("MAX_RETRIES_PER_NODE", 3),
        whisper_model_size=_env_str("WHISPER_MODEL_SIZE", "base"),
        confidence_threshold=_env_float("CONFIDENCE_THRESHOLD", 0.6),
        low_confidence_halt_ratio=_env_float("LOW_CONFIDENCE_HALT_RATIO", 0.5),
        db_path=_env_str("DB_PATH", "data/calls.db"),
        db_encryption_key=_env_opt("DB_ENCRYPTION_KEY"),
    )
