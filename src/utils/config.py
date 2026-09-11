"""Configuration loading.

GOAL: one place that turns the process environment into a typed, immutable object.

PROBLEM without it: every module reads os.getenv() on its own. Defaults drift apart, a typo
in one module silently produces a different value than the same name in another, and nothing
tells you at startup that a required key is missing — you find out three stages into a call.

CONCEPT: function = collect + coerce + freeze, once, at startup.
         form   = a frozen dataclass built by load_config(), passed down explicitly.

The freeze is the load-bearing part. `frozen=True` means no node can mutate config mid-run,
so a value read in the intake stage is the same value the report stage reads.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    # --- LLM ---
    llm_provider: str
    openai_api_key: str | None
    google_api_key: str | None
    groq_api_key: str | None
    llm_timeout_seconds: int
    max_retries_per_node: int

    # --- transcription ---
    whisper_model_size: str
    confidence_threshold: float
    low_confidence_halt_ratio: float

    # --- persistence ---
    db_path: str
    db_encryption_key: str | None


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _env_opt(name: str) -> str | None:
    """Optional secret. An unset var and an empty var both mean 'absent'.

    .env files routinely carry `OPENAI_API_KEY=` with nothing after it. os.getenv returns ""
    for that, which is truthy-adjacent enough to slip past a bare `if key:` check somewhere
    downstream and produce a 401 instead of a clear 'key not configured'.
    """
    value = os.getenv(name, "").strip()
    return value or None


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    return float(raw) if raw else default


def load_config(env_file: str | None = None) -> Config:
    """Read the environment (optionally seeded from a .env file) into a frozen Config.

    override=False: a variable already exported in the shell beats the .env file. That is the
    behaviour you want in Docker and on HuggingFace Spaces, where secrets arrive as real
    environment variables and no .env file exists at all.
    """
    load_dotenv(dotenv_path=env_file, override=False)

    return Config(
        llm_provider=_env_str("LLM_PROVIDER", "openai").lower(),
        openai_api_key=_env_opt("OPENAI_API_KEY"),
        google_api_key=_env_opt("GOOGLE_API_KEY"),
        groq_api_key=_env_opt("GROQ_API_KEY"),
        llm_timeout_seconds=_env_int("LLM_TIMEOUT_SECONDS", 60),
        max_retries_per_node=_env_int("MAX_RETRIES_PER_NODE", 3),
        whisper_model_size=_env_str("WHISPER_MODEL_SIZE", "tiny"),
        confidence_threshold=_env_float("CONFIDENCE_THRESHOLD", 0.6),
        low_confidence_halt_ratio=_env_float("LOW_CONFIDENCE_HALT_RATIO", 0.5),
        db_path=_env_str("DB_PATH", "data/calls.db"),
        db_encryption_key=_env_opt("DB_ENCRYPTION_KEY"),
    )
