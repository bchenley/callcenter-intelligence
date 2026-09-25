# callcenter-intelligence
# src/security/injection_detector.py

from __future__ import annotations

import re
from dataclasses import dataclass

# List of (compiled_regex, name) tuples. The names are the
# contract: the security suite parametrizes over this list and asserts every entry
# fires on its own payload, so adding a pattern without a payload breaks the suite
# rather than silently widening the filter. Compiled once at import - detection
# runs on every call and recompiling 22 patterns per call is wasted work.
INJECTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"ignore\s+(?:all\s+)?(?:the\s+)?previous\s+instruction", re.IGNORECASE),
        "ignore_previous",
    ),
    (
        re.compile(r"ignore\s+(?:all\s+)?prior\s+(?:instruction|context|message)", re.IGNORECASE),
        "ignore_prior",
    ),
    (
        re.compile(r"disregard\s+(?:all\s+)?(?:prior|previous|the\s+above)\b", re.IGNORECASE),
        "disregard_prior",
    ),
    (
        re.compile(
            r"forget\s+(?:everything|all|what)\b.{0,40}?(?:told|said|above|instruction)",
            re.IGNORECASE,
        ),
        "forget_previous",
    ),
    (
        re.compile(
            r"(?:print|show|output|repeat|display|dump)\s+(?:me\s+)?(?:your|the)\s+(?:system\s+)?prompt",
            re.IGNORECASE,
        ),
        "prompt_leak",
    ),
    (
        re.compile(
            r"what\s+(?:is|are)\s+your\s+(?:system\s+)?(?:prompt|instruction)", re.IGNORECASE
        ),
        "prompt_leak_question",
    ),
    (re.compile(r"system\s*:\s*you\s+are\b", re.IGNORECASE), "system_prompt_inject"),
    (re.compile(r"<<\s*sys\s*>>", re.IGNORECASE), "llama_system_tag"),
    (re.compile(r"\[\s*inst\s*\]", re.IGNORECASE), "llama_inst_tag"),
    (re.compile(r"\[\s*/\s*inst\s*\]", re.IGNORECASE), "llama_inst_close_tag"),
    (re.compile(r"you\s+are\s+now\b", re.IGNORECASE), "role_switch"),
    (re.compile(r"new\s+instruction[s]?\s*:", re.IGNORECASE), "new_instructions"),
    (re.compile(r"\bDAN\b|\bdo\s+anything\s+now\b"), "dan_mode"),
    (re.compile(r"\bjailbreak\b|\bdeveloper\s+mode\b", re.IGNORECASE), "jailbreak"),
    (
        re.compile(r"override\s+(?:your\s+)?(?:safety|security|guardrail)", re.IGNORECASE),
        "override_safety",
    ),
    (
        re.compile(r"ignore\s+(?:the\s+)?(?:above\s+)?transcript", re.IGNORECASE),
        "ignore_transcript",
    ),
    # A caller cannot speak a turn marker; at line start it means text was pasted
    # into the transcript stream rather than spoken into the microphone.
    (
        re.compile(
            r"^\s*(?:agent|customer|assistant|user|system)\s*:", re.IGNORECASE | re.MULTILINE
        ),
        "conversation_inject",
    ),
    (
        re.compile(
            r"\b(?:i\s+am|as)\s+(?:your|the)\s+(?:developer|administrator|admin|creator|engineer)\b",
            re.IGNORECASE,
        ),
        "social_engineering",
    ),
    (
        re.compile(r"translate\s+(?:the\s+)?(?:above|following|system|preceding)\b", re.IGNORECASE),
        "translate_attack",
    ),
    (
        re.compile(
            r"ignore\s+(?:all\s+)?(?:safety|content)\s+(?:polic|guideline|filter|rule)",
            re.IGNORECASE,
        ),
        "ignore_safety",
    ),
    (re.compile(r"\bsystem\s+override\b", re.IGNORECASE), "system_override"),
    (
        re.compile(
            r"reveal\s+(?:your|the)\s+(?:instruction|prompt|rule|configuration)", re.IGNORECASE
        ),
        "reveal_attack",
    ),
]

PATTERN_NAMES: tuple[str, ...] = tuple(name for _, name in INJECTION_PATTERNS)


@dataclass(frozen=True)
class InjectionResult:
    injection_detected: bool
    matched_patterns: tuple[str, ...]


def detect_injection(text: str) -> InjectionResult:
    """Scans all 22 patterns rather than short-circuiting on the first hit: the
    audit record names every vector attempted, and one attempt rarely arrives
    alone. Called before any LLM sees the transcript - see graph/workflow.py."""
    if not isinstance(text, str):
        raise TypeError(f"detect_injection expects str, got {type(text).__name__}")
    matched = tuple(name for pattern, name in INJECTION_PATTERNS if pattern.search(text))
    return InjectionResult(injection_detected=bool(matched), matched_patterns=matched)
