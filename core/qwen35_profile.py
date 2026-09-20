from __future__ import annotations

from typing import Any, Dict

# Canonical text-generation profiles for Qwen3.5, matching the model-card
# recommendations used by this project.  Keep sampler defaults in one place so
# config/UI/worker fallbacks cannot silently drift apart again.
QWEN35_SAMPLER_PROFILE_VERSION = 1

QWEN35_THINKING_TEXT: Dict[str, Any] = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "frequency_penalty": 0.0,
    "repeat_penalty": 1.0,
}

QWEN35_NONTHINKING_TEXT: Dict[str, Any] = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 2.0,
    "frequency_penalty": 0.0,
    "repeat_penalty": 1.0,
}

SAMPLER_KEYS = tuple(QWEN35_THINKING_TEXT.keys())


def sampler_profile(enable_thinking: bool) -> Dict[str, Any]:
    return dict(QWEN35_THINKING_TEXT if enable_thinking else QWEN35_NONTHINKING_TEXT)


def apply_sampler_profile(model_cfg: Dict[str, Any], *, enable_thinking: bool) -> None:
    model_cfg.update(sampler_profile(enable_thinking))
