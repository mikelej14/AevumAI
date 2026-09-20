from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

from .model_presets import GENERIC_TEXT as GENERIC_EXECUTIVE_TEXT
from .qwen35_profile import QWEN35_NONTHINKING_TEXT

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"
CONFIG_PATH = DATA_DIR / "config.json"
CHATS_PATH = DATA_DIR / "chats.json"
NEURAL_MEMORY_DIR = DATA_DIR / "neural_memory"


def _cpu_threads_exec() -> int:
    n = os.cpu_count() or 8
    return max(2, min(8, n // 2))


def _cpu_threads_sidecar() -> int:
    n = os.cpu_count() or 8
    return max(2, min(6, n // 3))


def normalize_identity_name(value: Any) -> str:
    text = " ".join(str(value or "").split())
    text = "".join(ch for ch in text if ch.isprintable())
    return (text[:64].strip() or "Assistant")


DEFAULT_CONFIG: Dict[str, Any] = {
    "chat": {
        "provider": "local_gguf",
    },
    "models": {
        "executive": {
            "path": "",
            "n_ctx": 8192,
            "n_batch": 512,
            "n_threads": _cpu_threads_exec(),
            "n_gpu_layers": -1,
            "offload_kqv": True,
            "max_tokens": 2048,
            **GENERIC_EXECUTIVE_TEXT,
            "sampler_profile": "auto",
            "enable_thinking": True,
            "thinking_loop_guard": True,
            "thinking_max_chars": 5000,
            "thinking_repeat_window_words": 48,
            "thinking_repeat_hits": 3,
            "thinking_fallback_nonthinking": True,
        },
        "annotator": {
            "enabled": False,
            "path": "",
            "n_ctx": 3072,
            "n_batch": 256,
            "n_threads": _cpu_threads_sidecar(),
            "n_gpu_layers": 0,
            "offload_kqv": False,
            "max_tokens": 420,
            **QWEN35_NONTHINKING_TEXT,
            "enable_thinking": False,
            "required_architecture": "qwen35",
        },
    },
    "runtime": {
        "recent_turns": 16,
        "recent_context_chars": 14000,
        "auto_memory_hints": 6,
        "tool_rounds": 4,
        "executive_timeout_seconds": 300,
        "annotator_timeout_seconds": 90,
        "auto_start_models": False,
    },
    "browser": {
        "enabled": True,
        "search_provider": "auto",
        "timeout_seconds": 12.0,
        "search_results": 8,
        "fetch_max_bytes": 2_000_000,
        "fetch_max_chars": 18_000,
    },
    "openai": {
        "model": "gpt-5.6-luna",
    },
    "memory": {
        "chunk_tokens": 128,
        "search_limit": 20,
    },
    "identity": {
        "name": "Assistant",
        "user_name": "User",
        "system_prompt": (
            "You are a persistent conversational assistant speaking directly with the user. "
            "Recent chat is your fast local context. Older lived experience is stored in persistent neural memory and may appear as metadata-only hints. "
            "When exact older wording or details matter, use neural memory tools to retrieve the complete decoded document. "
            "Use web tools when current public information matters. Do not invent remembered content from hints alone. "
            "Do not narrate private reasoning, tool mechanics, memory IDs, or runtime architecture unless the user asks."
        ),
        "personality_prompt": (
            "Be direct, capable, curious, grounded, and natural. Keep simple replies concise and expand when the task needs detail. "
            "Let persistent experience and functional state influence continuity without pretending to remember anything that was not actually recorded."
        ),
    },
    "ui": {
        "neural_minimap_quality": "smooth",
        "neural_minimap_speed_ms": 50,
        "neural_minimap_frames": 48,
    },
}


def _merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    out = json.loads(json.dumps(base))
    for k, v in (overlay or {}).items():
        if k not in out:
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _migrate_0142(raw: Dict[str, Any], cfg: Dict[str, Any]) -> bool:
    """Carry forward useful Aevum 0.1.x settings while dropping the old RMEM/two-worker architecture."""
    changed = False
    old_models = raw.get("models") if isinstance(raw.get("models"), dict) else {}
    if "annotator" not in old_models:
        # The old qwen2 slot was the closest semantic/memory worker. Reuse its path as
        # an opt-in suggestion but never make it required for chat.
        q2 = old_models.get("qwen2") if isinstance(old_models.get("qwen2"), dict) else {}
        if q2.get("path"):
            cfg["models"]["annotator"]["path"] = str(q2.get("path"))
            cfg["models"]["annotator"]["enabled"] = True
            changed = True

    old_rt = raw.get("runtime") if isinstance(raw.get("runtime"), dict) else {}
    mapping = {
        "executive_tool_rounds": ("runtime", "tool_rounds"),
        "browser_timeout_seconds": ("browser", "timeout_seconds"),
        "browser_search_results": ("browser", "search_results"),
        "browser_search_provider": ("browser", "search_provider"),
        "browser_enabled": ("browser", "enabled"),
    }
    for old_key, (section, new_key) in mapping.items():
        if old_key in old_rt:
            cfg[section][new_key] = old_rt[old_key]
            changed = True
    return changed


def load_config() -> Dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    NEURAL_MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG_PATH.exists():
        cfg = json.loads(json.dumps(DEFAULT_CONFIG))
        save_config(cfg)
        return cfg
    try:
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raw = {}
    except Exception:
        raw = {}
    cfg = _merge(DEFAULT_CONFIG, raw)
    changed = _migrate_0142(raw, cfg)
    cfg["identity"]["name"] = normalize_identity_name(cfg["identity"].get("name", "Assistant"))
    if changed:
        save_config(cfg)
    return cfg


def save_config(cfg: Dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CONFIG_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(CONFIG_PATH)


def api_key_from_environment() -> str:
    return os.environ.get("OPENAI_API_KEY", "").strip()
