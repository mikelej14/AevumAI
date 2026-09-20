from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


@dataclass(frozen=True)
class ModelPreset:
    key: str
    label: str
    architectures: Tuple[str, ...]
    filename_patterns: Tuple[str, ...]
    text_sampler: Dict[str, Any]
    structured_sampler: Dict[str, Any]
    system_mode: str = "auto"  # auto | native | primer
    thinking_default: bool = False
    thinking_mode: str = "auto"  # auto | template | tags | off


GENERIC_TEXT = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repeat_penalty": 1.05,
}
GENERIC_STRUCTURED = {
    "temperature": 0.25,
    "top_p": 0.9,
    "top_k": 40,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repeat_penalty": 1.0,
}

# These are runtime defaults, not hard restrictions. Embedded GGUF chat templates
# remain authoritative for token/role formatting. Presets only provide sensible
# sampler/system/thinking behavior around that template.
PRESETS: Tuple[ModelPreset, ...] = (
    ModelPreset(
        "qwen35", "Qwen 3.5", ("qwen35",), (r"qwen[-_. ]?3\.5", r"qwen35"),
        {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
         "presence_penalty": 1.5, "frequency_penalty": 0.0, "repeat_penalty": 1.0},
        {"temperature": 1.0, "top_p": 1.0, "top_k": 20, "min_p": 0.0,
         "presence_penalty": 2.0, "frequency_penalty": 0.0, "repeat_penalty": 1.0},
        system_mode="native", thinking_default=True, thinking_mode="template",
    ),
    ModelPreset(
        "qwen3", "Qwen 3", ("qwen3",), (r"qwen[-_. ]?3(?![.\d])",),
        {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
         "presence_penalty": 0.0, "frequency_penalty": 0.0, "repeat_penalty": 1.0},
        {"temperature": 0.2, "top_p": 0.9, "top_k": 20, "min_p": 0.0,
         "presence_penalty": 0.0, "frequency_penalty": 0.0, "repeat_penalty": 1.0},
        system_mode="native", thinking_default=True, thinking_mode="auto",
    ),
    ModelPreset(
        "qwen2", "Qwen 2 / 2.5", ("qwen2",), (r"qwen[-_. ]?2", r"qwen2\.5"),
        {"temperature": 0.7, "top_p": 0.9, "top_k": 40, "min_p": 0.0,
         "presence_penalty": 0.0, "frequency_penalty": 0.0, "repeat_penalty": 1.05},
        dict(GENERIC_STRUCTURED), system_mode="native", thinking_default=False, thinking_mode="tags",
    ),
    ModelPreset(
        "granite42", "Granite 4.2", ("granite",), (r"granite[-_. ]?4\.2",),
        {"temperature": 1.0, "top_p": 0.95, "top_k": 0, "min_p": 0.0,
         "presence_penalty": 0.0, "frequency_penalty": 0.0, "repeat_penalty": 1.0},
        {"temperature": 1.0, "top_p": 0.95, "top_k": 0, "min_p": 0.0,
         "presence_penalty": 0.0, "frequency_penalty": 0.0, "repeat_penalty": 1.0},
        system_mode="native", thinking_default=True, thinking_mode="template",
    ),
    ModelPreset(
        "granite", "Granite", ("granite",), (r"granite",),
        {"temperature": 0.7, "top_p": 0.95, "top_k": 40, "min_p": 0.0,
         "presence_penalty": 0.0, "frequency_penalty": 0.0, "repeat_penalty": 1.0},
        dict(GENERIC_STRUCTURED), system_mode="native", thinking_default=False, thinking_mode="auto",
    ),
    ModelPreset(
        "gemma3", "Gemma 3", ("gemma3",), (r"gemma[-_. ]?3",),
        {"temperature": 0.7, "top_p": 0.95, "top_k": 64, "min_p": 0.0,
         "presence_penalty": 0.0, "frequency_penalty": 0.0, "repeat_penalty": 1.0},
        {"temperature": 0.2, "top_p": 0.9, "top_k": 64, "min_p": 0.0,
         "presence_penalty": 0.0, "frequency_penalty": 0.0, "repeat_penalty": 1.0},
        system_mode="primer", thinking_default=False, thinking_mode="tags",
    ),
    ModelPreset(
        "gemma2", "Gemma 2", ("gemma2", "gemma"), (r"gemma[-_. ]?2",),
        {"temperature": 0.7, "top_p": 0.95, "top_k": 50, "min_p": 0.0,
         "presence_penalty": 0.0, "frequency_penalty": 0.0, "repeat_penalty": 1.1},
        {"temperature": 0.2, "top_p": 0.9, "top_k": 40, "min_p": 0.0,
         "presence_penalty": 0.0, "frequency_penalty": 0.0, "repeat_penalty": 1.0},
        system_mode="primer", thinking_default=False, thinking_mode="off",
    ),
    ModelPreset(
        "gemma", "Gemma", ("gemma",), (r"gemma",),
        {"temperature": 0.7, "top_p": 0.95, "top_k": 50, "min_p": 0.0,
         "presence_penalty": 0.0, "frequency_penalty": 0.0, "repeat_penalty": 1.05},
        dict(GENERIC_STRUCTURED), system_mode="primer", thinking_default=False, thinking_mode="off",
    ),
    ModelPreset(
        "mistral", "Mistral / Mixtral", ("mistral", "mixtral"), (r"mistral", r"mixtral", r"ministral", r"mistral-small", r"mistral-nemo"),
        {"temperature": 0.7, "top_p": 0.95, "top_k": 40, "min_p": 0.0,
         "presence_penalty": 0.0, "frequency_penalty": 0.0, "repeat_penalty": 1.05},
        dict(GENERIC_STRUCTURED), system_mode="auto", thinking_default=False, thinking_mode="tags",
    ),
    ModelPreset(
        "llama", "Llama", ("llama",), (r"llama",),
        {"temperature": 0.7, "top_p": 0.9, "top_k": 40, "min_p": 0.0,
         "presence_penalty": 0.0, "frequency_penalty": 0.0, "repeat_penalty": 1.05},
        dict(GENERIC_STRUCTURED), system_mode="native", thinking_default=False, thinking_mode="tags",
    ),
    ModelPreset(
        "phi", "Phi", ("phi2", "phi3", "phi4", "phi"), (r"(?:^|[-_. ])phi[-_. ]?[234]", r"phi[-_. ]?4", r"phi[-_. ]?3"),
        {"temperature": 0.7, "top_p": 0.95, "top_k": 40, "min_p": 0.0,
         "presence_penalty": 0.0, "frequency_penalty": 0.0, "repeat_penalty": 1.0},
        {"temperature": 0.2, "top_p": 0.9, "top_k": 40, "min_p": 0.0,
         "presence_penalty": 0.0, "frequency_penalty": 0.0, "repeat_penalty": 1.0},
        system_mode="auto", thinking_default=False, thinking_mode="tags",
    ),
    ModelPreset(
        "deepseek_r1", "DeepSeek R1", ("deepseek2", "deepseek"), (r"deepseek[-_. ]?r1", r"r1[-_. ]?distill"),
        {"temperature": 0.6, "top_p": 0.95, "top_k": 40, "min_p": 0.0,
         "presence_penalty": 0.0, "frequency_penalty": 0.0, "repeat_penalty": 1.0},
        {"temperature": 0.2, "top_p": 0.9, "top_k": 40, "min_p": 0.0,
         "presence_penalty": 0.0, "frequency_penalty": 0.0, "repeat_penalty": 1.0},
        system_mode="auto", thinking_default=True, thinking_mode="tags",
    ),
    ModelPreset(
        "deepseek", "DeepSeek", ("deepseek2", "deepseek"), (r"deepseek",),
        {"temperature": 0.7, "top_p": 0.95, "top_k": 40, "min_p": 0.0,
         "presence_penalty": 0.0, "frequency_penalty": 0.0, "repeat_penalty": 1.0},
        dict(GENERIC_STRUCTURED), system_mode="auto", thinking_default=False, thinking_mode="tags",
    ),
    ModelPreset(
        "commandr", "Command-R / Cohere", ("command-r", "command-r7b", "cohere2"), (r"command[-_. ]?r", r"cohere"),
        dict(GENERIC_TEXT), dict(GENERIC_STRUCTURED), system_mode="auto", thinking_default=False, thinking_mode="off",
    ),
    ModelPreset(
        "yi", "Yi", ("llama", "yi"), (r"(?:^|[-_. ])yi[-_. ]",),
        dict(GENERIC_TEXT), dict(GENERIC_STRUCTURED), system_mode="auto", thinking_default=False, thinking_mode="tags",
    ),
    ModelPreset(
        "falcon", "Falcon", ("falcon", "falcon-h1"), (r"falcon",),
        dict(GENERIC_TEXT), dict(GENERIC_STRUCTURED), system_mode="auto", thinking_default=False, thinking_mode="off",
    ),
)

GENERIC_PRESET = ModelPreset(
    "generic", "Generic GGUF", tuple(), tuple(), dict(GENERIC_TEXT), dict(GENERIC_STRUCTURED),
    system_mode="auto", thinking_default=False, thinking_mode="tags",
)

# More specific filename families must win over architecture-only matches. This is
# important for families that share a GGUF architecture (e.g. Yi/Llama-derived).
_FILENAME_ORDER = PRESETS


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def detect_preset(
    architecture: str = "",
    model_name: str = "",
    model_path: str = "",
    chat_template: str = "",
) -> ModelPreset:
    """Detect a best-fit Executive preset without ever rejecting unknown models.

    Metadata is preferred, but filename/name matching can refine ambiguous or shared
    architectures. Unknown models receive GENERIC_PRESET.
    """
    arch = _norm(architecture)
    name_blob = " ".join(x for x in (_norm(model_name), _norm(Path(model_path).name)) if x)
    tmpl = _norm(chat_template)

    # Capability metadata can be more reliable than a shortened/renamed GGUF filename.
    # Granite 4.2 exposes enable_thinking; treat that as the 4.2 reasoning contract even
    # if the user renamed the file to something generic like granite.gguf.
    if arch == "granite" and "enable_thinking" in tmpl and "<think>" in tmpl:
        return next(p for p in PRESETS if p.key == "granite42")

    # Highly distinctive filename/name matches first. This lets Granite 4.2 beat the
    # generic granite architecture, DeepSeek-R1 beat generic DeepSeek, etc.
    for preset in _FILENAME_ORDER:
        if any(re.search(pat, name_blob, flags=re.I) for pat in preset.filename_patterns):
            # Do not let a generic shared-architecture filename accidentally override
            # an explicit conflicting architecture unless the family is known to share it.
            if not arch or arch in preset.architectures or preset.key in {"mistral", "yi"}:
                return preset

    # Exact architecture matches next. Preserve table order for specificity.
    for preset in PRESETS:
        if arch and arch in preset.architectures:
            return preset

    # Template hints are intentionally weak and never used as permission gates.
    if "<|im_start|>" in tmpl and "enable_thinking" in tmpl:
        # Qwen/Granite-style modern thinking template; if metadata did not identify it,
        # use conservative generic behavior with tag separation rather than guessing a family.
        return ModelPreset(
            "generic_thinking", "Generic thinking GGUF", tuple(), tuple(),
            dict(GENERIC_TEXT), dict(GENERIC_STRUCTURED), system_mode="auto",
            thinking_default=True, thinking_mode="template",
        )
    return GENERIC_PRESET


def detect_preset_from_filename(path: str) -> ModelPreset:
    return detect_preset(model_path=path)


def sampler_profile(preset: ModelPreset, *, structured: bool = False) -> Dict[str, Any]:
    return dict(preset.structured_sampler if structured else preset.text_sampler)


def template_supports_variable(chat_template: str, variable: str) -> bool:
    return bool(chat_template and variable.lower() in chat_template.lower())


def template_prompt_opens_thinking(chat_template: str) -> bool:
    """Best-effort detection of a prompt-injected opening <think> block.

    Qwen3.5 and Granite 4.2 put the opening <think> into the generation prompt, so
    the completion can begin with bare reasoning and only later emit </think>.
    """
    t = str(chat_template or "").lower()
    if not t or "<think>" not in t:
        return False
    if "enable_thinking" in t:
        return True
    # If <think> appears near an add_generation_prompt branch, treat it as prompt-open.
    for m in re.finditer(r"add_generation_prompt", t):
        window = t[m.start():m.start() + 1200]
        if "<think>" in window:
            return True
    return False


def preset_summary(preset: ModelPreset) -> str:
    return f"{preset.label} ({preset.key})"


def preset_by_key(key: str) -> ModelPreset:
    k = _norm(key)
    if k == GENERIC_PRESET.key:
        return GENERIC_PRESET
    for preset in PRESETS:
        if preset.key == k:
            return preset
    if k == "generic_thinking":
        return ModelPreset(
            "generic_thinking", "Generic thinking GGUF", tuple(), tuple(),
            dict(GENERIC_TEXT), dict(GENERIC_STRUCTURED), system_mode="auto",
            thinking_default=True, thinking_mode="template",
        )
    return GENERIC_PRESET
