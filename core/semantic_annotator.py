from __future__ import annotations

import json
from typing import Any


ANNOTATION_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative", "mixed"]},
        "positivity": {"type": "number", "minimum": 0, "maximum": 1},
        "negativity": {"type": "number", "minimum": 0, "maximum": 1},
        "hostility": {"type": "number", "minimum": 0, "maximum": 1},
        "praise": {"type": "number", "minimum": 0, "maximum": 1},
        "urgency": {"type": "number", "minimum": 0, "maximum": 1},
        "salience": {"type": "number", "minimum": 0, "maximum": 1},
        "correction": {"type": "number", "minimum": 0, "maximum": 1},
        "uncertainty": {"type": "number", "minimum": 0, "maximum": 1},
        "novelty": {"type": "number", "minimum": 0, "maximum": 1},
        "directed_at_assistant": {"type": "boolean"},
        "intent": {"type": "string"},
        "context": {"type": "string"},
        "topics": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
        "entities": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
    },
    "required": [
        "sentiment", "positivity", "negativity", "hostility", "praise", "urgency",
        "salience", "correction", "uncertainty", "novelty", "directed_at_assistant",
        "intent", "context", "topics", "entities",
    ],
    "additionalProperties": False,
}


class SemanticAnnotator:
    """Optional Qwen3.5 sidecar.  It annotates; it never gates chat or memory."""

    def __init__(self, worker=None):
        self.worker = worker

    def set_worker(self, worker):
        self.worker = worker

    def annotate(self, text: str, *, role: str, speaker: str, timeout: float = 90.0) -> dict[str, Any]:
        if self.worker is None:
            raise RuntimeError("semantic annotator model is not loaded")
        prompt = (
            "Analyze ONE chat message as a narrow semantic sensor. Return only the requested JSON. "
            "Do not summarize away factual content and do not decide how the assistant should answer. "
            "Scores are 0..1. 'correction' means the speaker is correcting prior information/behavior. "
            "'context' is a short <=240 character description of what this message is doing in context.\n\n"
            f"ROLE: {role}\nSPEAKER: {speaker}\nMESSAGE:\n{text}"
        )
        result = self.worker.chat(
            [{"role": "user", "content": prompt}], timeout=timeout,
            json_schema=ANNOTATION_SCHEMA, stream=False,
            generation={"enable_thinking": False, "max_tokens": 420},
        )
        raw = str(result.get("result", "") or "").strip()
        data = json.loads(raw)
        # Bound strings/lists even if a backend ignores schema constraints.
        data["intent"] = str(data.get("intent", "") or "")[:160]
        data["context"] = str(data.get("context", "") or "")[:320]
        data["topics"] = [str(x)[:80] for x in list(data.get("topics") or [])[:8]]
        data["entities"] = [str(x)[:100] for x in list(data.get("entities") or [])[:12]]
        return data
