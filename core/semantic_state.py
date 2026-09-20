from __future__ import annotations

import json
import threading
from pathlib import Path


def _clamp(v, lo=0.0, hi=1.0):
    try:
        return max(lo, min(hi, float(v)))
    except Exception:
        return lo


class SemanticState:
    """Small deterministic affect/context state fed by optional Qwen annotations.

    The model supplies bounded observations; Python owns persistence, decay and state
    updates.  This state is context layered on top of the raw neural memory, never a
    substitute for the stored conversation.
    """

    DEFAULT = {
        "valence": 0.0,          # -1..1
        "arousal": 0.15,         # 0..1
        "frustration": 0.0,
        "warmth": 0.5,
        "confidence": 0.6,
        "curiosity": 0.35,
        "last_context": "",
        "last_topics": [],
        "turns_annotated": 0,
    }

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.state = dict(self.DEFAULT)
        self._load()

    def _load(self):
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self.state.update({k: raw[k] for k in self.DEFAULT if k in raw})
        except Exception:
            pass

    def save(self):
        with self.lock:
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.path)

    def apply(self, annotation: dict):
        with self.lock:
            # Inertia prevents one message from flipping the maintained state.
            positivity = _clamp(annotation.get("positivity", 0.0))
            negativity = _clamp(annotation.get("negativity", 0.0))
            hostility = _clamp(annotation.get("hostility", 0.0))
            praise = _clamp(annotation.get("praise", 0.0))
            urgency = _clamp(annotation.get("urgency", 0.0))
            salience = _clamp(annotation.get("salience", 0.0))
            correction = _clamp(annotation.get("correction", 0.0))
            uncertainty = _clamp(annotation.get("uncertainty", 0.0))
            novelty = _clamp(annotation.get("novelty", 0.0))

            target_valence = max(-1.0, min(1.0, positivity - negativity))
            self.state["valence"] = max(-1.0, min(1.0, float(self.state["valence"]) * 0.88 + target_valence * 0.12))
            self.state["arousal"] = _clamp(float(self.state["arousal"]) * 0.90 + max(urgency, hostility, salience) * 0.10)
            self.state["frustration"] = _clamp(float(self.state["frustration"]) * 0.90 + hostility * 0.07 + correction * 0.05)
            self.state["warmth"] = _clamp(float(self.state["warmth"]) * 0.94 + praise * 0.04 - hostility * 0.035)
            self.state["confidence"] = _clamp(float(self.state["confidence"]) * 0.96 - correction * 0.035 - uncertainty * 0.025 + praise * 0.01)
            self.state["curiosity"] = _clamp(float(self.state["curiosity"]) * 0.92 + max(novelty, salience) * 0.08)
            self.state["last_context"] = str(annotation.get("context", "") or "")[:320]
            self.state["last_topics"] = [str(x)[:80] for x in list(annotation.get("topics") or [])[:8]]
            self.state["turns_annotated"] = int(self.state.get("turns_annotated", 0) or 0) + 1
            self.save()
            return self.snapshot()

    def snapshot(self):
        with self.lock:
            return dict(self.state)

    def private_prompt(self):
        s = self.snapshot()
        mood = "neutral"
        if s["valence"] >= 0.25:
            mood = "positive"
        elif s["valence"] <= -0.25:
            mood = "negative"
        parts = [
            f"maintained mood={mood}",
            f"frustration={s['frustration']:.2f}",
            f"warmth={s['warmth']:.2f}",
            f"confidence={s['confidence']:.2f}",
            f"curiosity={s['curiosity']:.2f}",
        ]
        if s.get("last_context"):
            parts.append("recent semantic context=" + str(s["last_context"]))
        if s.get("last_topics"):
            parts.append("recent topics=" + ", ".join(s["last_topics"]))
        return "; ".join(parts)
