from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable

from .schemas import AffectState, AppraisalPacket, CognitiveState, ObserverPacket, AFFECT_KEYS


BASELINES = {
    "valence": 0.0,
    "arousal": 0.20,
    "anger": 0.04,
    "sadness": 0.04,
    "anxiety": 0.04,
    "curiosity": 0.42,
    "satisfaction": 0.18,
    "frustration": 0.04,
    "trust": 0.50,
    "confidence": 0.55,
    "goal_progress": 0.50,
}


class StateEngine:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load()
        self._last_update = time.time()
        self.lock = threading.RLock()

    def _load(self) -> CognitiveState:
        if not self.path.exists():
            return CognitiveState()
        try:
            return CognitiveState.from_dict(json.loads(self.path.read_text(encoding="utf-8")))
        except Exception:
            return CognitiveState()

    def save(self) -> None:
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.state.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.path)

    @staticmethod
    def _clamp(key: str, value: float) -> float:
        if key == "valence":
            return max(-1.0, min(1.0, value))
        return max(0.0, min(1.0, value))

    def decay(self, now: float | None = None) -> None:
        with self.lock:
            self._decay_locked(now)

    def _decay_locked(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        elapsed_minutes = max(0.0, (now - self._last_update) / 60.0)
        if elapsed_minutes <= 0:
            return
        # Half-lives: fast arousal, slower social/relationship-related state.
        half_lives = {
            "valence": 90, "arousal": 25, "anger": 55, "sadness": 180, "anxiety": 90,
            "curiosity": 180, "satisfaction": 120, "frustration": 65,
            "trust": 4320, "confidence": 720, "goal_progress": 1440,
        }
        for key in AFFECT_KEYS:
            current = getattr(self.state.affect, key)
            base = BASELINES[key]
            factor = math.exp(-math.log(2.0) * elapsed_minutes / half_lives[key])
            setattr(self.state.affect, key, self._clamp(key, base + (current - base) * factor))
        self._last_update = now

    def pre_appraisal_signal(self, observer: ObserverPacket) -> Dict[str, float]:
        """Convert narrow perceptual signals into small deterministic state impulses.

        These are deliberately limited. The 0.8B detector supplies perceptual evidence;
        it does not get authority to directly set emotion.
        """
        directed = 1.0 if observer.directed_at_system else 0.35
        hostility = observer.hostility * directed
        insult = observer.insult * directed
        praise = observer.praise * directed
        correction = 1.0 if observer.correction_detected else 0.0
        frustration = observer.frustration
        return {
            "valence": 0.07 * praise - 0.07 * hostility - 0.04 * insult,
            "arousal": 0.08 * max(hostility, observer.urgency, frustration),
            "anger": 0.045 * hostility + 0.035 * insult + 0.02 * frustration,
            "sadness": 0.025 * insult + 0.015 * hostility,
            "anxiety": 0.025 * observer.urgency + 0.015 * hostility,
            "curiosity": 0.035 * observer.salience - 0.015 * hostility,
            "satisfaction": 0.045 * praise,
            "frustration": 0.045 * frustration + 0.02 * correction,
            "trust": 0.015 * praise - 0.018 * hostility - 0.012 * insult,
            "confidence": -0.025 * correction + 0.012 * praise,
            "goal_progress": -0.012 * correction,
        }

    def apply(self, observer: ObserverPacket, appraisal: AppraisalPacket) -> CognitiveState:
        with self.lock:
            return self._apply_locked(observer, appraisal)

    def _apply_locked(self, observer: ObserverPacket, appraisal: AppraisalPacket) -> CognitiveState:
        self._decay_locked()
        deterministic = self.pre_appraisal_signal(observer)
        app_conf = appraisal.confidence
        for key in AFFECT_KEYS:
            current = getattr(self.state.affect, key)
            delta = deterministic.get(key, 0.0)
            proposed = float(appraisal.affect_deltas.get(key, 0.0)) * min(0.75, app_conf)
            # Inertia prevents a single turn from causing absurd state swings.
            total = max(-0.18, min(0.18, delta + proposed))
            setattr(self.state.affect, key, self._clamp(key, current + total))

        # A few appraisal dimensions influence state even when the deterministic appraiser supplies no direct delta.
        self.state.affect.arousal = self._clamp(
            "arousal", self.state.affect.arousal + 0.025 * appraisal.importance + 0.02 * appraisal.prediction_error
        )
        self.state.affect.curiosity = self._clamp(
            "curiosity", self.state.affect.curiosity + 0.03 * appraisal.novelty * (1.0 - appraisal.conflict)
        )
        self.state.affect.frustration = self._clamp(
            "frustration", self.state.affect.frustration + 0.025 * appraisal.conflict
        )

        self._merge_unique(self.state.active_goals, observer.goals, 12)
        self._merge_unique(self.state.open_loops, observer.open_loops, 12)
        self._merge_unique(self.state.attention_priorities, appraisal.state_bias, 10)
        if appraisal.expectation:
            self._merge_unique(self.state.expectations, [appraisal.expectation], 8)
        self.state.turn_number += 1
        self.save()
        return self.state


    def apply_post(self, appraisal: AppraisalPacket) -> CognitiveState:
        """Apply bounded post-turn appraisal without counting a second interaction turn."""
        with self.lock:
            return self._apply_post_locked(appraisal)

    def _apply_post_locked(self, appraisal: AppraisalPacket) -> CognitiveState:
        self._decay_locked()
        conf = min(0.75, appraisal.confidence)
        for key, proposed in appraisal.affect_deltas.items():
            if key not in AFFECT_KEYS:
                continue
            current = getattr(self.state.affect, key)
            total = max(-0.12, min(0.12, float(proposed) * conf))
            setattr(self.state.affect, key, self._clamp(key, current + total))
        self.state.affect.satisfaction = self._clamp(
            "satisfaction", self.state.affect.satisfaction + 0.02 * appraisal.social_positive
        )
        self.state.affect.frustration = self._clamp(
            "frustration", self.state.affect.frustration + 0.02 * appraisal.conflict
        )
        self._merge_unique(self.state.attention_priorities, appraisal.state_bias, 10)
        if appraisal.expectation:
            self._merge_unique(self.state.expectations, [appraisal.expectation], 8)
        self.save()
        return self.state

    @staticmethod
    def _merge_unique(target: list[str], values: Iterable[str], max_items: int) -> None:
        for value in values:
            s = str(value).strip()
            if not s:
                continue
            if s in target:
                target.remove(s)
            target.append(s)
        if len(target) > max_items:
            del target[: len(target) - max_items]

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return self.state.to_dict()
