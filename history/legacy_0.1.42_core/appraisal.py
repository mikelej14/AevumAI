from __future__ import annotations

import re
from typing import Any, Dict, Iterable

from .schemas import AppraisalPacket, MemoryPacket, ObserverPacket

_WORD_RE = re.compile(r"[A-Za-z0-9_'-]+")


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _tokens(values: Iterable[str]) -> set[str]:
    out: set[str] = set()
    for value in values:
        out.update(x.lower() for x in _WORD_RE.findall(str(value)) if len(x) > 2)
    return out


class DeterministicAppraiser:
    """Deterministic control appraisal.

    Qwen models supply narrow semantic observations and memory reconstruction. This
    component converts those bounded inputs into repeatable state/control signals.
    No model is allowed to directly set emotion or persistent state.
    """

    def pre(self, observer: ObserverPacket, memory: MemoryPacket, state_before: Dict[str, Any]) -> AppraisalPacket:
        active_goals = list(state_before.get("active_goals", []) or [])
        obs_goal_tokens = _tokens(observer.goals + ([observer.focus] if observer.focus else []))
        active_goal_tokens = _tokens(active_goals)
        overlap = 0.0
        if obs_goal_tokens and active_goal_tokens:
            overlap = len(obs_goal_tokens & active_goal_tokens) / max(1, len(obs_goal_tokens | active_goal_tokens))

        goal_relevance = _clamp01(max(
            0.20 if observer.goals else 0.0,
            observer.salience * 0.55 + overlap * 0.65,
        ))

        has_relevant_memory = bool(memory.relevant_record_ids or memory.summary.strip())
        novelty = _clamp01(
            0.72 - (0.38 * memory.confidence if has_relevant_memory else 0.0)
            + 0.20 * observer.salience
            + (0.12 if observer.correction_detected else 0.0)
        )
        contradiction_signal = min(1.0, len(memory.contradictions) / 2.0)
        unresolved_signal = min(1.0, len(memory.unresolved) / 3.0)
        conflict = _clamp01(max(
            contradiction_signal,
            observer.frustration * 0.60,
            observer.hostility * (0.55 if observer.directed_at_system else 0.20),
        ))
        prediction_error = _clamp01(
            (0.68 if observer.correction_detected else 0.0)
            + contradiction_signal * 0.35
            + unresolved_signal * 0.12
        )
        uncertainty = _clamp01(
            0.60 * (1.0 - observer.confidence)
            + 0.35 * (1.0 - memory.confidence)
            + 0.20 * unresolved_signal
        )
        importance = _clamp01(max(
            observer.salience,
            0.58 * goal_relevance + 0.32 * prediction_error + 0.20 * observer.urgency,
        ))
        directed = 1.0 if observer.directed_at_system else 0.35
        social_positive = _clamp01(observer.praise)
        social_negative = _clamp01(max(
            observer.hostility * directed,
            observer.insult * directed,
            observer.frustration * 0.45,
        ))

        state_bias: list[str] = []
        if observer.correction_detected:
            state_bias.append("prioritize correction and reconcile obsolete assumptions")
        if memory.contradictions:
            state_bias.append("resolve recalled contradictions before relying on them")
        if memory.unresolved:
            state_bias.append("preserve unresolved memory uncertainty")
        if observer.urgency >= 0.65:
            state_bias.append("prioritize the user's immediate requested outcome")
        if observer.goals:
            state_bias.append("advance the current explicit goal")
        if social_negative >= 0.65:
            state_bias.append("remain controlled despite negative social signal")

        expectation_parts: list[str] = []
        if observer.intent:
            expectation_parts.append(observer.intent.strip())
        elif observer.focus:
            expectation_parts.append(f"address {observer.focus.strip()}")
        if observer.goals:
            expectation_parts.append(f"advance goal: {observer.goals[0].strip()}")
        expectation = "; ".join(x for x in expectation_parts if x)[:500]

        affect_deltas: Dict[str, float] = {}
        if novelty > 0.55:
            affect_deltas["curiosity"] = min(0.05, (novelty - 0.55) * 0.10)
        if conflict > 0.45:
            affect_deltas["frustration"] = min(0.04, (conflict - 0.45) * 0.08)
        if uncertainty > 0.55:
            affect_deltas["confidence"] = -min(0.05, (uncertainty - 0.55) * 0.11)
        elif memory.confidence > 0.80 and observer.confidence > 0.80:
            affect_deltas["confidence"] = min(0.025, (memory.confidence - 0.80) * 0.08 + 0.01)
        if observer.correction_detected:
            affect_deltas["goal_progress"] = -0.025

        confidence = _clamp01(0.45 + 0.30 * observer.confidence + 0.20 * memory.confidence)
        return AppraisalPacket(
            goal_relevance=goal_relevance,
            novelty=novelty,
            prediction_error=prediction_error,
            conflict=conflict,
            importance=importance,
            uncertainty=uncertainty,
            social_positive=social_positive,
            social_negative=social_negative,
            expectation=expectation,
            state_bias=state_bias[:8],
            affect_deltas=affect_deltas,
            confidence=confidence,
        )

    def post(self, observer: ObserverPacket, memory: MemoryPacket, pre: AppraisalPacket,
             consolidation: Dict[str, Any]) -> AppraisalPacket:
        """Post-response update using only evidence already available in this turn.

        No user reaction has occurred yet, so this method never invents approval,
        rejection, success, or failure. It mainly records consolidation confidence and
        carries unresolved prediction/conflict forward conservatively.
        """
        consolidation_conf = _clamp01(float(consolidation.get("confidence", 0.0) or 0.0))
        consolidation_salience = _clamp01(float(consolidation.get("salience", 0.0) or 0.0))
        lesson = str(consolidation.get("lesson", "") or "").strip()
        what_changed = list(consolidation.get("what_changed", []) or [])

        state_bias = list(pre.state_bias)
        if lesson and consolidation_conf >= 0.65:
            state_bias.append("retain supported lesson from this completed episode")
        if what_changed:
            state_bias.append("carry forward concrete changes from this episode")

        affect_deltas: Dict[str, float] = {}
        # A clean high-confidence consolidation may slightly increase confidence in the
        # memory representation, but it is not treated as social reward.
        if consolidation_conf >= 0.80 and pre.uncertainty < 0.45:
            affect_deltas["confidence"] = 0.01

        return AppraisalPacket(
            goal_relevance=pre.goal_relevance,
            novelty=max(0.0, pre.novelty * 0.85),
            prediction_error=pre.prediction_error * 0.50,
            conflict=pre.conflict * 0.25,
            importance=max(pre.importance, consolidation_salience),
            uncertainty=_clamp01(max(pre.uncertainty * 0.90, 1.0 - consolidation_conf if consolidation else pre.uncertainty)),
            social_positive=0.0,
            social_negative=0.0,
            expectation=pre.expectation,
            state_bias=state_bias[:8],
            affect_deltas=affect_deltas,
            confidence=_clamp01(max(pre.confidence * 0.90, consolidation_conf)),
        )
