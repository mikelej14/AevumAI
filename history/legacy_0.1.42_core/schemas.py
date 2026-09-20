from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


AFFECT_KEYS = (
    "valence",
    "arousal",
    "anger",
    "sadness",
    "anxiety",
    "curiosity",
    "satisfaction",
    "frustration",
    "trust",
    "confidence",
    "goal_progress",
)


@dataclass
class AffectState:
    valence: float = 0.0          # -1..1
    arousal: float = 0.20         # 0..1
    anger: float = 0.05
    sadness: float = 0.05
    anxiety: float = 0.05
    curiosity: float = 0.45
    satisfaction: float = 0.20
    frustration: float = 0.05
    trust: float = 0.50
    confidence: float = 0.55
    goal_progress: float = 0.50

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AffectState":
        kwargs = {k: float(data.get(k, getattr(cls(), k))) for k in AFFECT_KEYS}
        return cls(**kwargs)


@dataclass
class CognitiveState:
    affect: AffectState = field(default_factory=AffectState)
    active_goals: List[str] = field(default_factory=list)
    open_loops: List[str] = field(default_factory=list)
    attention_priorities: List[str] = field(default_factory=list)
    expectations: List[str] = field(default_factory=list)
    relationship_notes: List[str] = field(default_factory=list)
    turn_number: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "affect": self.affect.to_dict(),
            "active_goals": list(self.active_goals),
            "open_loops": list(self.open_loops),
            "attention_priorities": list(self.attention_priorities),
            "expectations": list(self.expectations),
            "relationship_notes": list(self.relationship_notes),
            "turn_number": self.turn_number,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CognitiveState":
        return cls(
            affect=AffectState.from_dict(data.get("affect", {})),
            active_goals=list(data.get("active_goals", [])),
            open_loops=list(data.get("open_loops", [])),
            attention_priorities=list(data.get("attention_priorities", [])),
            expectations=list(data.get("expectations", [])),
            relationship_notes=list(data.get("relationship_notes", [])),
            turn_number=int(data.get("turn_number", 0)),
        )


@dataclass
class ObserverPacket:
    focus: str = ""
    intent: str = ""
    entities: List[str] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)
    open_loops: List[str] = field(default_factory=list)
    memory_queries: List[str] = field(default_factory=list)
    tone: str = "neutral"
    user_mood: str = "unclear"
    hostility: float = 0.0
    insult: float = 0.0
    praise: float = 0.0
    frustration: float = 0.0
    urgency: float = 0.0
    directed_at_system: bool = False
    correction_detected: bool = False
    salience: float = 0.5
    confidence: float = 0.5

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ObserverPacket":
        obj = cls()
        for k in obj.__dataclass_fields__:
            if k in d:
                setattr(obj, k, d[k])
        obj.entities = [str(x) for x in (obj.entities or [])][:16]
        obj.goals = [str(x) for x in (obj.goals or [])][:8]
        obj.open_loops = [str(x) for x in (obj.open_loops or [])][:8]
        obj.memory_queries = [str(x) for x in (obj.memory_queries or [])][:8]
        for k in ("hostility", "insult", "praise", "frustration", "urgency", "salience", "confidence"):
            setattr(obj, k, max(0.0, min(1.0, float(getattr(obj, k) or 0.0))))
        obj.directed_at_system = bool(obj.directed_at_system)
        obj.correction_detected = bool(obj.correction_detected)
        return obj

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryPacket:
    summary: str = ""
    relevant_record_ids: List[int] = field(default_factory=list)
    connections: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    unresolved: List[str] = field(default_factory=list)
    confidence: float = 0.5

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MemoryPacket":
        obj = cls()
        for k in obj.__dataclass_fields__:
            if k in d:
                setattr(obj, k, d[k])
        obj.relevant_record_ids = [int(x) for x in (obj.relevant_record_ids or [])][:16]
        obj.connections = [str(x) for x in (obj.connections or [])][:12]
        obj.contradictions = [str(x) for x in (obj.contradictions or [])][:8]
        obj.unresolved = [str(x) for x in (obj.unresolved or [])][:8]
        obj.confidence = max(0.0, min(1.0, float(obj.confidence or 0.0)))
        return obj

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AppraisalPacket:
    goal_relevance: float = 0.5
    novelty: float = 0.5
    prediction_error: float = 0.0
    conflict: float = 0.0
    importance: float = 0.5
    uncertainty: float = 0.4
    social_positive: float = 0.0
    social_negative: float = 0.0
    expectation: str = ""
    state_bias: List[str] = field(default_factory=list)
    affect_deltas: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.5

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AppraisalPacket":
        obj = cls()
        for k in obj.__dataclass_fields__:
            if k in d:
                setattr(obj, k, d[k])
        for k in ("goal_relevance", "novelty", "prediction_error", "conflict", "importance", "uncertainty", "social_positive", "social_negative", "confidence"):
            setattr(obj, k, max(0.0, min(1.0, float(getattr(obj, k) or 0.0))))
        clean: Dict[str, float] = {}
        for key, value in (obj.affect_deltas or {}).items():
            if key in AFFECT_KEYS:
                clean[key] = max(-0.25, min(0.25, float(value)))
        obj.affect_deltas = clean
        obj.state_bias = [str(x) for x in (obj.state_bias or [])][:8]
        return obj

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TurnTrace:
    turn_id: str
    user_text: str
    started_at: float
    observer: Optional[Dict[str, Any]] = None
    retrieved: List[Dict[str, Any]] = field(default_factory=list)
    memory_packet: Optional[Dict[str, Any]] = None
    appraisal: Optional[Dict[str, Any]] = None
    state_before: Optional[Dict[str, Any]] = None
    state_after: Optional[Dict[str, Any]] = None
    executive_prompt_chars: int = 0
    executive_prompt_tokens: Optional[int] = None
    assistant_text: str = ""
    executive_thinking: str = ""
    scheduling_mode: str = ""
    scheduling_reason: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_activity: List[Dict[str, Any]] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
