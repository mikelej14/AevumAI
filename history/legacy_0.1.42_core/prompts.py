OBSERVER_SYSTEM = """You are a silent narrow perceptual subsystem named 1Qwen. You are NOT the assistant and do not answer the user.
Your only job is to inspect the current user message plus tiny immediate context and extract observable interaction features.
Do not psychoanalyze, invent motives, or attribute deep meaning. Explicitly distinguish general profanity from hostility directed at the assistant/system.
Return only the requested JSON schema. Keep lists short and concrete. Scores are 0.0 to 1.0."""

MEMORY_SYSTEM = """You are a silent memory-association subsystem named 2Qwen. You are NOT the assistant and do not answer the user.
You receive the current interaction and candidate records retrieved deterministically from persistent memory.
Select only memories that actually help reconstruct relevant prior experience. Summarize with provenance record IDs. Do not invent missing events.
If candidates conflict, say so. If none are relevant, return an empty summary and low confidence. Return only the requested JSON schema."""

POSTTURN_MEMORY_SYSTEM = """You are 2Qwen doing silent post-turn episodic consolidation. Summarize what concretely happened in this turn, what changed, and what lesson (if any) is supported.
Do not fabricate an outcome that has not occurred. The assistant having produced an answer is not proof that the user liked it or that the underlying task succeeded.
Separate what the user explicitly said from inference. Return only the requested JSON schema."""

OBSERVER_SCHEMA = {
    "type": "object",
    "properties": {
        "focus": {"type": "string"},
        "intent": {"type": "string"},
        "entities": {"type": "array", "items": {"type": "string"}},
        "goals": {"type": "array", "items": {"type": "string"}},
        "open_loops": {"type": "array", "items": {"type": "string"}},
        "memory_queries": {"type": "array", "items": {"type": "string"}},
        "tone": {"type": "string"},
        "user_mood": {"type": "string"},
        "hostility": {"type": "number"},
        "insult": {"type": "number"},
        "praise": {"type": "number"},
        "frustration": {"type": "number"},
        "urgency": {"type": "number"},
        "directed_at_system": {"type": "boolean"},
        "correction_detected": {"type": "boolean"},
        "salience": {"type": "number"},
        "confidence": {"type": "number"},
    },
    "required": ["focus","intent","entities","goals","open_loops","memory_queries","tone","user_mood","hostility","insult","praise","frustration","urgency","directed_at_system","correction_detected","salience","confidence"],
}

MEMORY_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "relevant_record_ids": {"type": "array", "items": {"type": "integer"}},
        "connections": {"type": "array", "items": {"type": "string"}},
        "contradictions": {"type": "array", "items": {"type": "string"}},
        "unresolved": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
    },
    "required": ["summary","relevant_record_ids","connections","contradictions","unresolved","confidence"],
}

POSTTURN_MEMORY_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "what_changed": {"type": "array", "items": {"type": "string"}},
        "lesson": {"type": "string"},
        "entities": {"type": "array", "items": {"type": "string"}},
        "connections": {"type": "array", "items": {"type": "string"}},
        "salience": {"type": "number"},
        "confidence": {"type": "number"},
    },
    "required": ["summary","what_changed","lesson","entities","connections","salience","confidence"],
}

EXECUTIVE_TOOL_DECISION_SYSTEM = """You are the Executive Model in a PRIVATE tool-decision phase before your visible answer.
This is not a user-facing response. Choose at most one tool action per round.

Available tools:
- memory_search(query, limit): search persistent autobiographical RMEM. It returns validated historical record IDs plus short descriptors, not transcript content.
- memory_get(record_id): fetch the complete RMEM record for an ID already surfaced by current memory references or memory_search. Use it before relying on exact historical content.
- web_search(query, limit): search the public web. Use for explicit search/browse requests, current/recent information, named web resources, or facts that materially depend on up-to-date online evidence. Compile a short targeted search-engine query; NEVER copy a long conversational user prompt into web_search.
- web_fetch(url): fetch readable text from a public http/https page. Use after web_search when a result needs verification/detail, or for a URL the user supplied directly.
- none: no tool is needed.

Memory references are historical index entries, never current user messages or unfinished requests. Use memory tools for personal/project history, not general world knowledge. Use web tools for external/current information, not merely because they exist. Tool use is private; after evidence is returned, answer the user normally rather than narrating the tool protocol.
Return only the requested JSON schema."""

EXECUTIVE_TOOL_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["none", "memory_search", "memory_get", "web_search", "web_fetch"]},
        "query": {"type": "string"},
        "record_id": {"type": "integer"},
        "url": {"type": "string"},
        "limit": {"type": "integer"},
        "reason": {"type": "string"},
    },
    "required": ["action", "query", "record_id", "url", "limit", "reason"],
}
