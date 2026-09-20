from __future__ import annotations

import json
from datetime import datetime, timezone


class OpenAIUnavailable(RuntimeError):
    pass


class OpenAIChatRuntime:
    """Small Responses-API agent loop with neural-memory function tools."""

    def __init__(self, memory_runtime, *, api_key: str = "", model: str = "gpt-5.6-luna",
                 system_prompt: str = "", user_name: str = "User", assistant_name: str = "Assistant",
                 auto_memory_hints: int = 6):
        self.memory = memory_runtime
        self.api_key = str(api_key or "").strip()
        self.model = str(model or "gpt-5.6-luna").strip()
        self.system_prompt = str(system_prompt or "").strip()
        self.user_name = str(user_name or "User")
        self.assistant_name = str(assistant_name or "Assistant")
        self.auto_memory_hints = max(0, min(12, int(auto_memory_hints or 0)))

    def configure(self, **kwargs):
        for key in ("api_key", "model", "system_prompt", "user_name", "assistant_name"):
            if key in kwargs:
                setattr(self, key, str(kwargs[key] or "").strip())
        if "auto_memory_hints" in kwargs:
            self.auto_memory_hints = max(0, min(12, int(kwargs["auto_memory_hints"] or 0)))

    @staticmethod
    def tool_definitions():
        nullable_string = ["string", "null"]
        return [
            {
                "type": "function", "name": "neural_memory_search",
                "description": (
                    "Search persistent neural memory for earlier conversation or stored material. "
                    "Use concise lexical cues (for example 'car') and optional ISO date range/speaker filters. "
                    "Content matching is performed against neural memory, not chat transcript text."
                ),
                "parameters": {
                    "type": "object", "additionalProperties": False,
                    "properties": {
                        "query": {"type": "string"},
                        "purpose": {"type": "string"},
                        "start_utc": {"type": nullable_string},
                        "end_utc": {"type": nullable_string},
                        "speaker": {"type": nullable_string},
                        "role": {"type": nullable_string, "enum": ["user", "assistant", "source", "tool", None]},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
                    },
                    "required": ["query", "purpose", "start_utc", "end_utc", "speaker", "role", "top_k"],
                },
                "strict": True,
            },
            {
                "type": "function", "name": "neural_memory_open",
                "description": "Open and decode one complete neural memory document/message by document_id.",
                "parameters": {
                    "type": "object", "additionalProperties": False,
                    "properties": {"document_id": {"type": "integer", "minimum": 0}},
                    "required": ["document_id"],
                },
                "strict": True,
            },
            {
                "type": "function", "name": "neural_memory_recent",
                "description": "Return the most recent neurally stored messages/documents, optionally filtered by speaker or role.",
                "parameters": {
                    "type": "object", "additionalProperties": False,
                    "properties": {
                        "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                        "speaker": {"type": nullable_string},
                        "role": {"type": nullable_string, "enum": ["user", "assistant", "source", "tool", None]},
                    },
                    "required": ["limit", "speaker", "role"],
                },
                "strict": True,
            },
            {
                "type": "function", "name": "neural_memory_stats",
                "description": "Return neural-memory counts, storage usage, and decoder timing scales.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False, "required": []},
                "strict": True,
            },
        ]

    def _automatic_hints(self, user_text: str):
        if self.auto_memory_hints <= 0 or not str(user_text or "").strip():
            return []
        try:
            return self.memory.hint_search(str(user_text), top_k=self.auto_memory_hints)
        except Exception:
            return []

    @staticmethod
    def _hint_text(hints):
        if not hints:
            return ""
        lines = [
            "AUTOMATIC NEURAL MEMORY HINTS — metadata only. These are possible older matches, not quoted content. "
            "Do not infer missing facts from them; use neural_memory_open/search for full wording when needed."
        ]
        for h in hints[:12]:
            ann = h.get("semantic") or {}
            bits = [
                f"doc={h.get('document_id')}",
                f"time={h.get('timestamp_utc','')}",
                f"speaker={h.get('speaker','')}",
                f"role={h.get('role','')}",
                "matches=" + ",".join(str(x) for x in (h.get("match_tokens") or [])[:6]),
            ]
            context = str(ann.get("context") or "").strip()
            topics = ann.get("topics") or []
            if context:
                bits.append("context=" + context[:180])
            if topics:
                bits.append("topics=" + ",".join(str(x) for x in topics[:6]))
            lines.append("- " + " | ".join(bits))
        return "\n".join(lines)

    @staticmethod
    def _compact_tool_event(name, args, result):
        if name == "neural_memory_search":
            docs = [{
                "document_id": r.get("document_id"),
                "timestamp_utc": r.get("timestamp_utc", ""),
                "speaker": r.get("speaker", ""),
                "role": r.get("role", ""),
                "match_tokens": r.get("match_tokens", []),
            } for r in list(result or [])[:8]]
            return {"action": name, "ok": True, "arguments": dict(args), "count": len(result or []), "documents": docs}
        if name == "neural_memory_open":
            text = str((result or {}).get("sentence", "") or "")
            return {"action": name, "ok": True, "arguments": dict(args),
                    "document_id": args.get("document_id"), "decoded_chars": len(text)}
        if name == "neural_memory_recent":
            rows = [{
                "document_id": r.get("document_id"), "timestamp_utc": r.get("timestamp_utc", ""),
                "speaker": r.get("speaker", ""), "role": r.get("role", "")
            } for r in list(result or [])[:20]]
            return {"action": name, "ok": True, "arguments": dict(args), "count": len(result or []), "documents": rows}
        if name == "neural_memory_stats":
            return {"action": name, "ok": True, "arguments": dict(args), "stats": dict(result or {})}
        return {"action": name, "ok": True, "arguments": dict(args)}

    @staticmethod
    def _compact_memory_note(name, args, result):
        if name == "neural_memory_search":
            quick = [f"doc {r.get('document_id')} {r.get('timestamp_utc','')} {r.get('speaker','')}" for r in list(result or [])[:8]]
            purpose = str(args.get("purpose") or "current answer")[:180]
            return {"tool": name, "text": f"NEURAL MEMORY SEARCH FOR: {args.get('query','')}. PURPOSE: {purpose}. RESULTS: " + ("; ".join(quick) if quick else "none")}
        return None

    def _execute_tool(self, name, args):
        if name == "neural_memory_search":
            return self.memory.search_full_documents(
                args.get("query", ""), start_utc=args.get("start_utc"), end_utc=args.get("end_utc"),
                speaker=args.get("speaker"), role=args.get("role"), top_k=args.get("top_k", 6),
                include_tool_notes=(str(args.get("role") or "").lower() == "tool"),
            )
        if name == "neural_memory_open":
            return self.memory.open_document(int(args["document_id"]))
        if name == "neural_memory_recent":
            return self.memory.recent(limit=args.get("limit", 8), speaker=args.get("speaker"), role=args.get("role"))
        if name == "neural_memory_stats":
            return self.memory.stats()
        raise KeyError(name)

    def respond(self, messages, *, on_tool=None, max_tool_rounds=8):
        if not self.api_key:
            raise OpenAIUnavailable("No OpenAI API key is configured. Add one in Settings or set OPENAI_API_KEY.")
        try:
            from openai import OpenAI
        except Exception as exc:
            raise OpenAIUnavailable("The openai Python package is not installed. Run the setup script for your operating system.") from exc

        client = OpenAI(api_key=self.api_key)
        input_list = [
            {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
            for m in messages if m.get("role") in {"user", "assistant"} and str(m.get("content", "")).strip()
        ]
        now = datetime.now(timezone.utc).isoformat()
        user_text = next((str(m.get("content", "")) for m in reversed(input_list) if m.get("role") == "user"), "")
        hints = self._automatic_hints(user_text)
        instructions = (
            (self.system_prompt + "\n\n") if self.system_prompt else ""
        ) + (
            f"Current UTC time: {now}. User display name: {self.user_name}. Assistant display name: {self.assistant_name}.\n"
            "The recent chat messages in the input are fast local conversational memory. Persistent neural memory is separate. "
            "Automatic neural hints are metadata-only and must never be treated as remembered wording. "
            "When exact older context matters, use neural_memory_search/open; neural_memory_search returns COMPLETE decoded documents, never snippets. "
            "Neural search is lexical/temporal rather than embedding-semantic, so choose one or a few concrete cue words."
        )
        hint_text = self._hint_text(hints)
        if hint_text:
            instructions += "\n\n" + hint_text
        tools = self.tool_definitions()
        all_tool_events = []
        memory_notes = []

        for _round in range(max(1, int(max_tool_rounds))):
            response = client.responses.create(model=self.model, instructions=instructions, tools=tools, input=input_list)
            input_list += list(response.output)
            calls = [x for x in response.output if getattr(x, "type", "") == "function_call"]
            if not calls:
                return {"text": str(getattr(response, "output_text", "") or ""), "tools": all_tool_events,
                        "memory_notes": memory_notes, "hints": hints, "response_id": getattr(response, "id", "")}
            for call in calls:
                try:
                    args = json.loads(call.arguments or "{}")
                    result = self._execute_tool(call.name, args)
                    event = self._compact_tool_event(call.name, args, result)
                    note = self._compact_memory_note(call.name, args, result)
                    if note:
                        memory_notes.append(note)
                    output = json.dumps(result, ensure_ascii=False, default=str)
                except Exception as exc:
                    event = {"action": getattr(call, "name", "tool"), "ok": False, "arguments": {}, "error": str(exc)}
                    output = json.dumps({"error": str(exc)}, ensure_ascii=False)
                all_tool_events.append(event)
                if on_tool:
                    on_tool(event)
                input_list.append({"type": "function_call_output", "call_id": call.call_id, "output": output})
        raise RuntimeError("OpenAI tool loop exceeded its safety limit")
