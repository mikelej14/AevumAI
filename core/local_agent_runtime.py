from __future__ import annotations

import html
import json
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .browser import BrowserTools
from .model_worker import ModelWorker
from .semantic_annotator import SemanticAnnotator
from .semantic_state import SemanticState

_XML_TOOL_CALL_RE = re.compile(r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>", re.I | re.S)
_XML_TOOL_PARAM_RE = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.I | re.S)


def _clip(s: Any, n: int) -> str:
    text = str(s or "")
    return text if len(text) <= n else text[:n - 1] + "…"



NATIVE_TOOLS = {
    "neural_memory_search": {
        "type": "function", "function": {
            "name": "neural_memory_search",
            "description": (
                "Search persistent neural memory for older conversation or stored material. "
                "IMPORTANT: every hit returns the COMPLETE decoded neural document, not a chunk/snippet. "
                "Use concise concrete cue words and optional ISO UTC date/speaker filters. "
                "Use role='tool' only when the user specifically asks about prior searches/tool activity."
            ),
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string"},
                "purpose": {"type": "string", "description": "Short reason this recall is needed for the current answer."},
                "start_utc": {"type": ["string", "null"]},
                "end_utc": {"type": ["string", "null"]},
                "speaker": {"type": ["string", "null"]},
                "role": {"type": ["string", "null"]},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 8},
            }, "required": ["query"]},
        },
    },
    "neural_memory_open": {
        "type": "function", "function": {
            "name": "neural_memory_open",
            "description": "Open one complete neural document by document_id, usually from an automatic memory hint.",
            "parameters": {"type": "object", "properties": {"document_id": {"type": "integer"}}, "required": ["document_id"]},
        },
    },
    "web_search": {
        "type": "function", "function": {
            "name": "web_search",
            "description": "Search the public web for current, recent, uncertain, or externally verifiable information.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string"}, "purpose": {"type": "string", "description": "Short reason this web search is needed."}, "limit": {"type": "integer", "minimum": 1, "maximum": 12},
            }, "required": ["query"]},
        },
    },
    "web_fetch": {
        "type": "function", "function": {
            "name": "web_fetch",
            "description": "Fetch readable text from a public webpage supplied by the user or surfaced by web_search.",
            "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
        },
    },
    "current_weather": {
        "type": "function", "function": {
            "name": "current_weather",
            "description": "Get current measured weather for a named location.",
            "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]},
        },
    },
}


class LocalModelUnavailable(RuntimeError):
    pass


class LocalAgentRuntime:
    """Single Executive GGUF + optional asynchronous semantic Qwen sidecar.

    The runtime owns deterministic recent-chat context, automatic neural hint preload,
    memory/web tools and tool telemetry.  No worker model is required for recall.
    """

    def __init__(self, memory_runtime, config: dict, *, data_dir: str | Path):
        self.memory = memory_runtime
        self.config = config
        self.data_dir = Path(data_dir)
        self.browser = BrowserTools(self.config.get("browser", {}))
        self.state = SemanticState(self.data_dir / "semantic_state.json")
        self.executive: ModelWorker | None = None
        self.annotator_worker: ModelWorker | None = None
        self.annotator = SemanticAnnotator(None)
        self.lock = threading.RLock()

    def reconfigure(self, config: dict):
        self.config = config
        self.browser.config = self.config.get("browser", {})

    def _model_cfg(self, key: str) -> dict:
        return dict((self.config.get("models") or {}).get(key) or {})

    def load_executive(self, timeout: float = 180.0):
        cfg = self._model_cfg("executive")
        if not str(cfg.get("path", "")).strip():
            raise LocalModelUnavailable("Choose a Granite/chat GGUF in Settings first.")
        with self.lock:
            if self.executive and self.executive.config == cfg and self.executive.status().loaded:
                return self.executive.status().__dict__
            if self.executive:
                self.executive.stop()
            self.executive = ModelWorker("executive", cfg)
            self.executive.start(timeout=timeout)
            return self.executive.status().__dict__

    def load_annotator(self, timeout: float = 180.0):
        cfg = self._model_cfg("annotator")
        if not bool(cfg.get("enabled", False)) or not str(cfg.get("path", "")).strip():
            self.unload_annotator()
            return {"loaded": False, "disabled": True}
        with self.lock:
            if self.annotator_worker and self.annotator_worker.config == cfg and self.annotator_worker.status().loaded:
                self.annotator.set_worker(self.annotator_worker)
                return self.annotator_worker.status().__dict__
            if self.annotator_worker:
                self.annotator_worker.stop()
            self.annotator_worker = ModelWorker("annotator", cfg)
            self.annotator_worker.start(timeout=timeout)
            self.annotator.set_worker(self.annotator_worker)
            return self.annotator_worker.status().__dict__

    def load_configured_models(self):
        results = {}
        try:
            results["executive"] = self.load_executive()
        except Exception as exc:
            results["executive"] = {"loaded": False, "error": str(exc)}
        try:
            results["annotator"] = self.load_annotator()
        except Exception as exc:
            results["annotator"] = {"loaded": False, "error": str(exc)}
        return results

    def unload_annotator(self):
        with self.lock:
            if self.annotator_worker:
                self.annotator_worker.stop()
            self.annotator_worker = None
            self.annotator.set_worker(None)

    def stop(self):
        with self.lock:
            if self.executive:
                self.executive.stop()
            if self.annotator_worker:
                self.annotator_worker.stop()
            self.executive = None
            self.annotator_worker = None
            self.annotator.set_worker(None)

    def statuses(self):
        def status(worker, cfg):
            if worker:
                return worker.status().__dict__
            return {"loaded": False, "alive": False, "model_path": str(cfg.get("path", "")), "error": ""}
        return {
            "executive": status(self.executive, self._model_cfg("executive")),
            "annotator": status(self.annotator_worker, self._model_cfg("annotator")),
        }

    def annotate_message(self, text: str, *, role: str, speaker: str):
        if not self.annotator_worker or not self.annotator_worker.status().loaded:
            return None
        timeout = float((self.config.get("runtime") or {}).get("annotator_timeout_seconds", 90.0))
        ann = self.annotator.annotate(text, role=role, speaker=speaker, timeout=timeout)
        if role == "user":
            self.state.apply(ann)
        return ann

    @staticmethod
    def _native_tool_calls(response: dict) -> tuple[list[dict], str, str]:
        raw = str(response.get("result", "") or "")
        calls = []
        for index, item in enumerate(response.get("tool_calls", []) or []):
            fn = item.get("function", item) if isinstance(item, dict) else {}
            name = str(fn.get("name", "") or "").strip()
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try: args = json.loads(args)
                except Exception: args = {}
            if name and isinstance(args, dict):
                calls.append({"id": str(item.get("id", "") or f"call_{index+1}"), "name": name, "arguments": args})
        if not calls:
            for index, match in enumerate(_XML_TOOL_CALL_RE.finditer(raw)):
                args = {}
                for param in _XML_TOOL_PARAM_RE.finditer(match.group(2)):
                    value = html.unescape(param.group(2).strip())
                    try: value = json.loads(value)
                    except Exception: pass
                    args[html.unescape(param.group(1).strip())] = value
                calls.append({"id": f"call_{index+1}", "name": html.unescape(match.group(1).strip()), "arguments": args})
        cleaned = _XML_TOOL_CALL_RE.sub("", raw).strip() if calls else raw.strip()
        return calls, ("" if calls else cleaned), (cleaned if calls else "")

    @staticmethod
    def _tool_message_call(call: dict) -> dict:
        return {"id": str(call.get("id") or "call_1"), "type": "function", "function": {
            "name": str(call.get("name", "")), "arguments": dict(call.get("arguments") or {})}}

    def _automatic_hints(self, user_text: str):
        lim = max(0, min(12, int((self.config.get("runtime") or {}).get("auto_memory_hints", 6) or 6)))
        if lim <= 0:
            return []
        try:
            return self.memory.hint_search(user_text, top_k=lim)
        except Exception:
            return []

    def _bounded_recent(self, recent_messages: list[dict]) -> list[dict]:
        """Keep newest chat verbatim without crowding tools/recall out of Granite context."""
        budget = max(4000, min(32000, int((self.config.get("runtime") or {}).get("recent_context_chars", 14000) or 14000)))
        kept = []
        remaining = budget
        for msg in reversed(recent_messages):
            role = str(msg.get("role", "user"))
            content = str(msg.get("content", "") or "")
            if role not in {"user", "assistant"} or not content.strip() or remaining <= 0:
                continue
            # Keep the newest message intact whenever it fits; older oversized turns
            # contribute only their most recent tail. This is working context, not
            # archival memory; the complete message remains in neural memory.
            if len(content) > remaining:
                content = content[-remaining:]
            kept.append({"role": role, "content": content})
            remaining -= len(content)
        return list(reversed(kept))

    @staticmethod
    def _hint_text(hints: list[dict]) -> str:
        if not hints:
            return ""
        lines = [
            "AUTOMATIC NEURAL MEMORY HINTS — metadata only. These are possible older matches, NOT quoted content. "
            "Do not invent what they say. Use neural_memory_open or neural_memory_search when full historical content is needed."
        ]
        for h in hints:
            sem = h.get("semantic") or {}
            tail = []
            if sem.get("topics"): tail.append("topics=" + ",".join(str(x) for x in sem.get("topics", [])[:5]))
            if sem.get("context"): tail.append("context=" + _clip(sem.get("context"), 180))
            lines.append(
                f"- doc={h.get('document_id')} time={h.get('timestamp_utc','')} speaker={h.get('speaker','')} "
                f"role={h.get('role','')} cue={','.join(h.get('match_tokens') or [])} " + (" | ".join(tail))
            )
        return "\n".join(lines)

    def _system_prompt(self, user_text: str, hints: list[dict]) -> str:
        chat_cfg = self.config.get("chat") or {}
        ident = self.config.get("identity") or {}
        base = str(chat_cfg.get("system_prompt", "") or ident.get("system_prompt", "") or "").strip()
        personality = str(ident.get("personality_prompt", "") or "").strip()
        uname = str(ident.get("user_name", "User"))
        aname = str(ident.get("assistant_name") or ident.get("name") or "Assistant")
        now = datetime.now(timezone.utc).isoformat()
        parts = [base] if base else []
        if personality:
            parts.append("PERSONALITY / VOICE:\n" + personality)
        parts.append(
            f"Current UTC: {now}. User name: {uname}. Your configured name: {aname}.\n"
            "The recent chat messages supplied after this system message are your fast local conversational memory. "
            "Persistent neural memory is separate. Automatic neural hints are metadata-only; never treat a hint as the contents of a memory. "
            "If historical wording/details would improve the answer, you may call neural_memory_search or neural_memory_open. "
            "Explicit neural_memory_search returns COMPLETE decoded documents, not snippets. Automatic hints are suggestions only: decide yourself whether any memory needs to be opened, searched, or ignored; never guess content from hint metadata.\n"
            "Use web_search/web_fetch for current or externally verifiable information when useful. Tool evidence is private working context. "
            "Answer normally; do not narrate tool mechanics or memory IDs unless the user asks."
        )
        state = self.state.private_prompt()
        if state:
            parts.append("PRIVATE FUNCTIONAL STATE (derived by optional semantic sidecar; use subtly, do not report raw scores):\n" + state)
        ht = self._hint_text(hints)
        if ht:
            parts.append(ht)
        return "\n\n".join(x for x in parts if x.strip())

    def _execute_tool(self, name: str, args: dict, *, user_text: str):
        """Return (private_model_result, compact_receipt, compact_memory_note)."""
        started = time.perf_counter()
        if name == "neural_memory_search":
            result = self.memory.search_full_documents(
                str(args.get("query", "")), start_utc=args.get("start_utc"), end_utc=args.get("end_utc"),
                speaker=args.get("speaker"), role=args.get("role"), top_k=max(1, min(8, int(args.get("top_k", 4) or 4))),
                include_tool_notes=(str(args.get("role") or "").lower() == "tool"),
            )
            quick_rows = [{
                "document_id": r.get("document_id"), "timestamp_utc": r.get("timestamp_utc", ""),
                "speaker": r.get("speaker", ""), "role": r.get("role", ""),
                "match_tokens": r.get("match_tokens", []),
                "matched_memory_ids": r.get("matched_memory_ids", []),
            } for r in result[:8]]
            event = {"action": name, "ok": True, "arguments": dict(args), "count": len(result), "documents": quick_rows}
            quick = [f"doc {r.get('document_id')} {r.get('timestamp_utc','')} {r.get('speaker','')}" for r in result[:8]]
            purpose = _clip(args.get("purpose") or user_text or "current answer", 180)
            note = f"NEURAL MEMORY SEARCH FOR: {args.get('query','')}. PURPOSE: {purpose}. RESULTS: " + ("; ".join(quick) if quick else "none")
        elif name == "neural_memory_open":
            result = self.memory.open_document(int(args.get("document_id", -1)))
            event = {"action": name, "ok": True, "arguments": dict(args), "document_id": int(args.get("document_id", -1)),
                     "decoded_chars": len(str(result.get("sentence", "") or ""))}
            note = ""
        elif name == "web_search":
            result = self.browser.search(str(args.get("query", "")), limit=max(1, min(12, int(args.get("limit", 8) or 8))))
            compact_results = [{"title": _clip(r.get("title", ""), 160), "url": _clip(r.get("url", ""), 240)}
                               for r in list(result.get("results") or [])[:8]]
            event = {"action": name, "ok": bool(result.get("ok")), "arguments": dict(args),
                     "query": result.get("query", args.get("query", "")), "provider": result.get("provider", ""),
                     "count": len(result.get("results") or []), "results": compact_results,
                     "error": result.get("error", "")}
            quick = [f"{r['title']} — {r['url']}" for r in compact_results]
            purpose = _clip(args.get("purpose") or user_text or "current answer", 180)
            note = f"WEB SEARCH FOR: {args.get('query','')}. PURPOSE: {purpose}. RESULTS: " + ("; ".join(quick) if quick else "none")
        elif name == "web_fetch":
            result = self.browser.fetch(str(args.get("url", "")))
            event = {"action": name, "ok": bool(result.get("ok")), "arguments": dict(args),
                     "url": result.get("url", args.get("url", "")), "title": result.get("title", ""),
                     "chars": len(str(result.get("text", "") or "")), "error": result.get("error", "")}
            note = ""
        elif name == "current_weather":
            result = self.browser.current_weather(str(args.get("location", "")))
            event = {"action": name, "ok": bool(result.get("ok")), "arguments": dict(args),
                     "location": args.get("location", ""), "resolved_location": result.get("resolved_location", ""),
                     "observed_at": result.get("observed_at", ""), "condition": result.get("condition", ""),
                     "temperature_f": result.get("temperature_f"), "apparent_temperature_f": result.get("apparent_temperature_f"),
                     "relative_humidity_percent": result.get("relative_humidity_percent"), "wind_speed_mph": result.get("wind_speed_mph"),
                     "error": result.get("error", "")}
            note = f"CURRENT WEATHER LOOKUP FOR: {args.get('location','')}. RESULT: {_clip(result.get('condition') or result.get('error'),160)}"
        else:
            raise KeyError(name)
        event["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return result, event, note

    @staticmethod
    def _wire_tool_result(name: str, result: Any) -> str:
        return json.dumps({"tool": name, "result": result}, ensure_ascii=False, separators=(",", ":"), default=str)

    def respond(self, recent_messages: list[dict], *, on_tool: Callable[[dict], None] | None = None,
                on_token: Callable[[str], None] | None = None, on_reasoning: Callable[[str], None] | None = None,
                on_memory_activity: Callable[[dict], None] | None = None):
        worker = self.executive
        if worker is None or not worker.status().loaded:
            raise LocalModelUnavailable("Local GGUF model is not loaded. Choose the Granite GGUF in Settings and click Load models.")
        user_text = next((str(m.get("content", "")) for m in reversed(recent_messages) if m.get("role") == "user"), "")
        hints = self._automatic_hints(user_text)
        if on_memory_activity:
            for hint in hints[:6]:
                mids = list(hint.get("matched_memory_ids") or [])
                try:
                    on_memory_activity({"mode": "HINT", "document_id": int(hint.get("document_id", -1)),
                                        "memory_id": int(mids[0]) if mids else None,
                                        "label": ",".join(hint.get("match_tokens") or [])})
                except Exception:
                    pass
        transcript = [{"role": "system", "content": self._system_prompt(user_text, hints)}]
        transcript += self._bounded_recent(recent_messages)

        rt = self.config.get("runtime") or {}
        timeout = float(rt.get("executive_timeout_seconds", 300.0))
        max_rounds = max(1, min(6, int(rt.get("tool_rounds", 4) or 4)))
        generation = dict(self._model_cfg("executive"))
        generation = {
            "enable_thinking": bool(generation.get("enable_thinking", True)),
            "max_tokens": int(generation.get("max_tokens", 2048) or 2048),
            "thinking_max_chars": int(generation.get("thinking_max_chars", 5000) or 5000),
        }
        # When neural hints are present, leave Granite enough reasoning budget to
        # decide for itself whether they matter. Hint presence never forces a tool call.
        if hints:
            generation["reasoning_effort"] = "high"
        # Granite 4.2 natively accepts OpenAI-style tool schemas through its chat
        # template. Do not put a home-grown metadata/template-sniffing permission
        # gate in front of the model: always supply the runtime tools to the
        # Executive and let the model/template use them as designed.
        native_tools = True
        available = list(NATIVE_TOOLS.values())
        tool_events = []
        memory_notes = []
        thinking_parts = []

        for _ in range(max_rounds):
            response = worker.chat(
                transcript, timeout=timeout, stream=bool(on_token or on_reasoning), tools=available,
                on_token=on_token, on_reasoning=on_reasoning, generation=generation,
            )
            if response.get("thinking"):
                thinking_parts.append(str(response.get("thinking")))
            calls, answer, preamble = self._native_tool_calls(response)
            if preamble:
                thinking_parts.append(preamble)
            if not calls:
                return {
                    "text": answer.strip(), "thinking": "\n\n".join(thinking_parts).strip(),
                    "tools": tool_events, "memory_notes": memory_notes, "hints": hints,
                    "native_tools": native_tools,
                }
            call = calls[0]
            name = str(call.get("name", ""))
            if name not in NATIVE_TOOLS:
                transcript.append({"role": "assistant", "content": ""})
                transcript.append({"role": "user", "content": f"Private runtime notice: unsupported tool '{name}'. Answer without it."})
                available = []
                continue
            structured = self._tool_message_call(call)
            result, event, note = self._execute_tool(name, dict(call.get("arguments") or {}), user_text=user_text)
            tool_events.append(event)
            if note:
                memory_notes.append({"tool": name, "text": note})
            if on_tool:
                on_tool(event)
            transcript.append({"role": "assistant", "content": "", "tool_calls": [structured]})
            transcript.append({"role": "tool", "tool_call_id": structured["id"], "name": name,
                               "content": self._wire_tool_result(name, result)})

        # Bounded final answer pass after tool budget; no more tools.
        transcript.insert(1, {"role": "system", "content": "Tool budget is exhausted. Answer now from existing evidence; do not request another tool."})
        response = worker.chat(transcript, timeout=timeout, stream=bool(on_token or on_reasoning), tools=[],
                               on_token=on_token, on_reasoning=on_reasoning,
                               generation={**generation, "enable_thinking": False})
        calls, answer, _ = self._native_tool_calls(response)
        return {"text": answer.strip() or str(response.get("result", "")).strip(), "thinking": "\n\n".join(thinking_parts).strip(),
                "tools": tool_events, "memory_notes": memory_notes, "hints": hints, "native_tools": native_tools}
