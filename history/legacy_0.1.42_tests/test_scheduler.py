import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from core.config import DEFAULT_CONFIG
from core.memory import CognitiveRMEM, MemoryHit
from core.scheduler import CognitiveScheduler, _parse_json
from core.schemas import AppraisalPacket, MemoryPacket, ObserverPacket


class FakeWorker:
    def __init__(self, role, tool_mode="none", delay=0.0):
        self.role = role
        self.tool_mode = tool_mode
        self.delay = float(delay)
        self.tool_decisions = 0
        self.final_messages = []
        self.executive_calls = []
        self.generations = []

    def token_count(self, text, timeout=20):
        return max(1, len(text) // 4)

    def cancel(self):
        pass

    def chat(self, messages, timeout, json_schema=None, stream=False, on_token=None, on_reasoning=None, generation=None, tools=None):
        if self.delay > 0 and self.role in ("qwen1", "qwen2"):
            time.sleep(self.delay)
        # Reproduce the strict Qwen/llama.cpp contract that exposed the 0.1.3 bug:
        # a system message is allowed exactly once and only at index zero.
        if self.role == "executive":
            self.executive_calls.append([dict(m) for m in messages])
            self.generations.append(dict(generation or {}))
            system_positions = [i for i, m in enumerate(messages) if m.get("role") == "system"]
            if system_positions != [0]:
                raise RuntimeError("System message must be at the beginning.")
        system = messages[0]["content"] if messages else ""
        if self.role == "qwen1":
            obj = {
                "focus": "cognitive architecture", "intent": "continue build",
                "entities": ["Executive", "RMEM"], "goals": ["build working system"],
                "open_loops": [], "memory_queries": ["native cognitive architecture"],
                "tone": "direct", "user_mood": "engaged", "hostility": 0.0,
                "insult": 0.0, "praise": 0.0, "frustration": 0.1, "urgency": 0.2,
                "directed_at_system": False, "correction_detected": False,
                "salience": 0.8, "confidence": 0.9,
            }
        elif self.role == "qwen2" and "post-turn" in system.lower():
            obj = {
                "summary": "The user requested a working cognitive system and the Executive answered.",
                "what_changed": ["A working turn was completed"],
                "lesson": "Preserve direct in-process cognitive coordination.",
                "entities": ["Executive", "RMEM"], "connections": ["cognitive scheduler"],
                "salience": 0.8, "confidence": 0.9,
            }
        elif self.role == "qwen2":
            payload = json.loads(messages[1]["content"])
            ids = [x["record_id"] for x in payload.get("candidate_memories", [])[:2]]
            obj = {
                "summary": "Prior cognitive architecture is relevant.",
                "relevant_record_ids": ids, "connections": ["native coordination"],
                "contradictions": [], "unresolved": [], "confidence": 0.85,
            }
        elif self.role == "executive" and json_schema is not None:
            self.tool_decisions += 1
            if self.tool_mode == "search" and self.tool_decisions == 1:
                obj = {
                    "action": "memory_search", "query": "cobalt raven",
                    "record_id": 0, "url": "", "limit": 5,
                    "reason": "Earlier experience is materially relevant.",
                }
            elif self.tool_mode == "get" and self.tool_decisions == 1:
                obj = {
                    "action": "memory_get", "query": "", "record_id": 1,
                    "url": "", "limit": 1, "reason": "Need exact surfaced record.",
                }
            elif self.tool_mode in {"web", "web_bad_final"} and self.tool_decisions == 1:
                obj = {
                    "action": "web_search", "query": "Granite 4.2 3B latest model card",
                    "record_id": 0, "url": "", "limit": 5,
                    "reason": "The user explicitly asked for current public information.",
                }
            elif self.tool_mode in {"web", "web_bad_final"} and self.tool_decisions == 2:
                obj = {
                    "action": "web_fetch", "query": "", "record_id": 0,
                    "url": "https://example.com/granite", "limit": 1,
                    "reason": "Fetch the surfaced result for primary-page detail.",
                }
            elif self.tool_mode == "direct_fetch" and self.tool_decisions == 1:
                obj = {
                    "action": "web_fetch", "query": "", "record_id": 0,
                    "url": "https://example.com/page", "limit": 1,
                    "reason": "The user supplied this URL directly.",
                }
            else:
                obj = {"action": "none", "query": "", "record_id": 0, "url": "", "limit": 0, "reason": "Context is sufficient."}
        elif self.role == "executive":
            self.final_messages = list(messages)
            if self.tool_mode == "state_denial":
                if stream:
                    text = "I'm an AI, so I don't have feelings or emotions."
                    thinking = "I should answer with the usual AI disclaimer."
                    if on_reasoning:
                        on_reasoning(thinking)
                    if on_token:
                        on_token(text)
                    return {"result": text, "thinking": thinking, "elapsed": 0.01}
                return {
                    "result": "Functionally, I feel generally okay and calm right now—mildly curious and reasonably confident. That's my maintained software affective state, not a claim of biological sensation.",
                    "thinking": "", "elapsed": 0.01,
                }
            if self.tool_mode == "reasoning_only_stream_recovery" and stream and not bool((generation or {}).get("enable_thinking") is False):
                thinking = "I have reasoned through the request and now need to answer it."
                if on_reasoning:
                    for tok in ["I have reasoned ", "through the request ", "and now need to answer it."]:
                        on_reasoning(tok)
                return {"result": "", "thinking": thinking, "elapsed": 0.01}
            if self.tool_mode == "reasoning_only_stream_recovery" and bool((generation or {}).get("enable_thinking") is False):
                text = "This recovered answer streams live instead of appearing all at once."
                if stream and on_token:
                    for tok in ["This recovered answer ", "streams live ", "instead of appearing all at once."]:
                        on_token(tok)
                return {"result": text, "thinking": "", "elapsed": 0.01}
            if self.tool_mode == "web_planner_none_reasoning_only" and stream and not bool((generation or {}).get("enable_thinking") is False):
                thinking = (
                    "I need current weather, so I should use web_search and then web_fetch. "
                    "I will issue the web search now."
                )
                if on_reasoning:
                    on_reasoning(thinking)
                return {"result": "", "thinking": thinking, "elapsed": 0.01}
            if self.tool_mode == "web_planner_none_reasoning_only" and bool((generation or {}).get("enable_thinking") is False):
                text = "Manchester's current conditions are available from the fetched weather source [1]."
                if stream and on_token:
                    for tok in ["Manchester's current ", "conditions are available ", "from the fetched weather source [1]."]:
                        on_token(tok)
                return {"result": text, "thinking": "", "elapsed": 0.01}
            if self.tool_mode == "web_planner_none_reasoning_only" and not stream:
                return {
                    "result": "Manchester's current conditions are available from the fetched weather source [1].",
                    "thinking": "", "elapsed": 0.01,
                }
            if self.tool_mode == "web_bad_final" and stream:
                text = "I'll search the web now. web_search(current Granite release)"
                thinking = "I have evidence already but am mistakenly narrating tools."
                if on_reasoning:
                    on_reasoning(thinking)
                if on_token:
                    on_token(text)
                return {"result": text, "thinking": thinking, "elapsed": 0.01}
            if self.tool_mode == "web_bad_final" and not stream:
                return {
                    "result": "Granite 4.2 is documented on the primary model page [1].",
                    "thinking": "", "elapsed": 0.01,
                }
            if self.tool_mode == "length_cutoff":
                if messages and str(messages[0].get("content", "")).startswith("CONTINUE EXISTING ANSWER"):
                    text = " and then continues to completion with the remaining details."
                    if stream and on_token:
                        for tok in [" and then continues ", "to completion ", "with the remaining details."]:
                            on_token(tok)
                    return {"result": text, "thinking": "", "elapsed": 0.01, "finish_reason": "stop"}
                text = "The answer begins with useful context but is cut off mid-sentence"
                thinking = "Brief reasoning before the answer."
                if stream:
                    if on_reasoning:
                        on_reasoning(thinking)
                    if on_token:
                        for tok in ["The answer begins ", "with useful context ", "but is cut off mid-sentence"]:
                            on_token(tok)
                return {"result": text, "thinking": thinking, "elapsed": 0.01, "finish_reason": "length"}
            text = "This is the Executive response."
            thinking = "I should integrate the cognitive context before answering."
            if stream:
                if on_reasoning:
                    for tok in ["I should integrate ", "the cognitive context ", "before answering."]:
                        on_reasoning(tok)
                if on_token:
                    for tok in ["This is ", "the Executive ", "response."]:
                        on_token(tok)
            return {"result": text, "thinking": thinking, "elapsed": 0.01}
        else:
            raise RuntimeError(self.role)
        return {"result": json.dumps(obj), "elapsed": 0.01}


class FakePool:
    def __init__(self, tool_mode="none", qwen_delay=0.0):
        self.workers = {
            r: FakeWorker(r, tool_mode if r == "executive" else "none", qwen_delay if r in ("qwen1", "qwen2") else 0.0)
            for r in ("executive", "qwen1", "qwen2")
        }

    def get(self, role):
        return self.workers[role]

    def cancel_all(self):
        pass


class NativeToolWorker(FakeWorker):
    def __init__(self):
        super().__init__("executive")
        self.model_info = {"native_tools": True}
        self.native_transcripts = []
        self.native_tool_sets = []
        self.native_generations = []

    def chat(self, messages, timeout, json_schema=None, stream=False, on_token=None,
             on_reasoning=None, generation=None, tools=None):
        self.native_transcripts.append([dict(m) for m in messages])
        self.native_tool_sets.append(list(tools or []))
        self.native_generations.append(dict(generation or {}))
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        if not tool_messages:
            text = (
                "<tool_call>\n<function=web_search>\n"
                "<parameter=query>Granite 4.2 3B official model card</parameter>\n"
                "<parameter=limit>5</parameter>\n</function>\n</tool_call>"
            )
            if stream and on_token:
                for chunk in ["<tool", "_call>\n<function=web_search>\n", text.split("<function=web_search>\n", 1)[1]]:
                    on_token(chunk)
            return {"result": text, "thinking": "I need an official source before answering."}
        if len(tool_messages) == 1:
            text = (
                "<tool_call>\n<function=web_fetch>\n"
                "<parameter=url>https://example.com/granite</parameter>\n"
                "</function>\n</tool_call>"
            )
            if stream and on_token:
                for chunk in ["<tool_call>", text[len("<tool_call>"):]]:
                    on_token(chunk)
            return {"result": text, "thinking": "The search surfaced the primary page, so I should read it."}
        text = "Granite 4.2 is documented on the primary model page [1]."
        if stream and on_token:
            for chunk in ["Granite 4.2 ", "is documented on ", "the primary model page [1]."]:
                on_token(chunk)
        return {"result": text, "thinking": "I have enough evidence to answer."}


class NativeMemoryReferenceWorker(NativeToolWorker):
    def __init__(self):
        super().__init__()
        self.search_packet = None
        self.get_packet = None

    def chat(self, messages, timeout, json_schema=None, stream=False, on_token=None,
             on_reasoning=None, generation=None, tools=None):
        self.native_transcripts.append([dict(m) for m in messages])
        self.native_tool_sets.append(list(tools or []))
        self.native_generations.append(dict(generation or {}))
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        if not tool_messages:
            return {"result": (
                "<tool_call>\n<function=memory_search>\n"
                "<parameter=query>cobalt raven architecture</parameter>\n"
                "<parameter=limit>5</parameter>\n</function>\n</tool_call>"
            ), "thinking": "The historical reference may matter, so I should search memory."}
        if len(tool_messages) == 1:
            self.search_packet = json.loads(tool_messages[-1]["content"])
            rid = int(self.search_packet["results"][0]["record_id"])
            return {"result": (
                "<tool_call>\n<function=memory_get>\n"
                f"<parameter=record_id>{rid}</parameter>\n"
                "</function>\n</tool_call>"
            ), "thinking": "The descriptor is only a hint; I need the exact record."}
        self.get_packet = json.loads(tool_messages[-1]["content"])
        text = "I checked the exact historical record before answering."
        if stream and on_token:
            for chunk in ["I checked the exact ", "historical record ", "before answering."]:
                on_token(chunk)
        return {"result": text, "thinking": "The fetched memory is sufficient."}


class NativeMemoryReferencePool(FakePool):
    def __init__(self):
        super().__init__()
        self.workers["executive"] = NativeMemoryReferenceWorker()


class NativeUnsurfacedMemoryWorker(NativeToolWorker):
    def __init__(self):
        super().__init__()
        self.denied_packet = None

    def chat(self, messages, timeout, json_schema=None, stream=False, on_token=None,
             on_reasoning=None, generation=None, tools=None):
        self.native_transcripts.append([dict(m) for m in messages])
        self.native_tool_sets.append(list(tools or []))
        self.native_generations.append(dict(generation or {}))
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        if not tool_messages:
            return {"result": (
                "<tool_call>\n<function=memory_get>\n"
                "<parameter=record_id>999</parameter>\n"
                "</function>\n</tool_call>"
            ), "thinking": "I will try an ID that was never surfaced."}
        self.denied_packet = json.loads(tool_messages[-1]["content"])
        text = "That unsurfaced memory ID was correctly denied."
        if stream and on_token:
            on_token(text)
        return {"result": text, "thinking": "The runtime rejected the unauthorized memory read."}


class NativeUnsurfacedMemoryPool(FakePool):
    def __init__(self):
        super().__init__()
        self.workers["executive"] = NativeUnsurfacedMemoryWorker()


class NativeToolPool(FakePool):
    def __init__(self):
        super().__init__()
        self.workers["executive"] = NativeToolWorker()


class NativeNoCallWorker(NativeToolWorker):
    def chat(self, messages, timeout, json_schema=None, stream=False, on_token=None,
             on_reasoning=None, generation=None, tools=None):
        self.native_transcripts.append([dict(m) for m in messages])
        if not any(m.get("role") == "tool" for m in messages):
            return {"result": "I think the weather is probably mild.", "thinking": ""}
        return {"result": "Manchester is currently 71.4°F according to the retrieved source [1].", "thinking": ""}


class NativeNoCallPool(FakePool):
    def __init__(self):
        super().__init__()
        self.workers["executive"] = NativeNoCallWorker()


class NativeIdentityLeakWorker(NativeToolWorker):
    def chat(self, messages, timeout, json_schema=None, stream=False, on_token=None,
             on_reasoning=None, generation=None, tools=None):
        self.native_transcripts.append([dict(m) for m in messages])
        self.native_tool_sets.append(list(tools or []))
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        if not tool_messages:
            return {"result": (
                "<tool_call>\n<function=current_weather>\n"
                "<parameter=location>Manchester, New Hampshire</parameter>\n"
                "</function>\n</tool_call>"
            ), "thinking": "I should use the supplied current-weather tool."}
        system = str(messages[0].get("content", ""))
        if "Always finish the turn with user-facing assistant content" not in system:
            return {"result": (
                "The user says the prior temperature is wrong. We need to respond appropriately.\n\n"
                "As ChatGPT we should use [web_search] and [web_fetch], then provide the answer."
            ), "thinking": ""}
        return {"result": "Manchester's current conditions are taken from the retrieved source [1].", "thinking": "I have the live measurement and can answer normally."}


class NativeIdentityLeakPool(FakePool):
    def __init__(self):
        super().__init__()
        self.workers["executive"] = NativeIdentityLeakWorker()


class FakeBrowser:
    def __init__(self):
        self.enabled = True
        self.search_calls = []
        self.fetch_calls = []
        self.weather_calls = []

    def search(self, query, *, limit=8):
        self.search_calls.append((query, limit))
        return {
            "ok": True, "query": query, "provider": "fake",
            "results": [{
                "title": "Granite 4.2 model card",
                "url": "https://example.com/granite",
                "snippet": "Granite 4.2 3B is a public model release.",
            }],
        }

    def fetch(self, url, *, max_chars=None):
        self.fetch_calls.append(url)
        return {
            "ok": True, "url": url, "title": "Granite 4.2 model card",
            "content_type": "text/html",
            "text": "Primary page detail about Granite 4.2 3B.",
            "truncated": False,
        }

    def current_weather(self, location):
        self.weather_calls.append(location)
        return {
            "ok": True, "location": location,
            "resolved_location": "Manchester, New Hampshire, United States",
            "latitude": 42.9956, "longitude": -71.4548, "timezone": "America/New_York",
            "observed_at": "2026-09-15T18:00", "temperature_f": 71.4,
            "apparent_temperature_f": 70.1, "relative_humidity_percent": 48,
            "weather_code": 1, "condition": "mainly clear", "wind_speed_mph": 6.2, "source": "Open-Meteo",
            "source_url": "https://api.open-meteo.com/v1/forecast?latitude=42.9956&longitude=-71.4548",
        }


class NativeStaticWorker(NativeToolWorker):
    def chat(self, messages, timeout, json_schema=None, stream=False, on_token=None,
             on_reasoning=None, generation=None, tools=None):
        self.native_transcripts.append([dict(m) for m in messages])
        self.native_tool_sets.append(list(tools or []))
        return {"result": "Paris is the capital of France.", "thinking": "No tool is needed for stable common knowledge."}


class NativeStaticPool(FakePool):
    def __init__(self):
        super().__init__()
        self.workers["executive"] = NativeStaticWorker()


class NativeLiveThinkingWorker(NativeToolWorker):
    def __init__(self):
        super().__init__()
        self.stream_flags = []

    def chat(self, messages, timeout, json_schema=None, stream=False, on_token=None,
             on_reasoning=None, generation=None, tools=None):
        self.native_transcripts.append([dict(m) for m in messages])
        self.native_tool_sets.append(list(tools or []))
        self.stream_flags.append(bool(stream))
        chunks = ["First constraint. ", "Second constraint. ", "The order is forced."]
        if stream and on_reasoning:
            for chunk in chunks:
                on_reasoning(chunk)
        if stream and on_token:
            for chunk in ["D, ", "B, ", "E, ", "C, ", "A"]:
                on_token(chunk)
        return {
            "result": "D, B, E, C, A",
            "thinking": "".join(chunks),
            "elapsed": 0.01,
        }


class NativeLiveThinkingPool(FakePool):
    def __init__(self):
        super().__init__()
        self.workers["executive"] = NativeLiveThinkingWorker()




class NativeThinkingOnlyWorker(NativeToolWorker):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def chat(self, messages, timeout, json_schema=None, stream=False, on_token=None,
             on_reasoning=None, generation=None, tools=None):
        self.calls += 1
        self.native_transcripts.append([dict(m) for m in messages])
        self.native_tool_sets.append(list(tools or []))
        self.native_generations.append(dict(generation or {}))
        thought = "I have reached the answer, but this deliberately malformed model turn never exits reasoning."
        if stream and on_reasoning:
            on_reasoning(thought)
        return {"result": "", "thinking": thought, "elapsed": 0.01}


class NativeThinkingOnlyPool(FakePool):
    def __init__(self):
        super().__init__()
        self.workers["executive"] = NativeThinkingOnlyWorker()

class NativeFisherCatsWorker(NativeToolWorker):
    def chat(self, messages, timeout, json_schema=None, stream=False, on_token=None,
             on_reasoning=None, generation=None, tools=None):
        self.native_transcripts.append([dict(m) for m in messages])
        self.native_tool_sets.append(list(tools or []))
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        if not tool_messages:
            names = {str(t.get("function", {}).get("name", "")) for t in (tools or [])}
            if "web_search" not in names:
                return {"result": "I don't have access to real-time databases.", "thinking": ""}
            return {"result": (
                "<tool_call>\n<function=web_search>\n"
                "<parameter=query>New Hampshire Fisher Cats latest game result</parameter>\n"
                "<parameter=limit>5</parameter>\n</function>\n</tool_call>"
            ), "thinking": "I should check the current game result with the supplied web tool."}
        return {"result": "The Fisher Cats won their latest game, 5–2, according to the search result [1].", "thinking": ""}


class NativeFisherCatsPool(FakePool):
    def __init__(self):
        super().__init__()
        self.workers["executive"] = NativeFisherCatsWorker()




class NativeBudgetExhaustWorker(NativeToolWorker):
    """Requests a fourth tool after the runtime has executed its configured three."""
    def __init__(self, ignore_notice_once=False):
        super().__init__()
        self.ignore_notice_once = bool(ignore_notice_once)
        self.calls = 0

    def chat(self, messages, timeout, json_schema=None, stream=False, on_token=None,
             on_reasoning=None, generation=None, tools=None):
        self.calls += 1
        self.native_transcripts.append([dict(m) for m in messages])
        self.native_tool_sets.append(list(tools or []))
        self.native_generations.append(dict(generation or {}))
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        budget_seen = any("Tool action budget exhausted" in str(m.get("content", "")) for m in tool_messages)
        if budget_seen and (not self.ignore_notice_once or not bool((generation or {}).get("enable_thinking", True))):
            return {
                "result": "Using the evidence already gathered, the Fisher Cats won the latest completed game.",
                "thinking": "",
            }
        thought = f"I still want one more source after {len(tool_messages)} tool results."
        if stream and on_reasoning:
            on_reasoning(thought)
        return {
            "result": (
                "<tool_call>\n<function=web_search>\n"
                f"<parameter=query>Fisher Cats verification pass {self.calls}</parameter>\n"
                "<parameter=limit>5</parameter>\n</function>\n</tool_call>"
            ),
            "thinking": thought,
        }


class NativeBudgetExhaustPool(FakePool):
    def __init__(self, ignore_notice_once=False):
        super().__init__()
        self.workers["executive"] = NativeBudgetExhaustWorker(ignore_notice_once=ignore_notice_once)


class FakeFisherCatsBrowser(FakeBrowser):
    def search(self, query, *, limit=8):
        self.search_calls.append((query, limit))
        return {
            "ok": True, "query": query, "provider": "fake",
            "results": [{
                "title": "Fisher Cats Game Recap",
                "url": "https://example.com/fisher-cats",
                "snippet": "The New Hampshire Fisher Cats won 5-2 in their latest completed game.",
            }],
        }


class SchedulerTests(unittest.TestCase):
    def test_json_parser_handles_fence(self):
        self.assertEqual(_parse_json('```json\n{"a":1}\n```')["a"], 1)

    def _run(self, tool_mode="none", personality="TEST PERSONALITY: concise but warm.", scheduling=None, qwen_delay=0.0):
        td = tempfile.TemporaryDirectory()
        cfg = json.loads(json.dumps(DEFAULT_CONFIG))
        cfg["memory"]["path"] = str(Path(td.name) / "experience.rmem")
        cfg["identity"]["personality_prompt"] = personality
        cfg["identity"]["name"] = "Nova"
        if scheduling is not None:
            cfg["runtime"]["cognitive_scheduling"] = scheduling
        elif tool_mode != "none":
            cfg["runtime"]["cognitive_scheduling"] = "deep"
        mem = CognitiveRMEM(cfg["memory"]["path"])
        mem.append("episode", {"summary": "Earlier native cognitive architecture decision involving cobalt raven", "salience": 0.8})
        state_path = Path(td.name) / "state.json"
        log_dir = Path(td.name) / "logs"
        pool = FakePool(tool_mode, qwen_delay=qwen_delay)
        with patch("core.scheduler.STATE_PATH", state_path), patch("core.scheduler.LOGS_DIR", log_dir):
            sched = CognitiveScheduler(cfg, pool, mem)
            tokens = []
            events = []
            thoughts = []
            result = sched.run_turn(
                "Build the cognitive runtime", on_token=tokens.append, on_thought=thoughts.append,
                on_event=lambda n, d: events.append((n, d))
            )
            self.assertTrue(sched.wait_for_background(timeout=8.0))
            sched.shutdown()
        return td, cfg, mem, pool, result, tokens, thoughts, events

    def test_memory_reference_ids_and_descriptors_are_deterministic_and_validated(self):
        hit = MemoryHit(
            record_id=42, record_type="turn", timestamp=1234.5, score=2.0, salience=0.8,
            text="fallback",
            payload={
                "record_id": 42, "record_type": "turn", "role": "user",
                "content": "We discussed the cobalt raven architecture and its memory routing.\nThis is historical.",
            },
        )
        ref = CognitiveScheduler._memory_reference(hit)
        self.assertEqual(ref["record_id"], 42)
        self.assertEqual(ref["record_type"], "turn")
        self.assertEqual(ref["timestamp"], 1234.5)
        self.assertEqual(
            ref["descriptor"],
            "historical user turn: We discussed the cobalt raven architecture and its memory routing. This is historical.",
        )
        bad = MemoryHit(
            record_id=42, record_type="turn", timestamp=1234.5, score=2.0, salience=0.8,
            text="bad", payload={"record_id": 99, "record_type": "turn", "content": "wrong provenance"},
        )
        self.assertIsNone(CognitiveScheduler._memory_reference(bad))

    def test_native_memory_is_reference_first_then_exact_get(self):
        td = tempfile.TemporaryDirectory()
        try:
            cfg = json.loads(json.dumps(DEFAULT_CONFIG))
            cfg["runtime"]["cognitive_scheduling"] = "fast"
            cfg["memory"]["path"] = str(Path(td.name) / "experience.rmem")
            mem = CognitiveRMEM(cfg["memory"]["path"])
            long_summary = (
                "Cobalt raven architecture chose reference-first RMEM retrieval and explicit memory_get. "
                + "historical-detail " * 20
                + "SECRET_TRANSCRIPT_TAIL"
            )
            rid = mem.append("episode", {"summary": long_summary, "salience": 0.9})
            pool = NativeMemoryReferencePool()
            with patch("core.scheduler.STATE_PATH", Path(td.name) / "state.json"), patch("core.scheduler.LOGS_DIR", Path(td.name) / "logs"):
                sched = CognitiveScheduler(cfg, pool, mem)
                result = sched.run_turn("What did we decide about cobalt raven architecture?")
                self.assertTrue(sched.wait_for_background(timeout=8.0))
                sched.shutdown()

            worker = pool.get("executive")
            self.assertIsNotNone(worker.search_packet)
            refs = worker.search_packet.get("results", [])
            self.assertTrue(refs)
            ref = refs[0]
            self.assertEqual(ref["record_id"], rid)
            self.assertEqual(ref["record_type"], "episode")
            self.assertIn("cobalt raven architecture", ref["descriptor"].lower())
            self.assertNotIn("text", ref)
            self.assertEqual(mem.get(ref["record_id"])["record_id"], ref["record_id"])

            # Before memory_get, old transcript/body content must not be stuffed into
            # the Executive's initial system context. The tail only appears after the
            # model explicitly fetches the surfaced ID.
            initial_system = worker.native_transcripts[0][0]["content"]
            self.assertIn("memory_references", initial_system)
            self.assertIn(str(rid), initial_system)
            self.assertNotIn("SECRET_TRANSCRIPT_TAIL", initial_system)
            self.assertIn("historical index entries", initial_system)

            self.assertIsNotNone(worker.get_packet)
            self.assertTrue(worker.get_packet.get("ok"))
            self.assertEqual(worker.get_packet.get("record_id"), rid)
            self.assertEqual(worker.get_packet["record"]["record_id"], rid)
            self.assertIn("SECRET_TRANSCRIPT_TAIL", worker.get_packet["record"]["summary"])
            self.assertEqual(result["assistant_text"], "I checked the exact historical record before answering.")
        finally:
            td.cleanup()

    def test_native_memory_get_rejects_unsurfaced_id(self):
        td = tempfile.TemporaryDirectory()
        try:
            cfg = json.loads(json.dumps(DEFAULT_CONFIG))
            cfg["runtime"]["cognitive_scheduling"] = "fast"
            cfg["memory"]["path"] = str(Path(td.name) / "experience.rmem")
            mem = CognitiveRMEM(cfg["memory"]["path"])
            mem.append("episode", {"summary": "A legitimate historical memory", "salience": 0.8})
            pool = NativeUnsurfacedMemoryPool()
            with patch("core.scheduler.STATE_PATH", Path(td.name) / "state.json"), patch("core.scheduler.LOGS_DIR", Path(td.name) / "logs"):
                sched = CognitiveScheduler(cfg, pool, mem)
                result = sched.run_turn("Answer without inventing historical access.")
                self.assertTrue(sched.wait_for_background(timeout=8.0))
                sched.shutdown()
            worker = pool.get("executive")
            self.assertIsNotNone(worker.denied_packet)
            self.assertFalse(worker.denied_packet.get("ok"))
            self.assertEqual(worker.denied_packet.get("record_id"), 999)
            self.assertIn("not previously surfaced", worker.denied_packet.get("error", ""))
            self.assertEqual(result["assistant_text"], "That unsurfaced memory ID was correctly denied.")
        finally:
            td.cleanup()

    def test_complete_cognitive_turn_and_prompt_separation(self):
        td, cfg, mem, pool, result, tokens, thoughts, events = self._run()
        try:
            self.assertEqual(result["assistant_text"], "This is the Executive response.")
            self.assertEqual("".join(tokens), "This is the Executive response.")
            self.assertEqual("".join(thoughts), "I should integrate the cognitive context before answering.")
            self.assertEqual(result["thinking"], "I should integrate the cognitive context before answering.")
            types = mem.stats()["type_counts"]
            self.assertGreaterEqual(types.get("turn", 0), 2)
            self.assertGreaterEqual(types.get("episode", 0), 2)
            self.assertGreaterEqual(types.get("lesson", 0), 1)
            self.assertIn("episode_committed", [x[0] for x in events])
            self.assertTrue(mem.validate()["ok"])
            self.assertEqual(result["scheduling_mode"], "fast")
            self.assertIn("pre_executive", result["trace"]["timings"])
            self.assertIn("background_complete", [x[0] for x in events])
            visible_events = [d for n, d in events if n == "visible_complete"]
            self.assertTrue(visible_events)
            self.assertEqual(visible_events[-1].get("assistant_text"), result["assistant_text"])
            assistant_turns = [r for r in mem.latest(count=20, types=["turn"]) if r.get("role") == "assistant"]
            self.assertTrue(assistant_turns)
            self.assertEqual(assistant_turns[0].get("identity_name"), "Nova")

            final_messages = pool.get("executive").final_messages
            system_messages = [m["content"] for m in final_messages if m["role"] == "system"]
            self.assertEqual(len(system_messages), 1)
            self.assertEqual(final_messages[0]["role"], "system")
            self.assertIn("PERSONALITY", system_messages[0])
            self.assertIn("TEST PERSONALITY", system_messages[0])
            self.assertIn("IDENTITY", system_messages[0])
            self.assertIn('"Nova"', system_messages[0])
            self.assertIn("This is your name, not the user's", system_messages[0])
            self.assertIn("SELF-STATE (private)", system_messages[0])
            self.assertIn("current functional affective state", system_messages[0])
            self.assertIn("RESPONSE DISCIPLINE", system_messages[0])
            self.assertIn("Do not reopen, rehearse, debate, critique, or repeatedly reconsider", system_messages[0])
            final_generation = pool.get("executive").generations[-1]
            self.assertEqual(final_generation.get("reasoning_effort"), "low")
            self.assertEqual(final_generation.get("thinking_max_chars"), cfg["models"]["executive"]["thinking_max_chars"])
            self.assertEqual(final_generation.get("max_tokens"), cfg["models"]["executive"]["max_tokens"])
            final_user = final_messages[-1]["content"]
            self.assertEqual(final_user, "Build the cognitive runtime")
            # Runtime cognition is system-side only. The user-role message must remain
            # exactly the user's text with no private envelope or metadata appended.
            self.assertNotIn("private_context", final_user)
            # Visible Executive private background is telemetry-free. Long-term memory
            # is reference-only: IDs/descriptors may be surfaced, not transcript bodies.
            for leaked in ('persistent_state', 'observer_signals', 'appraisal', 'curiosity', 'trust'):
                self.assertNotIn(leaked, system_messages[0])
            self.assertIn("LONG-TERM MEMORY REFERENCES", system_messages[0])
            # The persisted/editable fields remain separate even though transport uses
            # one system-role message for strict chat-template compatibility.
            self.assertNotIn("TEST PERSONALITY", cfg["identity"]["system_prompt"])
            self.assertEqual(cfg["identity"]["personality_prompt"], "TEST PERSONALITY: concise but warm.")
            for call in pool.get("executive").executive_calls:
                self.assertEqual([i for i,m in enumerate(call) if m["role"] == "system"], [0])
                current_user = call[-1]
                self.assertEqual(current_user["role"], "user")
                self.assertEqual(current_user["content"], "Build the cognitive runtime")
        finally:
            td.cleanup()

    def test_executive_memory_search_tool_is_real_and_hidden(self):
        td, cfg, mem, pool, result, tokens, thoughts, events = self._run(tool_mode="search")
        try:
            calls = result["trace"]["tool_calls"]
            self.assertTrue(any(c.get("action") == "memory_search" and c.get("ok") for c in calls))
            self.assertTrue(any(name == "executive_tool" and data.get("action") == "memory_search" for name, data in events))
            final_messages = pool.get("executive").final_messages
            self.assertEqual([i for i,m in enumerate(final_messages) if m["role"] == "system"], [0])
            self.assertEqual(final_messages[-1]["content"], "Build the cognitive runtime")
            final_system = final_messages[0]["content"]
            self.assertIn("BACKGROUND (private)", final_system)
            self.assertIn("cobalt raven", final_system)
            self.assertIn('"kind":"memory_references"', final_system)
            self.assertIn('"descriptor"', final_system)
            self.assertNotIn('"text":"Earlier native cognitive architecture decision involving cobalt raven"', final_system)
            self.assertNotIn("PRIVATE TOOL RESULT", final_system)
            self.assertNotIn("PRIVATE TOOL RESULT", result["assistant_text"])
        finally:
            td.cleanup()

    def test_executive_memory_get_only_reads_surfaced_record(self):
        td, cfg, mem, pool, result, tokens, thoughts, events = self._run(tool_mode="get")
        try:
            calls = result["trace"]["tool_calls"]
            get_calls = [c for c in calls if c.get("action") == "memory_get"]
            self.assertEqual(len(get_calls), 1)
            self.assertTrue(get_calls[0]["ok"])
            final_messages = pool.get("executive").final_messages
            self.assertEqual([i for i,m in enumerate(final_messages) if m["role"] == "system"], [0])
            self.assertEqual(final_messages[-1]["content"], "Build the cognitive runtime")
            final_system = final_messages[0]["content"]
            self.assertIn('"kind":"memory_record"', final_system)
            self.assertNotIn('"identity_name"', final_system)
        finally:
            td.cleanup()

    def test_tool_narration_is_hidden_repaired_and_numeric_source_is_linked(self):
        td = tempfile.TemporaryDirectory()
        try:
            cfg = json.loads(json.dumps(DEFAULT_CONFIG))
            cfg["memory"]["path"] = str(Path(td.name) / "experience.rmem")
            cfg["runtime"]["cognitive_scheduling"] = "fast"
            cfg["identity"]["name"] = "Nova"
            mem = CognitiveRMEM(cfg["memory"]["path"])
            pool = FakePool(tool_mode="web_bad_final")
            state_path = Path(td.name) / "state.json"
            log_dir = Path(td.name) / "logs"
            with patch("core.scheduler.STATE_PATH", state_path), patch("core.scheduler.LOGS_DIR", log_dir):
                sched = CognitiveScheduler(cfg, pool, mem)
                sched.browser = FakeBrowser()
                tokens, thoughts = [], []
                result = sched.run_turn(
                    "Search the web for the latest Granite 4.2 model page",
                    on_token=tokens.append, on_thought=thoughts.append,
                )
                self.assertTrue(sched.wait_for_background(timeout=8.0))
                sched.shutdown()
            visible = "".join(tokens)
            self.assertNotIn("I'll search", visible)
            self.assertNotIn("web_search", visible)
            self.assertIn("Granite 4.2 is documented", visible)
            self.assertIn("[1](https://example.com/granite)", visible)
            self.assertEqual(result["assistant_text"], visible)
            self.assertIn("Draft/tool narration redirected", "".join(thoughts))
        finally:
            td.cleanup()

    def test_reasoning_only_recovery_streams_final_answer_live(self):
        td = tempfile.TemporaryDirectory()
        try:
            cfg = json.loads(json.dumps(DEFAULT_CONFIG))
            cfg["memory"]["path"] = str(Path(td.name) / "experience.rmem")
            cfg["runtime"]["cognitive_scheduling"] = "fast"
            mem = CognitiveRMEM(cfg["memory"]["path"])
            pool = FakePool(tool_mode="reasoning_only_stream_recovery")
            with patch("core.scheduler.STATE_PATH", Path(td.name) / "state.json"), patch("core.scheduler.LOGS_DIR", Path(td.name) / "logs"):
                sched = CognitiveScheduler(cfg, pool, mem)
                tokens, thoughts = [], []
                result = sched.run_turn(
                    "Explain the current architecture in one paragraph.",
                    on_token=tokens.append, on_thought=thoughts.append,
                )
                self.assertTrue(sched.wait_for_background(timeout=8.0))
                sched.shutdown()
            visible = "".join(tokens)
            self.assertGreaterEqual(len(tokens), 3, "recovery answer must arrive as multiple live token chunks")
            self.assertEqual(visible, "This recovered answer streams live instead of appearing all at once.")
            self.assertEqual(result["assistant_text"], visible)
            self.assertIn("I have reasoned", "".join(thoughts))
        finally:
            td.cleanup()

    def test_length_cutoff_auto_continues_and_respects_configured_output_budget(self):
        td = tempfile.TemporaryDirectory()
        try:
            cfg = json.loads(json.dumps(DEFAULT_CONFIG))
            cfg["memory"]["path"] = str(Path(td.name) / "experience.rmem")
            cfg["runtime"]["cognitive_scheduling"] = "fast"
            cfg["models"]["executive"]["n_ctx"] = 10000
            cfg["models"]["executive"]["thinking_max_chars"] = 4000
            cfg["models"]["executive"]["max_tokens"] = 6000
            mem = CognitiveRMEM(cfg["memory"]["path"])
            pool = FakePool(tool_mode="length_cutoff")
            with patch("core.scheduler.STATE_PATH", Path(td.name) / "state.json"), patch("core.scheduler.LOGS_DIR", Path(td.name) / "logs"):
                sched = CognitiveScheduler(cfg, pool, mem)
                tokens = []
                result = sched.run_turn("Give me a complete detailed answer.", on_token=tokens.append)
                self.assertTrue(sched.wait_for_background(timeout=8.0))
                sched.shutdown()
            visible = "".join(tokens)
            self.assertIn("continues to completion", visible)
            self.assertEqual(result["assistant_text"], visible)
            generations = pool.get("executive").generations
            self.assertEqual(generations[0].get("thinking_max_chars"), 4000)
            self.assertEqual(generations[0].get("max_tokens"), 6000)
            self.assertGreaterEqual(len(generations), 2)
            self.assertFalse(generations[1].get("enable_thinking", True))
            self.assertGreater(generations[1].get("max_tokens", 0), 1000)
            self.assertTrue(any("finish_reason=length" in err for err in result["trace"]["errors"]))
        finally:
            td.cleanup()

    def test_explicit_web_turn_cannot_end_with_thinking_only_when_planner_does_nothing(self):
        """Regression for 0.1.21 weather failure: planner returned none and Granite only thought about tools."""
        td = tempfile.TemporaryDirectory()
        try:
            cfg = json.loads(json.dumps(DEFAULT_CONFIG))
            cfg["memory"]["path"] = str(Path(td.name) / "experience.rmem")
            cfg["runtime"]["cognitive_scheduling"] = "fast"
            mem = CognitiveRMEM(cfg["memory"]["path"])
            pool = FakePool(tool_mode="web_planner_none_reasoning_only")
            with patch("core.scheduler.STATE_PATH", Path(td.name) / "state.json"), patch("core.scheduler.LOGS_DIR", Path(td.name) / "logs"):
                sched = CognitiveScheduler(cfg, pool, mem)
                fake_browser = FakeBrowser()
                sched.browser = fake_browser
                tokens, thoughts = [], []
                result = sched.run_turn(
                    "What is the current weather in Manchester, New Hampshire?",
                    on_token=tokens.append, on_thought=thoughts.append,
                )
                self.assertTrue(sched.wait_for_background(timeout=8.0))
                sched.shutdown()

            visible = "".join(tokens)
            self.assertTrue(fake_browser.search_calls, "explicit current-web request must cause a real search")
            self.assertTrue(fake_browser.fetch_calls, "fallback search should fetch its first surfaced result")
            self.assertIn("Manchester's current conditions", visible)
            self.assertIn("https://example.com/granite", visible)
            self.assertEqual(result["assistant_text"], visible)
            self.assertIn("I need current weather", "".join(thoughts))
            self.assertTrue(any("reasoning-only executive generation recovered" in e.lower() for e in result["trace"]["errors"]))
            self.assertTrue(any(c.get("round") == "fallback" and c.get("action") == "web_search" for c in result["trace"]["tool_calls"]))
        finally:
            td.cleanup()

    def test_web_intent_gate_only_adds_planner_for_online_or_fresh_requests(self):
        td = tempfile.TemporaryDirectory()
        try:
            cfg = json.loads(json.dumps(DEFAULT_CONFIG))
            cfg["memory"]["path"] = str(Path(td.name) / "experience.rmem")
            mem = CognitiveRMEM(cfg["memory"]["path"])
            with patch("core.scheduler.STATE_PATH", Path(td.name) / "state.json"), patch("core.scheduler.LOGS_DIR", Path(td.name) / "logs"):
                sched = CognitiveScheduler(cfg, FakePool(), mem)
                self.assertFalse(sched._web_tools_needed("Hello, how are you?"))
                self.assertFalse(sched._web_tools_needed("What is our current project design?"))
                self.assertTrue(sched._web_tools_needed("Search the web for the latest Granite release"))
                self.assertTrue(sched._web_tools_needed("What is the current weather in Manchester, New Hampshire?"))
                self.assertTrue(sched._web_tools_needed("What's the weather in Manchester, NH?"))
                self.assertTrue(sched._web_tools_needed("Read https://example.com/page and summarize it"))
                sched.shutdown()
        finally:
            td.cleanup()

    def test_web_search_and_fetch_are_real_hidden_tools_on_fast_path(self):
        td = tempfile.TemporaryDirectory()
        try:
            cfg = json.loads(json.dumps(DEFAULT_CONFIG))
            cfg["runtime"]["cognitive_scheduling"] = "adaptive"
            cfg["memory"]["path"] = str(Path(td.name) / "experience.rmem")
            mem = CognitiveRMEM(cfg["memory"]["path"])
            pool = FakePool(tool_mode="web")
            with patch("core.scheduler.STATE_PATH", Path(td.name) / "state.json"), patch("core.scheduler.LOGS_DIR", Path(td.name) / "logs"):
                sched = CognitiveScheduler(cfg, pool, mem)
                fake_browser = FakeBrowser()
                sched.browser = fake_browser
                result = sched.run_turn("Please search the web for the latest Granite 4.2 3B model card")
                self.assertTrue(sched.wait_for_background(timeout=8.0))
                sched.shutdown()

            self.assertEqual(result["scheduling_mode"], "fast")
            calls = result["trace"]["tool_calls"]
            self.assertTrue(any(c.get("action") == "web_search" and c.get("ok") for c in calls))
            self.assertTrue(any(c.get("action") == "web_fetch" and c.get("ok") for c in calls))
            self.assertEqual(fake_browser.search_calls[0][0], "Granite 4.2 3B latest model card")
            self.assertEqual(fake_browser.fetch_calls, ["https://example.com/granite"])
            final_messages = pool.get("executive").final_messages
            self.assertEqual(final_messages[-1]["content"], "Please search the web for the latest Granite 4.2 3B model card")
            final_system = final_messages[0]["content"]
            self.assertIn("https://example.com/granite", final_system)
            self.assertIn("Primary page detail about Granite 4.2 3B", final_system)
            self.assertNotIn("PRIVATE TOOL RESULT", final_system)
        finally:
            td.cleanup()

    def test_granite_native_tool_conversation_preserves_tool_roles_and_answers(self):
        td = tempfile.TemporaryDirectory()
        try:
            cfg = json.loads(json.dumps(DEFAULT_CONFIG))
            cfg["runtime"]["cognitive_scheduling"] = "fast"
            cfg["memory"]["path"] = str(Path(td.name) / "experience.rmem")
            mem = CognitiveRMEM(cfg["memory"]["path"])
            pool = NativeToolPool()
            with patch("core.scheduler.STATE_PATH", Path(td.name) / "state.json"), patch("core.scheduler.LOGS_DIR", Path(td.name) / "logs"):
                sched = CognitiveScheduler(cfg, pool, mem)
                browser = FakeBrowser()
                sched.browser = browser
                tokens, thoughts = [], []
                result = sched.run_turn(
                    "Search the web for the latest Granite 4.2 3B model card",
                    on_token=tokens.append, on_thought=thoughts.append,
                )
                self.assertTrue(sched.wait_for_background(timeout=8.0))
                sched.shutdown()
            self.assertEqual(browser.search_calls[0][0], "Granite 4.2 3B official model card")
            self.assertEqual(browser.fetch_calls, ["https://example.com/granite"])
            self.assertIn("Granite 4.2 is documented", result["assistant_text"])
            self.assertIn("[1](https://example.com/granite)", result["assistant_text"])
            worker = pool.get("executive")
            last = worker.native_transcripts[-1]
            self.assertEqual([m["role"] for m in last[-4:]], ["assistant", "tool", "assistant", "tool"])
            # Granite's native multi-turn contract preserves its reasoning alongside
            # each assistant tool call, and receives plain JSON in role=tool.
            self.assertIn("<think>", last[-4]["content"])
            self.assertIn("official source", last[-4]["content"])
            self.assertTrue(last[-3]["content"].startswith("{"))
            self.assertNotIn("PRIVATE TOOL RESULT", last[-3]["content"])
            # 0.1.30 secretly constrained native Granite to 1400 chars / 900 tokens.
            # Native turns now honor the Executive settings and keep thinking enabled
            # after tool results rather than forcing a runtime final-answer mode.
            self.assertTrue(all(g.get("enable_thinking") is True for g in worker.native_generations))
            self.assertTrue(all(g.get("thinking_max_chars") == cfg["models"]["executive"]["thinking_max_chars"] for g in worker.native_generations))
            self.assertTrue(all(g.get("max_tokens") == cfg["models"]["executive"]["max_tokens"] for g in worker.native_generations))
            self.assertTrue(all(len(ts) > 0 for ts in worker.native_tool_sets[:2]))
            self.assertFalse(any("<tool_call>" in part for part in tokens))
        finally:
            td.cleanup()

    def test_native_granite_can_decline_tools_when_context_is_sufficient(self):
        td = tempfile.TemporaryDirectory()
        try:
            cfg = json.loads(json.dumps(DEFAULT_CONFIG))
            cfg["runtime"]["cognitive_scheduling"] = "fast"
            cfg["memory"]["path"] = str(Path(td.name) / "experience.rmem")
            mem = CognitiveRMEM(cfg["memory"]["path"])
            pool = NativeStaticPool()
            with patch("core.scheduler.STATE_PATH", Path(td.name) / "state.json"), patch("core.scheduler.LOGS_DIR", Path(td.name) / "logs"):
                sched = CognitiveScheduler(cfg, pool, mem)
                browser = FakeBrowser()
                sched.browser = browser
                result = sched.run_turn("What is the capital of France?")
                self.assertTrue(sched.wait_for_background(timeout=8.0))
                sched.shutdown()
            self.assertEqual(result["assistant_text"], "Paris is the capital of France.")
            self.assertEqual(browser.search_calls, [])
            self.assertEqual(browser.fetch_calls, [])
            self.assertEqual(browser.weather_calls, [])
            names = {t.get("function", {}).get("name") for t in pool.get("executive").native_tool_sets[0]}
            self.assertIn("web_search", names)
            self.assertIn("memory_search", names)
        finally:
            td.cleanup()

    def test_native_tool_capable_granite_streams_reasoning_live_even_when_no_tool_is_used(self):
        """Regression for 0.1.29: native-tool availability must not buffer all Thinking until final."""
        td = tempfile.TemporaryDirectory()
        try:
            cfg = json.loads(json.dumps(DEFAULT_CONFIG))
            cfg["runtime"]["cognitive_scheduling"] = "fast"
            cfg["memory"]["path"] = str(Path(td.name) / "experience.rmem")
            mem = CognitiveRMEM(cfg["memory"]["path"])
            pool = NativeLiveThinkingPool()
            with patch("core.scheduler.STATE_PATH", Path(td.name) / "state.json"), patch("core.scheduler.LOGS_DIR", Path(td.name) / "logs"):
                sched = CognitiveScheduler(cfg, pool, mem)
                sched.browser = FakeBrowser()
                thoughts, tokens = [], []
                result = sched.run_turn(
                    "Solve the ordering puzzle without tools.",
                    on_thought=thoughts.append, on_token=tokens.append,
                )
                self.assertTrue(sched.wait_for_background(timeout=8.0))
                sched.shutdown()
            worker = pool.get("executive")
            self.assertTrue(worker.stream_flags[0], "native tool-capable pass regressed to stream=False")
            self.assertEqual(thoughts, ["First constraint. ", "Second constraint. ", "The order is forced."])
            self.assertEqual(result["assistant_text"], "D, B, E, C, A")
            self.assertEqual(tokens, ["D, ", "B, ", "E, ", "C, ", "A"], "native answer chunks were buffered instead of forwarded live")
            self.assertEqual("".join(tokens), "D, B, E, C, A")
            self.assertEqual(result["thinking"], "First constraint. Second constraint. The order is forced.")
        finally:
            td.cleanup()

    def test_native_thinking_is_never_promoted_or_extracted_as_the_chat_answer(self):
        """The runtime may orchestrate Granite, but Thinking is never converted into answer text."""
        td = tempfile.TemporaryDirectory()
        try:
            cfg = json.loads(json.dumps(DEFAULT_CONFIG))
            cfg["runtime"]["cognitive_scheduling"] = "fast"
            cfg["memory"]["path"] = str(Path(td.name) / "experience.rmem")
            mem = CognitiveRMEM(cfg["memory"]["path"])
            pool = NativeThinkingOnlyPool()
            with patch("core.scheduler.STATE_PATH", Path(td.name) / "state.json"), patch("core.scheduler.LOGS_DIR", Path(td.name) / "logs"):
                sched = CognitiveScheduler(cfg, pool, mem)
                sched.browser = FakeBrowser()
                thoughts, tokens = [], []
                with self.assertRaisesRegex(RuntimeError, "empty response"):
                    sched.run_turn("Give me the answer.", on_thought=thoughts.append, on_token=tokens.append)
                self.assertTrue(sched.wait_for_background(timeout=8.0))
                sched.shutdown()
            self.assertTrue(thoughts)
            self.assertEqual(tokens, [])
            self.assertEqual(pool.get("executive").calls, 1, "runtime silently launched a second answer-generation pass")
        finally:
            td.cleanup()

    def test_native_granite_sees_web_tools_even_when_python_web_heuristic_misses(self):
        """Native Granite, not a keyword router, decides that a live sports result needs web search."""
        td = tempfile.TemporaryDirectory()
        try:
            cfg = json.loads(json.dumps(DEFAULT_CONFIG))
            cfg["runtime"]["cognitive_scheduling"] = "fast"
            cfg["memory"]["path"] = str(Path(td.name) / "experience.rmem")
            mem = CognitiveRMEM(cfg["memory"]["path"])
            pool = NativeFisherCatsPool()
            with patch("core.scheduler.STATE_PATH", Path(td.name) / "state.json"), patch("core.scheduler.LOGS_DIR", Path(td.name) / "logs"):
                sched = CognitiveScheduler(cfg, pool, mem)
                browser = FakeFisherCatsBrowser()
                sched.browser = browser
                self.assertFalse(sched._web_tools_needed("Who won the last Fisher Cats game?"))
                result = sched.run_turn("Who won the last Fisher Cats game?")
                self.assertTrue(sched.wait_for_background(timeout=8.0))
                sched.shutdown()
            self.assertEqual(browser.search_calls[0][0], "New Hampshire Fisher Cats latest game result")
            self.assertIn("Fisher Cats won", result["assistant_text"])
            self.assertNotIn("don't have access to real-time", result["assistant_text"].lower())
            calls = result["trace"]["tool_calls"]
            self.assertTrue(any(c.get("action") == "web_search" and c.get("ok") for c in calls))
            first_tools = pool.get("executive").native_tool_sets[0]
            names = {t.get("function", {}).get("name") for t in first_tools}
            self.assertIn("web_search", names)
            self.assertIn("web_fetch", names)
            self.assertIn("current_weather", names)
            first_system = pool.get("executive").native_transcripts[0][0]["content"]
            self.assertIn("tools supplied with this turn are real runtime capabilities", first_system)
            self.assertIn("do not claim that you lack real-time", first_system)
        finally:
            td.cleanup()

    def test_native_tool_budget_exhaustion_returns_to_granite_for_a_final_answer(self):
        """Regression: exactly N verified tools must not end the turn with an empty answer."""
        td = tempfile.TemporaryDirectory()
        try:
            cfg = json.loads(json.dumps(DEFAULT_CONFIG))
            cfg["runtime"]["cognitive_scheduling"] = "fast"
            cfg["runtime"]["executive_tool_rounds"] = 3
            cfg["memory"]["path"] = str(Path(td.name) / "experience.rmem")
            mem = CognitiveRMEM(cfg["memory"]["path"])
            pool = NativeBudgetExhaustPool()
            with patch("core.scheduler.STATE_PATH", Path(td.name) / "state.json"), patch("core.scheduler.LOGS_DIR", Path(td.name) / "logs"):
                sched = CognitiveScheduler(cfg, pool, mem)
                browser = FakeBrowser()
                sched.browser = browser
                result = sched.run_turn("Who won the last Fisher Cats game?")
                self.assertTrue(sched.wait_for_background(timeout=8.0))
                sched.shutdown()
            self.assertIn("Fisher Cats won", result["assistant_text"])
            self.assertEqual(len(browser.search_calls), 3, "over-budget tool request was actually executed")
            worker = pool.get("executive")
            self.assertEqual([len(x) for x in worker.native_tool_sets[:5]], [5, 5, 5, 0, 0])
            self.assertTrue(any(c.get("round") == "budget" and not c.get("ok") for c in result["trace"]["tool_calls"]))
            final_transcript = worker.native_transcripts[-1]
            self.assertTrue(any("Tool action budget exhausted" in str(m.get("content", "")) for m in final_transcript if m.get("role") == "tool"))
        finally:
            td.cleanup()

    def test_native_tool_budget_exhaustion_has_bounded_final_answer_retry(self):
        """If Granite asks for another tool even after the budget notice, retry Granite once without thinking."""
        td = tempfile.TemporaryDirectory()
        try:
            cfg = json.loads(json.dumps(DEFAULT_CONFIG))
            cfg["runtime"]["cognitive_scheduling"] = "fast"
            cfg["runtime"]["executive_tool_rounds"] = 3
            cfg["memory"]["path"] = str(Path(td.name) / "experience.rmem")
            mem = CognitiveRMEM(cfg["memory"]["path"])
            pool = NativeBudgetExhaustPool(ignore_notice_once=True)
            with patch("core.scheduler.STATE_PATH", Path(td.name) / "state.json"), patch("core.scheduler.LOGS_DIR", Path(td.name) / "logs"):
                sched = CognitiveScheduler(cfg, pool, mem)
                browser = FakeBrowser()
                sched.browser = browser
                result = sched.run_turn("Who won the last Fisher Cats game?")
                self.assertTrue(sched.wait_for_background(timeout=8.0))
                sched.shutdown()
            self.assertIn("Fisher Cats won", result["assistant_text"])
            self.assertEqual(len(browser.search_calls), 3)
            worker = pool.get("executive")
            self.assertFalse(worker.native_generations[-1].get("enable_thinking", True))
            self.assertEqual(worker.native_tool_sets[-1], [])
            self.assertTrue(any("bounded final-answer retry" in e for e in result["trace"]["errors"]))
        finally:
            td.cleanup()

    def test_native_meta_identity_leak_is_never_shown_as_the_answer(self):
        td = tempfile.TemporaryDirectory()
        try:
            cfg = json.loads(json.dumps(DEFAULT_CONFIG))
            cfg["runtime"]["cognitive_scheduling"] = "fast"
            cfg["memory"]["path"] = str(Path(td.name) / "experience.rmem")
            mem = CognitiveRMEM(cfg["memory"]["path"])
            pool = NativeIdentityLeakPool()
            with patch("core.scheduler.STATE_PATH", Path(td.name) / "state.json"), patch("core.scheduler.LOGS_DIR", Path(td.name) / "logs"):
                sched = CognitiveScheduler(cfg, pool, mem)
                sched.browser = FakeBrowser()
                tokens, thoughts = [], []
                result = sched.run_turn(
                    "What is the current weather in Manchester, New Hampshire?",
                    on_token=tokens.append, on_thought=thoughts.append,
                )
                self.assertTrue(sched.wait_for_background(timeout=8.0))
                sched.shutdown()
            visible = "".join(tokens)
            self.assertEqual(visible, result["assistant_text"])
            self.assertNotIn("As ChatGPT", visible)
            self.assertNotIn("The user says", visible)
            self.assertNotIn("[web_search]", visible)
            self.assertNotIn("As ChatGPT", "".join(thoughts))
            self.assertIn("retrieved source", visible)
            first_system = pool.get("executive").native_transcripts[0][0]["content"]
            self.assertIn("local Executive", first_system)
            self.assertNotIn("ChatGPT", first_system)
            self.assertNotIn("QUICK REFERENCE", first_system)
        finally:
            td.cleanup()

    def test_user_supplied_public_url_can_be_fetched_without_prior_search(self):
        td = tempfile.TemporaryDirectory()
        try:
            cfg = json.loads(json.dumps(DEFAULT_CONFIG))
            cfg["memory"]["path"] = str(Path(td.name) / "experience.rmem")
            mem = CognitiveRMEM(cfg["memory"]["path"])
            pool = FakePool(tool_mode="direct_fetch")
            with patch("core.scheduler.STATE_PATH", Path(td.name) / "state.json"), patch("core.scheduler.LOGS_DIR", Path(td.name) / "logs"):
                sched = CognitiveScheduler(cfg, pool, mem)
                fake_browser = FakeBrowser()
                sched.browser = fake_browser
                result = sched.run_turn("Read https://example.com/page and summarize it")
                self.assertTrue(sched.wait_for_background(timeout=8.0))
                sched.shutdown()
            self.assertEqual(fake_browser.fetch_calls, ["https://example.com/page"])
            self.assertTrue(any(c.get("action") == "web_fetch" and c.get("ok") for c in result["trace"]["tool_calls"]))
        finally:
            td.cleanup()

    def test_adaptive_scheduler_uses_fast_assisted_and_deep_modes(self):
        td = tempfile.TemporaryDirectory()
        try:
            cfg = json.loads(json.dumps(DEFAULT_CONFIG))
            cfg["memory"]["path"] = str(Path(td.name) / "experience.rmem")
            mem = CognitiveRMEM(cfg["memory"]["path"])
            with patch("core.scheduler.STATE_PATH", Path(td.name) / "state.json"), patch("core.scheduler.LOGS_DIR", Path(td.name) / "logs"):
                sched = CognitiveScheduler(cfg, FakePool(), mem)
                self.assertEqual(sched._determine_scheduling_mode("Hello, how are you?")[0], "fast")
                self.assertEqual(sched._determine_scheduling_mode("Continue where we left off on the design")[0], "assisted")
                self.assertEqual(sched._determine_scheduling_mode("Do you remember what I told you last time?")[0], "deep")
                sched.shutdown()
        finally:
            td.cleanup()

    def test_deep_turn_uses_high_effort_reasoning_budget(self):
        td, cfg, mem, pool, result, tokens, thoughts, events = self._run(scheduling="deep")
        try:
            final_generation = pool.get("executive").generations[-1]
            self.assertEqual(result["scheduling_mode"], "deep")
            self.assertEqual(final_generation.get("reasoning_effort"), "high")
            self.assertEqual(final_generation.get("thinking_max_chars"), 4000)
        finally:
            td.cleanup()

    def test_fast_path_does_not_wait_for_cpu_qwens(self):
        td = tempfile.TemporaryDirectory()
        try:
            cfg = json.loads(json.dumps(DEFAULT_CONFIG))
            cfg["runtime"]["cognitive_scheduling"] = "fast"
            cfg["memory"]["path"] = str(Path(td.name) / "experience.rmem")
            mem = CognitiveRMEM(cfg["memory"]["path"])
            pool = FakePool(qwen_delay=0.25)
            with patch("core.scheduler.STATE_PATH", Path(td.name) / "state.json"), patch("core.scheduler.LOGS_DIR", Path(td.name) / "logs"):
                sched = CognitiveScheduler(cfg, pool, mem)
                started = time.perf_counter()
                result = sched.run_turn("Hello there")
                visible_elapsed = time.perf_counter() - started
                # qwen1 + qwen2 each sleep 250 ms in the background. The visible
                # turn must return before that sequential CPU work can finish.
                self.assertLess(visible_elapsed, 0.40)
                self.assertTrue(result["background_pending"])
                self.assertTrue(sched.wait_for_background(timeout=8.0))
                types = mem.stats()["type_counts"]
                self.assertGreaterEqual(types.get("episode", 0), 1)
                sched.shutdown()
        finally:
            td.cleanup()


    def test_real_app_memory_facade_can_complete_fast_turn(self):
        """Regression for 0.1.17: app.py passes ResilientMemory, not CognitiveRMEM."""
        from core.memory import ResilientMemory
        from core.chats import ChatStore
        with tempfile.TemporaryDirectory() as tdname:
            cfg = json.loads(json.dumps(DEFAULT_CONFIG))
            cfg["memory"]["path"] = str(Path(tdname) / "experience.rmem")
            cfg["runtime"]["cognitive_scheduling"] = "fast"
            mem = ResilientMemory(cfg["memory"]["path"])
            chats = ChatStore(Path(tdname) / "chats.json")
            chat_id = chats.active_chat_id()
            user_msg_id = chats.add_message(chat_id, "user", "Hello! My name is Mike. How are you?")
            pool = FakePool()
            with patch("core.scheduler.STATE_PATH", Path(tdname) / "state.json"), patch("core.scheduler.LOGS_DIR", Path(tdname) / "logs"):
                sched = CognitiveScheduler(cfg, pool, mem, chat_store=chats)
                result = sched.run_turn(
                    "Hello! My name is Mike. How are you?",
                    chat_id=chat_id,
                    chat_user_message_id=user_msg_id,
                )
                self.assertEqual(result["assistant_text"], "This is the Executive response.")
                self.assertTrue(sched.wait_for_background(timeout=8.0))
                sched.shutdown()
            self.assertTrue(mem.stats()["available"])
            self.assertGreaterEqual(mem.stats()["records"], 2)

    def test_background_turns_commit_in_conversation_order_without_cross_leakage(self):
        td = tempfile.TemporaryDirectory()
        try:
            cfg = json.loads(json.dumps(DEFAULT_CONFIG))
            cfg["runtime"]["cognitive_scheduling"] = "fast"
            cfg["memory"]["path"] = str(Path(td.name) / "experience.rmem")
            mem = CognitiveRMEM(cfg["memory"]["path"])
            pool = FakePool(qwen_delay=0.08)
            with patch("core.scheduler.STATE_PATH", Path(td.name) / "state.json"), patch("core.scheduler.LOGS_DIR", Path(td.name) / "logs"):
                sched = CognitiveScheduler(cfg, pool, mem)
                first = sched.run_turn("first unique turn alpha")
                second = sched.run_turn("second unique turn beta")
                self.assertNotEqual(first["turn_id"], second["turn_id"])
                self.assertTrue(sched.wait_for_background(timeout=8.0))
                episodes = list(reversed(mem.latest(count=10, types=["episode"])))
                ours = [e for e in episodes if e.get("turn_id") in {first["turn_id"], second["turn_id"]}]
                self.assertEqual([e.get("turn_id") for e in ours], [first["turn_id"], second["turn_id"]])
                self.assertEqual(ours[0].get("user_text"), "first unique turn alpha")
                self.assertEqual(ours[1].get("user_text"), "second unique turn beta")
                sched.shutdown()
        finally:
            td.cleanup()

    def test_self_state_is_first_person_qualitative_and_hides_affect_telemetry(self):
        td = tempfile.TemporaryDirectory()
        try:
            cfg = json.loads(json.dumps(DEFAULT_CONFIG))
            cfg["memory"]["path"] = str(Path(td.name) / "experience.rmem")
            mem = CognitiveRMEM(cfg["memory"]["path"])
            with patch("core.scheduler.STATE_PATH", Path(td.name) / "state.json"), patch("core.scheduler.LOGS_DIR", Path(td.name) / "logs"):
                sched = CognitiveScheduler(cfg, FakePool(), mem)
                state = {"affect": {
                    "curiosity": 0.537, "trust": 0.49, "confidence": 0.55,
                    "frustration": 0.05, "valence": 0.0, "satisfaction": 0.18,
                    "anger": 0.04, "sadness": 0.04, "anxiety": 0.04, "arousal": 0.20,
                }}
                self_state = sched._qualitative_self_state(state)
                system = sched._executive_system_message(self_state=self_state)
            self.assertIn("SELF-STATE (private)", system)
            self.assertIn("generally okay", system)
            self.assertIn("calm", system)
            self.assertIn("mildly curious", system)
            self.assertIn("reasonably confident", system)
            self.assertIn("current functional affective state", system)
            self.assertIn("answer naturally in first person", system)
            self.assertIn("Do not erase it with a generic 'AI has no feelings' disclaimer", system)
            for leaked in ('curiosity:', 'trust:', 'confidence:', 'frustration:', '0.537', '0.49', '0.55', '0.05'):
                self.assertNotIn(leaked, system)
        finally:
            td.cleanup()

    def test_direct_feelings_question_buffers_and_repairs_generic_ai_disclaimer(self):
        td = tempfile.TemporaryDirectory()
        try:
            cfg = json.loads(json.dumps(DEFAULT_CONFIG))
            cfg["memory"]["path"] = str(Path(td.name) / "experience.rmem")
            cfg["runtime"]["cognitive_scheduling"] = "fast"
            mem = CognitiveRMEM(cfg["memory"]["path"])
            pool = FakePool(tool_mode="state_denial")
            with patch("core.scheduler.STATE_PATH", Path(td.name) / "state.json"), patch("core.scheduler.LOGS_DIR", Path(td.name) / "logs"):
                sched = CognitiveScheduler(cfg, pool, mem)
                tokens, thoughts = [], []
                result = sched.run_turn(
                    "How do you feel right now?",
                    on_token=tokens.append, on_thought=thoughts.append,
                )
                self.assertTrue(sched.wait_for_background(timeout=8.0))
                sched.shutdown()

            visible = "".join(tokens)
            self.assertNotIn("don't have feelings", visible.lower())
            self.assertIn("generally okay", visible.lower())
            self.assertIn("calm", visible.lower())
            self.assertEqual(result["assistant_text"], visible)
            self.assertTrue(any("self-state disclaimer" in e.lower() for e in result["trace"]["errors"]))
            self.assertIn("generic AI self-state disclaimer rejected", "".join(thoughts))
        finally:
            td.cleanup()

    def test_assistant_name_never_leaks_from_recalled_memory_as_user_identity(self):
        td = tempfile.TemporaryDirectory()
        try:
            cfg = json.loads(json.dumps(DEFAULT_CONFIG))
            cfg["memory"]["path"] = str(Path(td.name) / "experience.rmem")
            cfg["identity"]["name"] = "Nova"
            mem = CognitiveRMEM(cfg["memory"]["path"])
            rid = mem.append("turn", {
                "role": "assistant", "identity_name": "Nova",
                "content": "Earlier assistant reply", "salience": 0.8,
            })
            with patch("core.scheduler.STATE_PATH", Path(td.name) / "state.json"), patch("core.scheduler.LOGS_DIR", Path(td.name) / "logs"):
                sched = CognitiveScheduler(cfg, FakePool(), mem)
                visible = sched._record_for_visible_context(mem.get(rid))
                tool = sched._record_for_tool_context(mem.get(rid))
                system = sched._executive_system_message()
            self.assertNotIn("identity_name", visible)
            self.assertNotIn("identity_name", tool)
            self.assertEqual(visible.get("role"), "assistant")
            self.assertEqual(tool.get("role"), "assistant")
            self.assertIn('Your name is "Nova"', system)
            self.assertIn("not the user's", system)
        finally:
            td.cleanup()


    def test_chat_store_scopes_recent_history_and_persists_assistant(self):
        from core.chats import ChatStore
        with tempfile.TemporaryDirectory() as tdname:
            cfg = json.loads(json.dumps(DEFAULT_CONFIG))
            cfg["memory"]["path"] = str(Path(tdname) / "experience.rmem")
            cfg["runtime"]["cognitive_scheduling"] = "fast"
            mem = CognitiveRMEM(cfg["memory"]["path"])
            chats = ChatStore(Path(tdname) / "chats.json")
            chat1 = chats.active_chat_id()
            chats.add_message(chat1, "user", "alpha history")
            chats.add_message(chat1, "assistant", "alpha answer")
            chat2 = chats.create_chat("beta")["id"]
            chats.add_message(chat2, "user", "beta history")
            pool = FakePool()
            state_path = Path(tdname) / "state.json"
            log_dir = Path(tdname) / "logs"
            with patch("core.scheduler.STATE_PATH", state_path), patch("core.scheduler.LOGS_DIR", log_dir):
                sched = CognitiveScheduler(cfg, pool, mem, chat_store=chats)
                user_msg_id = chats.add_message(chat1, "user", "current alpha")
                result = sched.run_turn("current alpha", chat_id=chat1, chat_user_message_id=user_msg_id)
                self.assertTrue(sched.wait_for_background(timeout=8.0))
                sched.shutdown()
            final = pool.get("executive").final_messages
            joined = "\n".join(m.get("content", "") for m in final)
            self.assertIn("alpha history", joined)
            self.assertIn("alpha answer", joined)
            self.assertNotIn("beta history", joined)
            msgs = chats.messages(chat1)
            self.assertEqual(msgs[-1]["role"], "assistant")
            self.assertEqual(msgs[-1]["content"], result["assistant_text"])
            self.assertEqual(result["chat_id"], chat1)

if __name__ == "__main__":
    unittest.main()
