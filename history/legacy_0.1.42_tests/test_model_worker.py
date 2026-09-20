import queue
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from core.model_presets import GENERIC_TEXT as GENERIC_EXECUTIVE_TEXT, detect_preset_from_filename, sampler_profile
from core.model_worker import _resolve_chat_handler, _worker_main


class FakeLLM:
    calls = []
    architecture = "mistral"
    template = "{{ messages }}"
    supports_system = True
    use_handler = True

    def __init__(self, model_path, **kwargs):
        self.model_path = model_path
        self.metadata = {
            "general.architecture": self.architecture,
            "general.name": f"{self.architecture}-test",
            "tokenizer.chat_template": self.template,
        }
        self.chat_format = "chat_template.default" if self.use_handler else None
        self._chat_handlers = {}
        self.chat_handler = self._handler if self.use_handler else None

    def _handler(self, **kwargs):
        type(self).calls.append(dict(kwargs))
        messages = kwargs.get("messages", [])
        if not self.supports_system and any(m.get("role") == "system" for m in messages):
            raise ValueError("System message is not supported by this chat template")
        enable = bool(kwargs.get("enable_thinking"))
        if kwargs.get("stream"):
            text = "private thought</think>visible answer" if enable else "visible answer"
            def gen():
                for i in range(0, len(text), 7):
                    yield {"choices": [{"delta": {"content": text[i:i+7]}}]}
            return gen()
        if kwargs.get("response_format"):
            return {"choices": [{"message": {"content": '{"ok":true}'}}]}
        return {"choices": [{"message": {"content": "visible answer"}}]}

    def create_chat_completion(self, **kwargs):
        type(self).calls.append(dict(kwargs))
        if kwargs.get("stream"):
            def gen():
                yield {"choices": [{"delta": {"content": "fallback answer"}}]}
            return gen()
        return {"choices": [{"message": {"content": "fallback answer"}}]}

    def tokenize(self, data, add_bos=True, special=True):
        return list(range(max(1, len(data) // 4)))

    def close(self):
        pass


def fake_modules(llama_cls):
    pkg = types.ModuleType("llama_cpp")
    pkg.__path__ = []
    pkg.Llama = llama_cls
    fmt = types.ModuleType("llama_cpp.llama_chat_format")
    fmt.get_chat_completion_handler = lambda _name: None
    return {"llama_cpp": pkg, "llama_cpp.llama_chat_format": fmt}


def run_worker(llama_cls, role, cfg, request=None):
    req = queue.Queue(); resp = queue.Queue(); cancel = threading.Event()
    with patch.dict(sys.modules, fake_modules(llama_cls), clear=False):
        t = threading.Thread(target=_worker_main, args=(role, cfg, req, resp, cancel), daemon=True)
        t.start()
        first = resp.get(timeout=2)
        if first.get("type") != "ready" or request is None:
            if first.get("type") == "ready":
                req.put({"op": "shutdown"})
            t.join(timeout=2)
            return first, []
        req.put(request)
        events = []
        while True:
            m = resp.get(timeout=2)
            if m.get("request_id") != request.get("request_id"):
                continue
            events.append(m)
            if m.get("type") in {"final", "error"}:
                break
        req.put({"op": "shutdown"}); t.join(timeout=2)
        return first, events


class ExecutiveModelAgnosticTests(unittest.TestCase):
    def setUp(self):
        FakeLLM.calls = []

    def _file(self, td, name="model.gguf"):
        p = Path(td) / name
        p.write_bytes(b"fake")
        return p

    def test_embedded_handler_is_preferred_when_available(self):
        good = object(); bad = object()
        llm = types.SimpleNamespace(chat_handler=None, chat_format="other", _chat_handlers={"other": bad, "chat_template.default": good})
        self.assertIs(_resolve_chat_handler(llm, types.SimpleNamespace()), good)

    def test_unknown_architecture_loads_as_executive(self):
        class Unknown(FakeLLM):
            architecture = "some_future_architecture"
            template = "{{ messages }}"
        with tempfile.TemporaryDirectory() as td:
            ready, _ = run_worker(Unknown, "executive", {
                "path": str(self._file(td)), "n_ctx": 1024, "n_batch": 32,
                "n_threads": 1, "n_gpu_layers": 0, "sampler_profile": "auto",
            })
        self.assertEqual(ready["type"], "ready")
        self.assertEqual(ready["model_info"]["architecture"], "some_future_architecture")

    def test_unknown_architecture_uses_generic_auto_sampler(self):
        class Unknown(FakeLLM):
            architecture = "future"
            template = "{{ messages }}"
        Unknown.calls = []
        with tempfile.TemporaryDirectory() as td:
            cfg = {"path": str(self._file(td)), "n_ctx": 1024, "n_batch": 32, "n_threads": 1,
                   "n_gpu_layers": 0, "sampler_profile": "auto", "enable_thinking": True}
            _, events = run_worker(Unknown, "executive", cfg, {
                "op": "chat", "request_id": "u", "messages": [{"role": "user", "content": "hi"}],
                "stream": False, "generation": {},
            })
        self.assertEqual(events[-1]["type"], "final")
        call = Unknown.calls[-1]
        for key, value in GENERIC_EXECUTIVE_TEXT.items():
            self.assertEqual(call[key], value)
        self.assertNotIn("enable_thinking", call)

    def test_granite_auto_sampler_and_native_thinking_are_detected_not_required(self):
        class Granite(FakeLLM):
            architecture = "granite"
            template = "{{ messages }}{% if enable_thinking %}<think>{% endif %}"
        Granite.calls = []
        with tempfile.TemporaryDirectory() as td:
            cfg = {"path": str(self._file(td, "granite.gguf")), "n_ctx": 1024, "n_batch": 32,
                   "n_threads": 1, "n_gpu_layers": 0, "sampler_profile": "auto",
                   "enable_thinking": True, "thinking_loop_guard": False}
            ready, events = run_worker(Granite, "executive", cfg, {
                "op": "chat", "request_id": "g", "messages": [{"role": "user", "content": "hi"}],
                "stream": True, "generation": {},
            })
        self.assertEqual(ready["type"], "ready")
        self.assertTrue(ready["model_info"]["native_thinking_template"])
        thoughts = "".join(e.get("text", "") for e in events if e.get("type") == "reasoning")
        answer = "".join(e.get("text", "") for e in events if e.get("type") == "token")
        self.assertEqual(thoughts, "private thought")
        self.assertEqual(answer, "visible answer")
        call = Granite.calls[-1]
        self.assertTrue(call["enable_thinking"])
        for key, value in sampler_profile(detect_preset_from_filename("granite-4.2-3b.gguf")).items():
            self.assertEqual(call[key], value)

    def test_granite42_uses_low_effort_reasoning_when_template_supports_it(self):
        class GraniteLow(FakeLLM):
            architecture = "granite"
            template = "{{ messages }}{% if enable_thinking %}<think>{% endif %} {{ reasoning_effort }} {{ low_effort }}"

        GraniteLow.calls = []
        with tempfile.TemporaryDirectory() as td:
            cfg = {
                "path": str(self._file(td, "granite-4.2-3b.gguf")), "n_ctx": 1024, "n_batch": 32,
                "n_threads": 1, "n_gpu_layers": 0, "sampler_profile": "auto",
                "thinking_loop_guard": False,
            }
            ready, events = run_worker(GraniteLow, "executive", cfg, {
                "op": "chat", "request_id": "low", "messages": [{"role": "user", "content": "hi"}],
                "stream": True, "generation": {"reasoning_effort": "low"},
            })
        self.assertEqual(events[-1]["type"], "final")
        self.assertTrue(ready["model_info"]["native_reasoning_effort_template"])
        call = GraniteLow.calls[-1]
        self.assertTrue(call["enable_thinking"])
        self.assertEqual(call["reasoning_effort"], "low")

    def test_per_call_reasoning_guard_override_is_honored(self):
        class RunawayGranite(FakeLLM):
            architecture = "granite"
            template = "{{ messages }}{% if enable_thinking %}<think>{% endif %}"

            def _handler(self, **kwargs):
                type(self).calls.append(dict(kwargs))
                if kwargs.get("stream"):
                    enabled = bool(kwargs.get("enable_thinking"))
                    def gen():
                        if enabled:
                            reasoning = (
                                "The answer is already clear: use the maintained self-state and answer directly. "
                                "That conclusion is sufficient. But maybe I should reconsider it again anyway. "
                                + ("reconsider " * 180)
                            )
                            yield {"choices": [{"delta": {"content": reasoning}}]}
                        else:
                            yield {"choices": [{"delta": {"content": "bounded answer"}}]}
                    return gen()
                return {"choices": [{"message": {"content": "bounded answer"}}]}

        RunawayGranite.calls = []
        with tempfile.TemporaryDirectory() as td:
            cfg = {
                "path": str(self._file(td, "granite-4.2-3b.gguf")), "n_ctx": 2048, "n_batch": 32,
                "n_threads": 1, "n_gpu_layers": 0, "sampler_profile": "auto",
                "thinking_loop_guard": True, "thinking_fallback_nonthinking": True,
                "thinking_max_chars": 4000,
            }
            _, events = run_worker(RunawayGranite, "executive", cfg, {
                "op": "chat", "request_id": "guard-override",
                "messages": [{"role": "user", "content": "hi"}], "stream": True,
                "generation": {"thinking_max_chars": 1400, "reasoning_effort": "low"},
            })
        final = events[-1]
        self.assertTrue(final["loop_guard_triggered"])
        self.assertTrue(final["fallback_used"])
        self.assertIn("reasoning exceeded 1400 characters", final["thinking"])
        self.assertEqual(final["result"], "bounded answer")

    def test_reasoning_channel_is_guarded_even_without_enable_thinking_metadata(self):
        class ReasoningChannelGranite(FakeLLM):
            architecture = "granite"
            # Deliberately omits enable_thinking from template capability metadata.
            template = "{{ messages }}"

            def _handler(self, **kwargs):
                type(self).calls.append(dict(kwargs))
                call_no = len(type(self).calls)
                if kwargs.get("stream"):
                    def gen():
                        if call_no == 1:
                            yield {"choices": [{"delta": {"reasoning_content": "r" * 1600}}]}
                        else:
                            yield {"choices": [{"delta": {"content": "visible after guarded reasoning"}}]}
                    return gen()
                return {"choices": [{"message": {"content": "visible after guarded reasoning"}}]}

        ReasoningChannelGranite.calls = []
        with tempfile.TemporaryDirectory() as td:
            cfg = {
                "path": str(self._file(td, "granite-4.2-3b.gguf")), "n_ctx": 2048, "n_batch": 32,
                "n_threads": 1, "n_gpu_layers": 0, "sampler_profile": "auto",
                "thinking_loop_guard": True, "thinking_fallback_nonthinking": True,
            }
            ready, events = run_worker(ReasoningChannelGranite, "executive", cfg, {
                "op": "chat", "request_id": "reasoning-channel-guard",
                "messages": [{"role": "user", "content": "hi"}], "stream": True,
                "generation": {"thinking_max_chars": 1400, "reasoning_effort": "low"},
            })
        final = events[-1]
        self.assertFalse(ready["model_info"]["native_thinking_template"])
        self.assertTrue(final["loop_guard_triggered"])
        self.assertTrue(final["fallback_used"])
        self.assertEqual(final["result"], "visible after guarded reasoning")
        self.assertEqual("".join(e.get("text", "") for e in events if e.get("type") == "token"),
                         "visible after guarded reasoning")

    def test_default_reasoning_guard_recovers_before_reasoning_consumes_generation(self):
        class RunawayGranite(FakeLLM):
            architecture = "granite"
            template = "{{ messages }}{% if enable_thinking %}<think>{% endif %}"

            def _handler(self, **kwargs):
                type(self).calls.append(dict(kwargs))
                if kwargs.get("stream"):
                    enabled = bool(kwargs.get("enable_thinking"))

                    def gen():
                        if enabled:
                            # Exceeds the 0.1.26 default 4k guard before any answer.
                            yield {"choices": [{"delta": {"content": "r" * 4100}}]}
                        else:
                            yield {"choices": [{"delta": {"content": "recovered visible answer"}}]}
                    return gen()
                return {"choices": [{"message": {"content": "recovered visible answer"}}]}

        RunawayGranite.calls = []
        with tempfile.TemporaryDirectory() as td:
            cfg = {
                "path": str(self._file(td, "granite.gguf")), "n_ctx": 2048, "n_batch": 32,
                "n_threads": 1, "n_gpu_layers": 0, "sampler_profile": "auto",
                "enable_thinking": True, "thinking_loop_guard": True,
                "thinking_fallback_nonthinking": True,
            }
            _, events = run_worker(RunawayGranite, "executive", cfg, {
                "op": "chat", "request_id": "guard", "messages": [{"role": "user", "content": "hi"}],
                "stream": True, "generation": {},
            })
        final = events[-1]
        answer = "".join(e.get("text", "") for e in events if e.get("type") == "token")
        self.assertEqual(final["type"], "final")
        self.assertTrue(final["loop_guard_triggered"])
        self.assertTrue(final["fallback_used"])
        self.assertEqual(answer, "recovered visible answer")
        self.assertEqual(final["result"], "recovered visible answer")
        self.assertIn("reasoning exceeded 4000 characters", final["thinking"])

    def test_embedded_native_tool_template_is_detected_and_receives_tools(self):
        class ToolTemplate(FakeLLM):
            architecture = "granite"
            template = "{{ messages }} {{ tools }} <tool_call> <tool_response> {% if enable_thinking %}<think>{% endif %}"
        with tempfile.TemporaryDirectory() as td:
            request = {
                "op": "chat", "request_id": "tools", "messages": [{"role": "user", "content": "weather"}],
                "stream": False, "tools": [{"type": "function", "function": {"name": "web_search", "parameters": {"type": "object"}}}],
                "generation": {"enable_thinking": False},
            }
            ready, events = run_worker(ToolTemplate, "executive", {"path": str(self._file(td))}, request)
        self.assertTrue(ready["model_info"]["native_tools"])
        self.assertEqual(events[-1]["type"], "final")
        self.assertEqual(ToolTemplate.calls[-1]["tools"][0]["function"]["name"], "web_search")

    def test_qwen_worker_remains_strictly_qwen35(self):
        class Granite(FakeLLM):
            architecture = "granite"
            template = "{{ messages }}{% if enable_thinking %}<think>{% endif %}"
        with tempfile.TemporaryDirectory() as td:
            msg, _ = run_worker(Granite, "qwen1", {"path": str(self._file(td)), "required_architecture": "qwen35"})
        self.assertEqual(msg["type"], "startup_error")
        self.assertIn("requires 'qwen35'", msg["error"])

    def test_qwen_worker_requires_native_template_but_executive_does_not(self):
        class QwenNoThinking(FakeLLM):
            architecture = "qwen35"
            template = "{{ messages }}"
        with tempfile.TemporaryDirectory() as td:
            worker_msg, _ = run_worker(QwenNoThinking, "qwen2", {"path": str(self._file(td)), "required_architecture": "qwen35"})
            exec_msg, _ = run_worker(QwenNoThinking, "executive", {"path": str(self._file(td, "exec.gguf"))})
        self.assertEqual(worker_msg["type"], "startup_error")
        self.assertEqual(exec_msg["type"], "ready")

    def test_executive_without_embedded_template_uses_llama_cpp_fallback(self):
        class NoTemplate(FakeLLM):
            architecture = "llama"
            template = ""
            use_handler = False
        NoTemplate.calls = []
        with tempfile.TemporaryDirectory() as td:
            ready, events = run_worker(NoTemplate, "executive", {"path": str(self._file(td))}, {
                "op": "chat", "request_id": "f", "messages": [{"role": "user", "content": "hi"}],
                "stream": False, "generation": {},
            })
        self.assertEqual(ready["type"], "ready")
        self.assertFalse(ready["model_info"]["chat_template"])
        self.assertEqual(events[-1]["result"], "fallback answer")

    def test_system_role_fallback_preserves_current_user_message(self):
        class NoSystem(FakeLLM):
            architecture = "gemma"
            template = "{{ messages }}"  # no literal system support -> proactive adaptation
            supports_system = False
        NoSystem.calls = []
        messages = [
            {"role": "system", "content": "PRIVATE IDENTITY"},
            {"role": "user", "content": "older"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "THIS IS MY EXACT CURRENT MESSAGE"},
        ]
        with tempfile.TemporaryDirectory() as td:
            _, events = run_worker(NoSystem, "executive", {"path": str(self._file(td))}, {
                "op": "chat", "request_id": "s", "messages": messages,
                "stream": False, "generation": {},
            })
        self.assertEqual(events[-1]["type"], "final")
        sent = NoSystem.calls[-1]["messages"]
        self.assertFalse(any(m["role"] == "system" for m in sent))
        self.assertEqual(sent[-1], {"role": "user", "content": "THIS IS MY EXACT CURRENT MESSAGE"})
        self.assertIn("PRIVATE IDENTITY", sent[0]["content"])


    def test_system_role_error_retries_with_private_primer(self):
        class RejectSystem(FakeLLM):
            architecture = "future_arch"
            template = "{{ messages }} system token appears here"  # defeats proactive heuristic
            supports_system = False
        RejectSystem.calls = []
        messages = [
            {"role": "system", "content": "PRIVATE SYSTEM"},
            {"role": "user", "content": "CURRENT RAW USER"},
        ]
        with tempfile.TemporaryDirectory() as td:
            _, events = run_worker(RejectSystem, "executive", {"path": str(self._file(td))}, {
                "op": "chat", "request_id": "retry", "messages": messages,
                "stream": False, "generation": {},
            })
        self.assertEqual(events[-1]["type"], "final")
        self.assertTrue(events[-1]["system_fallback_used"])
        self.assertGreaterEqual(len(RejectSystem.calls), 2)
        sent = RejectSystem.calls[-1]["messages"]
        self.assertFalse(any(m["role"] == "system" for m in sent))
        self.assertEqual(sent[-1], {"role": "user", "content": "CURRENT RAW USER"})


    def test_qwen35_prompt_injected_reasoning_never_leaks_to_chat(self):
        class Qwen(FakeLLM):
            architecture = "qwen35"
            template = "{{ messages }}{% if add_generation_prompt %}{% if enable_thinking %}<think>{% endif %}{% endif %}"
        Qwen.calls = []
        with tempfile.TemporaryDirectory() as td:
            cfg = {"path": str(self._file(td, "renamed.gguf")), "n_ctx": 1024, "n_batch": 32,
                   "n_threads": 1, "n_gpu_layers": 0, "sampler_profile": "auto",
                   "thinking_loop_guard": False}
            ready, events = run_worker(Qwen, "executive", cfg, {
                "op": "chat", "request_id": "qthink",
                "messages": [{"role": "user", "content": "think about this"}],
                "stream": True, "generation": {},
            })
        self.assertEqual(ready["model_info"]["preset_key"], "qwen35")
        self.assertTrue(ready["model_info"]["prompt_opens_thinking"])
        thoughts = "".join(e.get("text", "") for e in events if e.get("type") == "reasoning")
        answer = "".join(e.get("text", "") for e in events if e.get("type") == "token")
        self.assertEqual(thoughts, "private thought")
        self.assertEqual(answer, "visible answer")
        self.assertNotIn("private thought", answer)

    def test_streaming_preserves_finish_reason_length(self):
        class LengthStop(FakeLLM):
            architecture = "granite"
            template = "{{ messages }}"

            def _handler(self, **kwargs):
                type(self).calls.append(dict(kwargs))
                if kwargs.get("stream"):
                    def gen():
                        yield {"choices": [{"delta": {"content": "partial answer"}, "finish_reason": None}]}
                        yield {"choices": [{"delta": {}, "finish_reason": "length"}]}
                    return gen()
                return {"choices": [{"message": {"content": "partial answer"}, "finish_reason": "length"}]}

        LengthStop.calls = []
        with tempfile.TemporaryDirectory() as td:
            cfg = {"path": str(self._file(td, "granite.gguf")), "n_ctx": 1024, "n_batch": 32,
                   "n_threads": 1, "n_gpu_layers": 0, "sampler_profile": "auto",
                   "enable_thinking": False, "thinking_loop_guard": False}
            _, events = run_worker(LengthStop, "executive", cfg, {
                "op": "chat", "request_id": "length-reason",
                "messages": [{"role": "user", "content": "hi"}], "stream": True,
                "generation": {"enable_thinking": False, "max_tokens": 500},
            })
        self.assertEqual(events[-1]["type"], "final")
        self.assertEqual(events[-1]["result"], "partial answer")
        self.assertEqual(events[-1]["finish_reason"], "length")

    def test_streaming_preserves_structured_tool_call_deltas(self):
        class ToolGranite(FakeLLM):
            architecture = "granite"
            template = "{{ messages }} {{ tools }} {% if enable_thinking %}<think>{% endif %} <tool_call>"

            def _handler(self, **kwargs):
                type(self).calls.append(dict(kwargs))
                if kwargs.get("stream"):
                    def gen():
                        yield {"choices": [{"delta": {"content": "plan</think>"}}]}
                        yield {"choices": [{"delta": {"tool_calls": [{
                            "index": 0, "id": "call_1", "type": "function",
                            "function": {"name": "web_", "arguments": '{"query":"Fisher '},
                        }]}}]}
                        yield {"choices": [{"delta": {"tool_calls": [{
                            "index": 0,
                            "function": {"name": "search", "arguments": 'Cats"}'},
                        }]}}]}
                    return gen()
                return {"choices": [{"message": {"content": ""}}]}

        ToolGranite.calls = []
        with tempfile.TemporaryDirectory() as td:
            cfg = {
                "path": str(self._file(td, "granite-4.2-3b.gguf")),
                "n_ctx": 1024, "n_batch": 32, "n_threads": 1, "n_gpu_layers": 0,
                "sampler_profile": "auto", "thinking_loop_guard": False,
            }
            _, events = run_worker(ToolGranite, "executive", cfg, {
                "op": "chat", "request_id": "tool-stream",
                "messages": [{"role": "user", "content": "latest game?"}],
                "stream": True, "generation": {"reasoning_effort": "low"},
                "tools": [{"type": "function", "function": {
                    "name": "web_search", "description": "search",
                    "parameters": {"type": "object"},
                }}],
            })
        final = events[-1]
        self.assertEqual(final["type"], "final")
        self.assertEqual(final["thinking"], "plan")
        self.assertEqual(len(final["tool_calls"]), 1)
        self.assertEqual(final["tool_calls"][0]["function"]["name"], "web_search")
        self.assertEqual(final["tool_calls"][0]["function"]["arguments"], '{"query":"Fisher Cats"}')

    def test_mistral_filename_refines_shared_llama_architecture(self):
        class Mistral(FakeLLM):
            architecture = "llama"
            template = "{{ messages }}"
        Mistral.calls = []
        with tempfile.TemporaryDirectory() as td:
            ready, events = run_worker(Mistral, "executive", {
                "path": str(self._file(td, "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf")),
                "n_ctx": 1024, "n_batch": 32, "n_threads": 1, "n_gpu_layers": 0,
                "sampler_profile": "auto",
            }, {
                "op": "chat", "request_id": "mistral",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False, "generation": {},
            })
        self.assertEqual(ready["model_info"]["preset_key"], "mistral")
        self.assertEqual(events[-1]["type"], "final")
        self.assertAlmostEqual(Mistral.calls[-1]["temperature"], 0.7)

    def test_json_schema_disables_native_thinking_when_available(self):
        class Thinking(FakeLLM):
            architecture = "granite"
            template = "{{ messages }}{% if enable_thinking %}<think>{% endif %}"
        Thinking.calls = []
        with tempfile.TemporaryDirectory() as td:
            _, events = run_worker(Thinking, "executive", {"path": str(self._file(td)), "enable_thinking": True}, {
                "op": "chat", "request_id": "j", "messages": [{"role": "user", "content": "json"}],
                "stream": False, "json_schema": {"type": "object"}, "generation": {"enable_thinking": True},
            })
        self.assertEqual(events[-1]["type"], "final")
        self.assertFalse(events[-1]["thinking_enabled"])
        self.assertFalse(Thinking.calls[-1]["enable_thinking"])


if __name__ == "__main__":
    unittest.main()
