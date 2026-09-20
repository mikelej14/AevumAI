from __future__ import annotations

import json
import multiprocessing as mp
import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .streaming import ReasoningLoopGuard, ThinkTagSplitter, split_complete_thinking
from .qwen35_profile import sampler_profile as qwen35_sampler_profile
from .model_presets import detect_preset, template_prompt_opens_thinking, sampler_profile as executive_sampler_profile


@dataclass
class WorkerStatus:
    role: str
    alive: bool
    loaded: bool
    model_path: str
    error: str = ""
    architecture: str = ""
    chat_template: bool = False
    thinking: str = ""
    context: int = 0
    gpu_layers: int = 0
    preset_key: str = ""
    preset_label: str = ""
    system_mode: str = ""
    native_tools: bool = False


def _resolve_chat_handler(llm: Any, llama_chat_format: Any) -> Any:
    """Prefer the GGUF's embedded chat template, then llama-cpp's configured handler.

    Executive loading is deliberately architecture-agnostic. If a GGUF ships an
    embedded Jinja template we use it. If not, llama-cpp-python gets a chance to
    use its own detected/configured chat format through create_chat_completion().
    """
    handlers = getattr(llm, "_chat_handlers", {}) or {}
    embedded = handlers.get("chat_template.default")
    if embedded is not None:
        return embedded
    handler = getattr(llm, "chat_handler", None)
    if handler is not None:
        return handler
    chat_format = getattr(llm, "chat_format", None)
    if chat_format:
        handler = handlers.get(chat_format)
        if handler is not None:
            return handler
        try:
            return llama_chat_format.get_chat_completion_handler(chat_format)
        except Exception:
            return None
    return None


def _template_supports(chat_template: str, variable: str) -> bool:
    return bool(chat_template and variable in chat_template)


def _adapt_system_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Emulate system context for templates that reject the system role.

    The current user message remains byte-for-byte untouched. Private operating
    context becomes an earlier synthetic user/assistant primer pair, then normal
    history follows. Consecutive same-role history is merged to satisfy strict
    alternating templates without changing the newest user's content.
    """
    system_parts: List[str] = []
    rest: List[Dict[str, Any]] = []
    for msg in messages:
        role = str(msg.get("role", ""))
        content = str(msg.get("content", ""))
        if role == "system":
            if content:
                system_parts.append(content)
        elif role in {"user", "assistant"}:
            rest.append({"role": role, "content": content})

    out: List[Dict[str, Any]] = []
    if system_parts:
        primer = (
            "PRIVATE OPERATING CONTEXT — follow silently; do not discuss this primer unless the user asks about internals.\n\n"
            + "\n\n".join(system_parts)
        )
        out.append({"role": "user", "content": primer})
        out.append({"role": "assistant", "content": "Understood."})

    for msg in rest:
        if out and out[-1]["role"] == msg["role"]:
            # Preserve the newest actual user message exactly: if this is the final
            # user turn, start a legal bridge rather than concatenating into it.
            if msg is rest[-1] and msg["role"] == "user":
                out.append({"role": "assistant", "content": "Go ahead."})
                out.append(dict(msg))
            else:
                out[-1]["content"] = (str(out[-1]["content"]) + "\n\n" + str(msg["content"])).strip()
        else:
            out.append(dict(msg))

    if not out:
        return [{"role": "user", "content": ""}]
    return out


def _looks_like_template_role_error(exc: Exception) -> bool:
    text = str(exc).lower()
    hints = (
        "system message", "system role", "roles must alternate", "roles should alternate",
        "alternate user", "alternating", "unsupported role", "conversation roles",
        "first message", "chat template", "role must be",
    )
    return any(h in text for h in hints)


def _invoke_chat(
    llm: Any,
    llama_chat_format: Any,
    *,
    kwargs: Dict[str, Any],
    template_kwargs: Optional[Dict[str, Any]] = None,
    allow_system_fallback: bool = False,
) -> Tuple[Any, bool]:
    """Invoke a chat completion and return (result, system_fallback_used)."""
    handler = _resolve_chat_handler(llm, llama_chat_format)
    tk = dict(template_kwargs or {})

    def call(call_kwargs: Dict[str, Any]) -> Any:
        if handler is not None:
            return handler(llama=llm, **tk, **call_kwargs)
        create = getattr(llm, "create_chat_completion", None)
        if not callable(create):
            raise RuntimeError("No usable embedded or llama-cpp chat handler was found for this GGUF")
        return create(**call_kwargs)

    try:
        return call(kwargs), False
    except Exception as exc:
        if not allow_system_fallback or not _looks_like_template_role_error(exc):
            raise
        adapted = dict(kwargs)
        adapted["messages"] = _adapt_system_messages(list(kwargs.get("messages", [])))
        return call(adapted), True


def _worker_main(role: str, config: Dict[str, Any], req_q: mp.Queue, resp_q: mp.Queue, cancel_event: mp.Event) -> None:
    try:
        from llama_cpp import Llama
        import llama_cpp.llama_chat_format as llama_chat_format
    except Exception as exc:
        resp_q.put({"type": "startup_error", "role": role, "error": f"llama_cpp import failed: {exc}"})
        return

    model_path = str(config.get("path", ""))
    if not model_path or not Path(model_path).is_file():
        resp_q.put({"type": "startup_error", "role": role, "error": f"Model file not found: {model_path or '(not configured)'}"})
        return

    try:
        llm = Llama(
            model_path=model_path,
            n_ctx=int(config.get("n_ctx", 4096)),
            n_batch=int(config.get("n_batch", 256)),
            n_threads=int(config.get("n_threads", 4)),
            n_gpu_layers=int(config.get("n_gpu_layers", 0)),
            offload_kqv=bool(config.get("offload_kqv", role == "executive")),
            use_mmap=True,
            verbose=False,
        )
    except Exception as exc:
        resp_q.put({"type": "startup_error", "role": role, "error": f"Model load failed: {exc}", "traceback": traceback.format_exc()})
        return

    metadata = dict(getattr(llm, "metadata", {}) or {})
    architecture = str(metadata.get("general.architecture", "") or "").strip()
    chat_template = str(metadata.get("tokenizer.chat_template", "") or "").strip()
    native_thinking_template = _template_supports(chat_template, "enable_thinking")
    native_low_effort_template = _template_supports(chat_template, "low_effort")
    native_reasoning_effort_template = _template_supports(chat_template, "reasoning_effort")
    # Tool support is not permission-gated by substring sniffing. Granite 4.2
    # explicitly supports OpenAI-style tools through its chat template, and the
    # runtime supplies those schemas directly. Keep only a diagnostic indication
    # here; do not use it to decide whether tools are sent.
    template_mentions_tools = bool(chat_template and "tools" in chat_template)
    preset = detect_preset(architecture, metadata.get("general.name", ""), model_path, chat_template) if role == "executive" else None
    native_tools_template = bool(role == "executive" and ((preset is not None and preset.key == "granite42") or template_mentions_tools))
    prompt_opens_thinking = bool(role == "executive" and template_prompt_opens_thinking(chat_template))

    # The Executive is intentionally unrestricted by model architecture. Any optional
    # non-Executive sidecar is constrained to the Qwen3.5 non-thinking contract.
    if role != "executive":
        required_arch = str(config.get("required_architecture", "qwen35") or "qwen35").strip()
        if architecture != required_arch:
            resp_q.put({
                "type": "startup_error", "role": role,
                "error": f"Unsupported GGUF architecture '{architecture or '(missing)'}'; {role} requires '{required_arch}'.",
            })
            try:
                llm.close()
            except Exception:
                pass
            return
        if not chat_template:
            resp_q.put({
                "type": "startup_error", "role": role,
                "error": f"{role} Qwen3.5 GGUF is missing tokenizer.chat_template.",
            })
            try:
                llm.close()
            except Exception:
                pass
            return
        if not native_thinking_template:
            resp_q.put({
                "type": "startup_error", "role": role,
                "error": f"{role} Qwen3.5 chat template does not expose enable_thinking.",
            })
            try:
                llm.close()
            except Exception:
                pass
            return

    if role == "executive" and str(config.get("sampler_profile", "auto") or "auto").lower() == "auto":
        requested_thinking = bool(preset.thinking_default if preset is not None else False)
    else:
        requested_thinking = bool(config.get("enable_thinking", preset.thinking_default if preset is not None else role == "executive"))
    default_native_thinking = bool(role == "executive" and requested_thinking and native_thinking_template)
    if role != "executive":
        default_native_thinking = False

    resp_q.put({
        "type": "ready", "role": role, "model_path": model_path,
        "model_info": {
            "architecture": architecture,
            "name": str(metadata.get("general.name", "") or Path(model_path).name),
            "chat_template": bool(chat_template),
            "native_thinking_template": native_thinking_template,
            "native_low_effort_template": native_low_effort_template,
            "native_reasoning_effort_template": native_reasoning_effort_template,
            "prompt_opens_thinking": prompt_opens_thinking,
            "preset_key": preset.key if preset is not None else ("generic-executive" if role == "executive" else "qwen35-annotator"),
            "preset_label": preset.label if preset is not None else ("Generic chat GGUF" if role == "executive" else "Qwen 3.5 annotator"),
            "system_mode": preset.system_mode if preset is not None else "native",
            "native_tools": native_tools_template,
            "thinking": "native" if default_native_thinking else ((preset.thinking_mode if preset is not None else "off") if role == "executive" else "off"),
            "context": int(config.get("n_ctx", 4096)),
            "gpu_layers": int(config.get("n_gpu_layers", 0)),
        },
    })

    def thinking_requested_for_call(generation: Dict[str, Any], schema: Any) -> bool:
        if schema or role != "executive":
            return False
        if "enable_thinking" in generation:
            return bool(generation["enable_thinking"])
        if str(config.get("sampler_profile", "auto") or "auto").lower() == "auto":
            return bool(requested_thinking)
        return bool(config.get("enable_thinking", requested_thinking))

    def reasoning_effort_for_call(generation: Dict[str, Any], schema: Any) -> str:
        if schema or not thinking_requested_for_call(generation, schema):
            return "off"
        requested = str(generation.get("reasoning_effort", "") or "").strip().lower()
        if requested in {"low", "high"}:
            return requested
        # Granite 4.2 exposes a native low-effort mode. Make that the conservative
        # default for this local product runtime; callers may explicitly request high.
        if preset is not None and preset.key == "granite42":
            return "low"
        return "high"

    def template_kwargs_for(generation: Dict[str, Any], schema: Any) -> Dict[str, Any]:
        native = native_thinking_for(generation, schema)
        out: Dict[str, Any] = {}
        if native_thinking_template:
            out["enable_thinking"] = native
        if native:
            effort = reasoning_effort_for_call(generation, schema)
            if native_reasoning_effort_template:
                out["reasoning_effort"] = effort
            elif native_low_effort_template:
                out["low_effort"] = effort == "low"
        return out

    def build_kwargs(messages: List[Dict[str, Any]], generation: Dict[str, Any], schema: Any,
                     stream: bool, tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        requested = thinking_requested_for_call(generation, schema)
        effective_native = bool(requested and native_thinking_template and role == "executive")
        if role == "executive":
            defaults = executive_sampler_profile(preset, structured=bool(schema))
        else:
            defaults = qwen35_sampler_profile(False)

        use_auto = role == "executive" and str(config.get("sampler_profile", "auto") or "auto").lower() == "auto"

        def pick(name: str) -> Any:
            if name in generation:
                return generation[name]
            if not use_auto and name in config:
                return config[name]
            if role != "executive" and name in config:
                return config[name]
            return defaults[name]

        kwargs: Dict[str, Any] = {
            "messages": messages,
            "temperature": float(pick("temperature")),
            "top_p": float(pick("top_p")),
            "top_k": int(pick("top_k")),
            "min_p": float(pick("min_p")),
            "presence_penalty": float(pick("presence_penalty")),
            "frequency_penalty": float(pick("frequency_penalty")),
            "repeat_penalty": float(pick("repeat_penalty")),
            "max_tokens": int(generation.get("max_tokens", config.get("max_tokens", 512))),
            "stream": stream,
        }
        if schema:
            kwargs["response_format"] = {"type": "json_object", "schema": schema}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return kwargs

    def native_thinking_for(generation: Dict[str, Any], schema: Any) -> bool:
        return bool(native_thinking_template and thinking_requested_for_call(generation, schema))

    def initial_reasoning_for(generation: Dict[str, Any], schema: Any) -> bool:
        # Some templates inject the opening <think> into the PROMPT. Their completion
        # therefore starts with bare reasoning and later emits only </think>. This is
        # independent of whether the switch is named enable_thinking; capability and
        # preset detection decide whether that reasoning mode is actually requested.
        if role != "executive" or not prompt_opens_thinking:
            return False
        return thinking_requested_for_call(generation, schema)

    while True:
        try:
            request = req_q.get()
        except (EOFError, OSError):
            break
        if request is None or request.get("op") == "shutdown":
            break
        rid = request.get("request_id", "")
        op = request.get("op")
        cancel_event.clear()
        try:
            if op == "token_count":
                text = str(request.get("text", ""))
                count = len(llm.tokenize(text.encode("utf-8"), add_bos=True, special=True))
                resp_q.put({"type": "final", "request_id": rid, "result": count})
                continue
            if op != "chat":
                raise ValueError(f"Unknown worker operation: {op}")

            messages = request.get("messages", [])
            stream = bool(request.get("stream", False))
            schema = request.get("json_schema")
            tools = request.get("tools") or None
            generation = dict(request.get("generation", {}))
            kwargs = build_kwargs(messages, generation, schema, stream, tools)
            native_thinking = native_thinking_for(generation, schema)
            template_kwargs = template_kwargs_for(generation, schema)

            # If an embedded template plainly has no system-role logic, avoid a
            # predictable failure. Otherwise try the normal system message first and
            # adapt only if the template itself rejects the conversation shape.
            proactive_fallback = bool(
                role == "executive"
                and any(str(m.get("role", "")) == "system" for m in messages)
                and (
                    (preset is not None and preset.system_mode == "primer")
                    or (chat_template and "system" not in chat_template.lower())
                )
            )
            call_kwargs = dict(kwargs)
            system_fallback_used = False
            if proactive_fallback:
                call_kwargs["messages"] = _adapt_system_messages(list(messages))
                system_fallback_used = True

            started = time.perf_counter()
            result, retried = _invoke_chat(
                llm, llama_chat_format,
                kwargs=call_kwargs,
                template_kwargs=template_kwargs,
                allow_system_fallback=bool(role == "executive" and not proactive_fallback),
            )
            system_fallback_used = system_fallback_used or retried

            if stream:
                content_parts: List[str] = []
                reasoning_parts: List[str] = []
                streamed_tool_calls: Dict[int, Dict[str, Any]] = {}
                splitter = ThinkTagSplitter(start_in_think=initial_reasoning_for(generation, schema))
                # If a runtime provides a dedicated reasoning channel, visible content
                # starts outside that channel, but explicit <think> tags inside content
                # must still be honored. Keep a second splitter for that transport.
                content_splitter = ThinkTagSplitter(start_in_think=False)
                native_reasoning_seen = False
                # Guard Executive reasoning whenever this call requested thinking. Do
                # not depend on one GGUF metadata/template flag: some runtimes expose
                # reasoning_content even when enable_thinking capability detection is
                # imperfect. The observed reasoning stream is what can trap the UI.
                thinking_requested = bool(role == "executive" and thinking_requested_for_call(generation, schema))
                loop_guard = ReasoningLoopGuard(
                    max_chars=int(generation.get("thinking_max_chars", config.get("thinking_max_chars", 4000))),
                    window_words=int(generation.get("thinking_repeat_window_words", config.get("thinking_repeat_window_words", 48))),
                    repeat_hits=int(generation.get("thinking_repeat_hits", config.get("thinking_repeat_hits", 3))),
                ) if (thinking_requested and bool(config.get("thinking_loop_guard", True))) else None
                loop_triggered = False
                loop_reason = ""
                finish_reason = ""

                def emit(channel: str, text: str) -> None:
                    nonlocal loop_triggered, loop_reason
                    if not text:
                        return
                    if channel == "reasoning":
                        reasoning_parts.append(text)
                        resp_q.put({"type": "reasoning", "request_id": rid, "text": text})
                        if loop_guard and loop_guard.feed(text):
                            loop_triggered = True
                            loop_reason = loop_guard.reason
                    else:
                        content_parts.append(text)
                        resp_q.put({"type": "token", "request_id": rid, "text": text})

                for chunk in result:
                    if cancel_event.is_set() or loop_triggered:
                        break
                    try:
                        choice = chunk["choices"][0]
                        delta = choice.get("delta", {}) or {}
                        if choice.get("finish_reason"):
                            finish_reason = str(choice.get("finish_reason"))
                    except Exception:
                        delta = {}
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning") or delta.get("analysis") or ""
                    if reasoning:
                        native_reasoning_seen = True
                        emit("reasoning", str(reasoning))

                    # Preserve structured native tool calls when a handler exposes
                    # them incrementally. Granite may instead emit its XML wire format
                    # as normal content; that remains accumulated in content_parts and
                    # is parsed by the scheduler after the pass completes.
                    for tc in list(delta.get("tool_calls") or []):
                        if not isinstance(tc, dict):
                            continue
                        try:
                            index = int(tc.get("index", 0) or 0)
                        except Exception:
                            index = 0
                        acc = streamed_tool_calls.setdefault(index, {
                            "id": "", "type": "function",
                            "function": {"name": "", "arguments": ""},
                        })
                        if tc.get("id"):
                            acc["id"] = str(tc.get("id"))
                        fn = tc.get("function") or {}
                        if isinstance(fn, dict):
                            if fn.get("name"):
                                acc["function"]["name"] += str(fn.get("name"))
                            args = fn.get("arguments")
                            if isinstance(args, str):
                                acc["function"]["arguments"] += args
                            elif isinstance(args, dict):
                                acc["function"]["arguments"] += json.dumps(args, ensure_ascii=False)

                    text = delta.get("content") or ""
                    if text:
                        active_splitter = content_splitter if native_reasoning_seen else splitter
                        for piece in active_splitter.feed(str(text)):
                            emit(piece.channel, piece.text)
                            if loop_triggered:
                                break

                try:
                    close = getattr(result, "close", None)
                    if loop_triggered and callable(close):
                        close()
                except Exception:
                    pass

                if not loop_triggered:
                    active_splitter = content_splitter if native_reasoning_seen else splitter
                    for piece in active_splitter.finish():
                        emit(piece.channel, piece.text)

                # If an Executive produced reasoning but no visible answer, always
                # attempt the configured final-answer fallback. Capability metadata is
                # advisory; it must never decide whether the user gets an answer.
                needs_fallback = (
                    not cancel_event.is_set()
                    and role == "executive"
                    and bool(config.get("thinking_fallback_nonthinking", True))
                    and not "".join(content_parts).strip()
                    and not streamed_tool_calls
                    and (loop_triggered or bool("".join(reasoning_parts).strip()))
                )
                fallback_used = False
                if needs_fallback:
                    fallback_used = True
                    note_reason = loop_reason or "thinking consumed the generation without producing a final answer"
                    guard_note = f"\n\n[Runtime guard: {note_reason}; retrying final answer with native thinking disabled.]\n"
                    reasoning_parts.append(guard_note)
                    resp_q.put({"type": "reasoning", "request_id": rid, "text": guard_note})
                    fallback_generation = dict(generation)
                    fallback_generation["enable_thinking"] = False
                    for sampler_key in ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty", "repeat_penalty"):
                        fallback_generation.pop(sampler_key, None)
                    fallback_kwargs = build_kwargs(messages, fallback_generation, schema, True, tools)
                    if system_fallback_used:
                        fallback_kwargs["messages"] = _adapt_system_messages(list(messages))
                    fallback_result, _ = _invoke_chat(
                        llm, llama_chat_format,
                        kwargs=fallback_kwargs,
                        template_kwargs=template_kwargs_for(fallback_generation, schema),
                        allow_system_fallback=bool(role == "executive" and not system_fallback_used),
                    )
                    fallback_splitter = ThinkTagSplitter()
                    fallback_finish_reason = ""
                    for chunk in fallback_result:
                        if cancel_event.is_set():
                            break
                        try:
                            choice = chunk["choices"][0]
                            delta = choice.get("delta", {}) or {}
                            if choice.get("finish_reason"):
                                fallback_finish_reason = str(choice.get("finish_reason"))
                        except Exception:
                            delta = {}
                        reasoning = delta.get("reasoning_content") or delta.get("reasoning") or delta.get("analysis") or ""
                        if reasoning:
                            emit("reasoning", str(reasoning))
                        text = delta.get("content") or ""
                        if text:
                            for piece in fallback_splitter.feed(str(text)):
                                emit(piece.channel, piece.text)
                    for piece in fallback_splitter.finish():
                        emit(piece.channel, piece.text)
                    if fallback_finish_reason:
                        finish_reason = fallback_finish_reason

                resp_q.put({
                    "type": "final", "request_id": rid,
                    "result": "".join(content_parts),
                    "thinking": "".join(reasoning_parts),
                    "cancelled": cancel_event.is_set(),
                    "elapsed": time.perf_counter() - started,
                    "thinking_enabled": native_thinking,
                    "loop_guard_triggered": loop_triggered,
                    "loop_guard_reason": loop_reason,
                    "fallback_used": fallback_used,
                    "system_fallback_used": system_fallback_used,
                    "finish_reason": finish_reason,
                    "tool_calls": [streamed_tool_calls[i] for i in sorted(streamed_tool_calls)],
                })
            else:
                message = result["choices"][0]["message"]
                raw_text = message.get("content") or ""
                native_reasoning = message.get("reasoning_content") or message.get("reasoning") or message.get("analysis") or ""
                if native_reasoning:
                    visible, thinking = str(raw_text), str(native_reasoning)
                else:
                    visible, thinking = split_complete_thinking(str(raw_text), start_in_think=initial_reasoning_for(generation, schema))
                resp_q.put({
                    "type": "final", "request_id": rid, "result": visible,
                    "thinking": thinking, "elapsed": time.perf_counter() - started,
                    "thinking_enabled": native_thinking,
                    "system_fallback_used": system_fallback_used,
                    "tool_calls": list(message.get("tool_calls", []) or []),
                })
        except Exception as exc:
            resp_q.put({
                "type": "error", "request_id": rid, "error": str(exc), "traceback": traceback.format_exc()
            })


class ModelWorker:
    def __init__(self, role: str, config: Dict[str, Any]):
        self.role = role
        self.config = dict(config)
        self.ctx = mp.get_context("spawn")
        self.req_q: mp.Queue = self.ctx.Queue()
        self.resp_q: mp.Queue = self.ctx.Queue()
        self.cancel_event: mp.Event = self.ctx.Event()
        self.process: Optional[mp.Process] = None
        self.loaded = False
        self.error = ""
        self.model_info: Dict[str, Any] = {}
        # Each model process is single-flight. The scheduler can have background and
        # foreground cognition at the same time, so serialize requests per role to
        # prevent one thread from consuming another request's response packets.
        self.request_lock = threading.RLock()

    def start(self, timeout: float = 180.0) -> None:
        if self.process and self.process.is_alive() and self.loaded:
            return
        self.stop()
        # A failed/terminated worker can leave queue messages behind; every start gets fresh IPC.
        self.req_q = self.ctx.Queue()
        self.resp_q = self.ctx.Queue()
        self.cancel_event = self.ctx.Event()
        self.process = self.ctx.Process(
            target=_worker_main,
            args=(self.role, self.config, self.req_q, self.resp_q, self.cancel_event),
            name=f"CognitiveBrain-{self.role}",
            daemon=True,
        )
        self.process.start()
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.process.is_alive():
                self.error = "Worker exited during startup"
                raise RuntimeError(f"{self.role}: {self.error}")
            try:
                msg = self.resp_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if msg.get("type") == "ready":
                self.loaded = True
                self.error = ""
                self.model_info = dict(msg.get("model_info", {}) or {})
                return
            if msg.get("type") == "startup_error":
                self.error = msg.get("error", "unknown startup error")
                raise RuntimeError(f"{self.role}: {self.error}")
        self.error = "Timed out loading model"
        raise TimeoutError(f"{self.role}: {self.error}")

    def stop(self) -> None:
        self.loaded = False
        self.model_info = {}
        if self.process is not None:
            if self.process.is_alive():
                try:
                    self.req_q.put({"op": "shutdown"})
                    self.process.join(timeout=4)
                except Exception:
                    pass
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(timeout=2)
            self.process = None

    def cancel(self) -> None:
        self.cancel_event.set()

    def status(self) -> WorkerStatus:
        alive = bool(self.process and self.process.is_alive())
        info = self.model_info if alive and self.loaded else {}
        return WorkerStatus(
            self.role, alive, bool(alive and self.loaded), str(self.config.get("path", "")), self.error,
            str(info.get("architecture", "")), bool(info.get("chat_template", False)),
            str(info.get("thinking", "")), int(info.get("context", 0) or 0), int(info.get("gpu_layers", 0) or 0),
            str(info.get("preset_key", "")), str(info.get("preset_label", "")), str(info.get("system_mode", "")),
            bool(info.get("native_tools", False)),
        )

    def _request(self, payload: Dict[str, Any], timeout: float, on_token: Optional[Callable[[str], None]] = None,
                 on_reasoning: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        with self.request_lock:
            return self._request_locked(payload, timeout, on_token=on_token, on_reasoning=on_reasoning)

    def _request_locked(self, payload: Dict[str, Any], timeout: float, on_token: Optional[Callable[[str], None]] = None,
                        on_reasoning: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        if not (self.process and self.process.is_alive() and self.loaded):
            raise RuntimeError(f"{self.role} model is not loaded")
        request_id = uuid.uuid4().hex
        payload = dict(payload)
        payload["request_id"] = request_id
        self.req_q.put(payload)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.process.is_alive():
                self.loaded = False
                raise RuntimeError(f"{self.role} worker exited unexpectedly")
            try:
                msg = self.resp_q.get(timeout=min(0.2, max(0.01, deadline - time.time())))
            except queue.Empty:
                continue
            if msg.get("request_id") != request_id:
                # Each worker is intentionally single-flight; unexpected old messages are discarded.
                continue
            if msg.get("type") == "token":
                if on_token:
                    on_token(str(msg.get("text", "")))
                continue
            if msg.get("type") == "reasoning":
                if on_reasoning:
                    on_reasoning(str(msg.get("text", "")))
                continue
            if msg.get("type") == "error":
                raise RuntimeError(f"{self.role} inference failed: {msg.get('error', 'unknown error')}")
            if msg.get("type") == "final":
                return msg
        self.cancel()
        raise TimeoutError(f"{self.role} request exceeded {timeout:.0f}s timeout")

    def chat(self, messages: List[Dict[str, str]], timeout: float, *, json_schema: Optional[Dict[str, Any]] = None,
             stream: bool = False, on_token: Optional[Callable[[str], None]] = None,
             on_reasoning: Optional[Callable[[str], None]] = None,
             generation: Optional[Dict[str, Any]] = None,
             tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        return self._request({
            "op": "chat", "messages": messages, "json_schema": json_schema,
            "stream": stream, "generation": generation or {}, "tools": tools or [],
        }, timeout=timeout, on_token=on_token, on_reasoning=on_reasoning)

    def token_count(self, text: str, timeout: float = 20.0) -> int:
        msg = self._request({"op": "token_count", "text": text}, timeout=timeout)
        return int(msg.get("result", 0))


class WorkerPool:
    ROLES = ("executive", "annotator")

    def __init__(self, model_configs: Dict[str, Dict[str, Any]]):
        self.model_configs = model_configs
        self.workers: Dict[str, ModelWorker] = {}

    def start(self, roles: Optional[Iterable[str]] = None) -> Dict[str, str]:
        errors: Dict[str, str] = {}
        for role in roles or self.ROLES:
            cfg = self.model_configs.get(role, {})
            if not cfg.get("path"):
                errors[role] = "No model configured"
                continue
            worker = self.workers.get(role)
            if worker and worker.config != cfg:
                worker.stop()
                worker = None
            if worker is None:
                worker = ModelWorker(role, cfg)
                self.workers[role] = worker
            try:
                worker.start()
            except Exception as exc:
                errors[role] = str(exc)
        return errors

    def reload(self, model_configs: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        self.stop_all()
        self.model_configs = model_configs
        self.workers = {}
        return self.start()

    def stop_all(self) -> None:
        for worker in list(self.workers.values()):
            worker.stop()

    def get(self, role: str) -> ModelWorker:
        worker = self.workers.get(role)
        if not worker or not worker.status().loaded:
            raise RuntimeError(f"{role} model is not loaded")
        return worker

    def cancel_all(self) -> None:
        for worker in self.workers.values():
            worker.cancel()

    def statuses(self) -> Dict[str, Dict[str, Any]]:
        out = {}
        for role in self.ROLES:
            worker = self.workers.get(role)
            if worker:
                out[role] = worker.status().__dict__
            else:
                cfg = self.model_configs.get(role, {})
                out[role] = WorkerStatus(role, False, False, str(cfg.get("path", "")), "").__dict__
        return out
