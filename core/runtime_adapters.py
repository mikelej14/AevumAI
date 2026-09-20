from __future__ import annotations

import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

from .config import api_key_from_environment
from .local_agent_runtime import LocalAgentRuntime
from .openai_runtime import OpenAIChatRuntime


def _iso_from_chat_message(msg: dict) -> str:
    try:
        return datetime.fromtimestamp(float(msg.get("timestamp", 0) or 0), timezone.utc).isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()


class LocalPoolAdapter:
    """Small Aevum-UI compatibility adapter around the new two-slot runtime."""

    def __init__(self, local: LocalAgentRuntime, config: dict):
        self.local = local
        self.config = config
        self.model_configs = self.config.get("models", {})

    def start(self):
        results = self.local.load_configured_models()
        errors = {}
        ex = results.get("executive") or {}
        if not ex.get("loaded"):
            errors["executive"] = str(ex.get("error") or "Executive model did not load")
        ann_cfg = (self.config.get("models") or {}).get("annotator") or {}
        ann = results.get("annotator") or {}
        if bool(ann_cfg.get("enabled")) and str(ann_cfg.get("path", "")).strip() and not ann.get("loaded"):
            errors["annotator"] = str(ann.get("error") or "Semantic annotator did not load")
        return errors

    def reload(self, model_configs):
        self.config["models"] = model_configs
        self.model_configs = model_configs
        self.local.reconfigure(self.config)
        self.local.stop()
        return self.start()

    def stop_all(self):
        self.local.stop()

    def cancel_all(self):
        for worker in (self.local.executive, self.local.annotator_worker):
            try:
                if worker:
                    worker.cancel()
            except Exception:
                pass

    def statuses(self):
        return self.local.statuses()


class AevumNeuralScheduler:
    """Aevum turn facade over Granite + neural memory + optional semantic sidecar.

    Visible chat behavior stays compatible with the 0.1.x UI while the old RMEM and
    mandatory 1Qwen/2Qwen pipeline are completely bypassed.
    """

    def __init__(self, config: dict, local: LocalAgentRuntime, memory, chat_store):
        self.config = config
        self.local = local
        self.memory = memory
        self.chat_store = chat_store
        self.state_engine = local.state
        self.browser = local.browser
        self.cancelled = threading.Event()
        self._pending = 0
        self._lock = threading.RLock()
        self.openai = OpenAIChatRuntime(
            memory,
            api_key=api_key_from_environment(),
            model=str((config.get("openai") or {}).get("model", "gpt-5.6-luna")),
            system_prompt=str((config.get("identity") or {}).get("system_prompt", "")),
            user_name=str((config.get("identity") or {}).get("user_name", "User")),
            assistant_name=str((config.get("identity") or {}).get("name", "Assistant")),
            auto_memory_hints=int((config.get("runtime") or {}).get("auto_memory_hints", 6) or 6),
        )

    def reconfigure(self):
        self.local.reconfigure(self.config)
        ident = self.config.get("identity") or {}
        self.openai.configure(
            model=str((self.config.get("openai") or {}).get("model", "gpt-5.6-luna")),
            system_prompt=str(ident.get("system_prompt", "")),
            user_name=str(ident.get("user_name", "User")),
            assistant_name=str(ident.get("name", "Assistant")),
            auto_memory_hints=int((self.config.get("runtime") or {}).get("auto_memory_hints", 6) or 6),
        )

    def background_pending_count(self):
        with self._lock:
            return int(self._pending)

    def _emit(self, cb, name, **data):
        if cb:
            try:
                cb(name, data)
            except Exception:
                pass

    def _blob_activity(self, cb, *, mode: str, memory_id=None, document_id=None, label=""):
        if not cb:
            return
        try:
            blob = None
            if memory_id is not None:
                blob = self.memory.memory_blob(int(memory_id))
            elif document_id is not None:
                blob = self.memory.document_first_blob(int(document_id))
            if blob:
                self._emit(cb, "neural_activity", mode=str(mode), label=str(label or ""), blob=blob)
        except Exception:
            pass

    def _archive_with_activity(self, text, *, role, speaker, chat_id, message_id, timestamp_utc, on_event):
        def progress(_i, _n, info):
            try:
                self._blob_activity(on_event, mode="REMEMBER", memory_id=info.get("id"), label=speaker)
            except Exception:
                pass
        return self.memory.store_message(
            text, role=role, speaker=speaker, chat_id=chat_id,
            message_id=message_id, timestamp_utc=timestamp_utc, progress=progress,
        )

    def _annotate_async(self, document_id: int, text: str, role: str, speaker: str, on_event=None):
        if not bool(((self.config.get("models") or {}).get("annotator") or {}).get("enabled", False)):
            return
        with self._lock:
            self._pending += 1

        def run():
            try:
                ann = self.local.annotate_message(text, role=role, speaker=speaker)
                if ann and document_id is not None:
                    self.memory.annotate_document(int(document_id), ann)
                    self._emit(on_event, "background_complete", kind="semantic_annotation", document_id=int(document_id), pending=max(0, self.background_pending_count()-1))
            except Exception as exc:
                self._emit(on_event, "warning", message=f"Semantic annotation skipped: {exc}")
            finally:
                with self._lock:
                    self._pending = max(0, self._pending - 1)
        threading.Thread(target=run, name="aevum-semantic-annotation", daemon=True).start()

    def run_turn(self, user_text: str, *, on_token=None, on_thought=None, on_event=None,
                 chat_id: str = "", chat_user_message_id: str = "") -> Dict[str, Any]:
        self.cancelled.clear()
        self.reconfigure()
        started = time.perf_counter()
        turn_id = f"turn_{uuid.uuid4().hex}"
        ident = self.config.get("identity") or {}
        user_name = str(ident.get("user_name", "User"))
        assistant_name = str(ident.get("name", "Assistant"))
        recent_turns = max(1, int((self.config.get("runtime") or {}).get("recent_turns", 16) or 16))
        recent = self.chat_store.recent_messages(chat_id, turns=recent_turns)
        provider = str((self.config.get("chat") or {}).get("provider", "local_gguf") or "local_gguf")
        self._emit(on_event, "turn_mode", mode="neural-context")
        self._emit(on_event, "executive_start", pre_exec_seconds=round(time.perf_counter()-started, 4), provider=provider)

        def memory_activity(ev):
            mode = str(ev.get("mode") or "RECALL")
            self._blob_activity(on_event, mode=mode, memory_id=ev.get("memory_id"), document_id=ev.get("document_id"), label=ev.get("label", ""))

        def tool_cb(ev):
            self._emit(on_event, "executive_tool", **dict(ev))
            action = str(ev.get("action", ""))
            if action == "neural_memory_search":
                for d in list(ev.get("documents") or [])[:3]:
                    mids = list(d.get("matched_memory_ids") or [])
                    self._blob_activity(on_event, mode="RECALL", memory_id=(mids[0] if mids else None), document_id=d.get("document_id"), label="search")
            elif action == "neural_memory_open":
                self._blob_activity(on_event, mode="RECALL", document_id=ev.get("document_id"), label="open")

        if provider == "openai":
            result = self.openai.respond(
                recent,
                on_tool=tool_cb,
                max_tool_rounds=max(1, int((self.config.get("runtime") or {}).get("tool_rounds", 4) or 4)),
            )
            assistant_text = str(result.get("text", "") or "")
            thinking = ""
            tool_events = list(result.get("tools") or [])
            memory_notes = list(result.get("memory_notes") or [])
        else:
            result = self.local.respond(
                recent, on_tool=tool_cb, on_token=on_token, on_reasoning=on_thought,
                on_memory_activity=memory_activity,
            )
            assistant_text = str(result.get("text", "") or "")
            thinking = str(result.get("thinking", "") or "")
            tool_events = list(result.get("tools") or [])
            memory_notes = list(result.get("memory_notes") or [])

        if self.cancelled.is_set():
            raise RuntimeError("Turn cancelled")
        if not assistant_text.strip():
            raise RuntimeError("Executive model returned an empty response")

        # Persist the visible assistant reply first so the chat always has the full
        # answer even if neural archival is interrupted later.
        assistant_mid = self.chat_store.add_message(
            chat_id, "assistant", assistant_text, thinking=thinking,
            tools=tool_events, name=assistant_name, turn_id=turn_id,
        )

        # Archive the current user/assistant messages only after the answer is complete.
        # This prevents current-question self-retrieval during generation.
        user_msg = next((m for m in self.chat_store.messages(chat_id) if m.get("id") == chat_user_message_id), None) or {}
        user_info = self._archive_with_activity(
            user_text, role="user", speaker=user_name, chat_id=chat_id,
            message_id=chat_user_message_id, timestamp_utc=_iso_from_chat_message(user_msg), on_event=on_event,
        )
        assistant_msg = next((m for m in self.chat_store.messages(chat_id) if m.get("id") == assistant_mid), None) or {}
        assistant_info = self._archive_with_activity(
            assistant_text, role="assistant", speaker=assistant_name, chat_id=chat_id,
            message_id=assistant_mid, timestamp_utc=_iso_from_chat_message(assistant_msg), on_event=on_event,
        )
        self._annotate_async(int(user_info.get("document_id")), user_text, "user", user_name, on_event)
        self._annotate_async(int(assistant_info.get("document_id")), assistant_text, "assistant", assistant_name, on_event)

        # Tool receipts are compact by design: query/purpose/result list only. Full
        # fetched web pages and recalled neural documents are private working evidence.
        for note in memory_notes:
            try:
                text = str(note.get("text", "") or "").strip()
                if text:
                    self.memory.store_tool_note(text, tool=str(note.get("tool", "tool")), chat_id=chat_id, turn_id=turn_id)
            except Exception:
                pass

        pending = self.background_pending_count()
        self._emit(on_event, "visible_complete", assistant_text=assistant_text, background_pending=pending,
                   visible_seconds=round(time.perf_counter()-started, 4), turn_id=turn_id)
        return {
            "assistant_text": assistant_text,
            "thinking": thinking,
            "turn_id": turn_id,
            "chat_id": chat_id,
            "chat_assistant_message_id": assistant_mid,
            "background_pending": pending > 0,
            "scheduling_mode": "neural-context",
            "trace": {"provider": provider, "tools": tool_events, "neural_user_document": user_info.get("document_id"),
                      "neural_assistant_document": assistant_info.get("document_id")},
        }

    def cancel(self):
        self.cancelled.set()
        for worker in (self.local.executive, self.local.annotator_worker):
            try:
                if worker:
                    worker.cancel()
            except Exception:
                pass

    def shutdown(self):
        self.cancel()
        self.local.stop()
