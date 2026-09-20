from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


CHAT_STORE_VERSION = 1


def _now() -> float:
    return time.time()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _clean_title(text: str, limit: int = 52) -> str:
    text = " ".join(str(text or "").split()).strip()
    if not text:
        return "New chat"
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)].rstrip() + "…"


class ChatStore:
    """Durable multi-chat transcript store, intentionally independent from neural memory.

    Neural memory is long-term lived experience. This file is the canonical source for
    visible chat/session transcripts, so chats survive a neural-memory reset/loss and deleting
    a chat never requires rewriting the append-only memory pack.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.warning: str = ""
        self.data: Dict[str, Any] = {"version": CHAT_STORE_VERSION, "active_chat_id": "", "legacy_import_done": False, "chats": []}
        self._load()
        if not self.data.get("chats"):
            chat = self._create_chat_locked("New chat")
            self.data["active_chat_id"] = chat["id"]
            self._save_locked()
        elif not self.get_chat(self.data.get("active_chat_id", "")):
            self.data["active_chat_id"] = self.data["chats"][0]["id"]
            self._save_locked()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or int(raw.get("version", 0) or 0) != CHAT_STORE_VERSION:
                raise ValueError("unsupported chat-store schema")
            chats = raw.get("chats")
            if not isinstance(chats, list):
                raise ValueError("chat list is missing")
            cleaned = []
            for c in chats:
                if not isinstance(c, dict) or not str(c.get("id", "")).strip():
                    continue
                messages = c.get("messages", [])
                if not isinstance(messages, list):
                    messages = []
                cleaned.append({
                    "id": str(c["id"]),
                    "title": _clean_title(c.get("title") or "New chat"),
                    "created_at": float(c.get("created_at", _now()) or _now()),
                    "updated_at": float(c.get("updated_at", c.get("created_at", _now())) or _now()),
                    "messages": [dict(m) for m in messages if isinstance(m, dict)],
                })
            self.data = {
                "version": CHAT_STORE_VERSION,
                "active_chat_id": str(raw.get("active_chat_id", "") or ""),
                "legacy_import_done": bool(raw.get("legacy_import_done", False)),
                "chats": cleaned,
            }
        except Exception as exc:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            backup = self.path.with_name(f"{self.path.stem}.corrupt-{stamp}{self.path.suffix}")
            try:
                self.path.replace(backup)
                self.warning = f"Chat store was unreadable and was preserved as {backup.name}. A fresh chat list was created. ({exc})"
            except Exception:
                self.warning = f"Chat store was unreadable. A fresh in-memory chat list was created. ({exc})"
            self.data = {"version": CHAT_STORE_VERSION, "active_chat_id": "", "legacy_import_done": False, "chats": []}

    def _save_locked(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        text = json.dumps(self.data, ensure_ascii=False, separators=(",", ":"))
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(self.path)

    def _create_chat_locked(self, title: str = "New chat") -> Dict[str, Any]:
        now = _now()
        chat = {"id": _new_id("chat"), "title": _clean_title(title), "created_at": now, "updated_at": now, "messages": []}
        self.data["chats"].insert(0, chat)
        return chat

    def create_chat(self, title: str = "New chat") -> Dict[str, Any]:
        with self.lock:
            chat = self._create_chat_locked(title)
            self.data["active_chat_id"] = chat["id"]
            self._save_locked()
            return dict(chat)

    def list_chats(self) -> List[Dict[str, Any]]:
        with self.lock:
            chats = sorted(self.data.get("chats", []), key=lambda c: float(c.get("updated_at", 0)), reverse=True)
            return [{k: v for k, v in c.items() if k != "messages"} | {"message_count": len(c.get("messages", []))} for c in chats]

    def get_chat(self, chat_id: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            for chat in self.data.get("chats", []):
                if chat.get("id") == chat_id:
                    return chat
            return None

    def active_chat_id(self) -> str:
        with self.lock:
            return str(self.data.get("active_chat_id", "") or "")

    def set_active(self, chat_id: str) -> bool:
        with self.lock:
            if not self.get_chat(chat_id):
                return False
            self.data["active_chat_id"] = chat_id
            self._save_locked()
            return True

    def rename_chat(self, chat_id: str, title: str) -> bool:
        with self.lock:
            chat = self.get_chat(chat_id)
            if not chat:
                return False
            chat["title"] = _clean_title(title)
            chat["updated_at"] = _now()
            self._save_locked()
            return True

    def delete_chat(self, chat_id: str) -> Optional[str]:
        """Delete transcript/session metadata only. Neural memory is intentionally untouched.

        Returns the new active chat id.
        """
        with self.lock:
            chats = self.data.get("chats", [])
            idx = next((i for i, c in enumerate(chats) if c.get("id") == chat_id), None)
            if idx is None:
                return self.active_chat_id()
            chats.pop(idx)
            if not chats:
                replacement = self._create_chat_locked("New chat")
                new_active = replacement["id"]
            else:
                ordered = sorted(chats, key=lambda c: float(c.get("updated_at", 0)), reverse=True)
                new_active = ordered[0]["id"]
            if self.data.get("active_chat_id") == chat_id or not self.get_chat(self.data.get("active_chat_id", "")):
                self.data["active_chat_id"] = new_active
            self._save_locked()
            return str(self.data.get("active_chat_id", "") or new_active)

    def add_message(self, chat_id: str, role: str, content: str, *, thinking: str = "",
                    tools: Optional[List[Dict[str, Any]]] = None, name: str = "",
                    turn_id: str = "", rmem_record_id: int = 0) -> str:
        role = str(role or "").strip().lower()
        if role not in {"user", "assistant"}:
            raise ValueError("chat messages must be user or assistant")
        with self.lock:
            chat = self.get_chat(chat_id)
            if not chat:
                raise KeyError(f"Unknown chat id: {chat_id}")
            msg_id = _new_id("msg")
            msg = {
                "id": msg_id,
                "role": role,
                "content": str(content or ""),
                "timestamp": _now(),
            }
            if thinking:
                msg["thinking"] = str(thinking)
            if tools:
                msg["tools"] = [dict(item) for item in tools if isinstance(item, dict)]
            if name:
                msg["name"] = str(name)
            if turn_id:
                msg["turn_id"] = str(turn_id)
            if rmem_record_id:
                msg["rmem_record_id"] = int(rmem_record_id)
            chat.setdefault("messages", []).append(msg)
            chat["updated_at"] = msg["timestamp"]
            if role == "user" and chat.get("title") in {"", "New chat"}:
                chat["title"] = _clean_title(content)
            self.data["active_chat_id"] = chat_id
            self._save_locked()
            return msg_id

    def update_message(self, chat_id: str, message_id: str, **updates: Any) -> bool:
        allowed = {"content", "thinking", "tools", "name", "turn_id", "rmem_record_id"}
        with self.lock:
            chat = self.get_chat(chat_id)
            if not chat:
                return False
            for msg in chat.get("messages", []):
                if msg.get("id") == message_id:
                    for key, value in updates.items():
                        if key in allowed:
                            msg[key] = value
                    chat["updated_at"] = _now()
                    self._save_locked()
                    return True
            return False

    def messages(self, chat_id: str) -> List[Dict[str, Any]]:
        with self.lock:
            chat = self.get_chat(chat_id)
            return [dict(m) for m in chat.get("messages", [])] if chat else []

    def recent_messages(self, chat_id: str, turns: int = 8, *, before_message_id: str = "") -> List[Dict[str, str]]:
        with self.lock:
            messages = self.messages(chat_id)
            if before_message_id:
                cut = next((i for i, m in enumerate(messages) if m.get("id") == before_message_id), len(messages))
                messages = messages[:cut]
            out = [
                {"role": str(m.get("role")), "content": str(m.get("content", ""))}
                for m in messages
                if m.get("role") in {"user", "assistant"} and str(m.get("content", "")).strip()
            ]
            return out[-max(0, int(turns)) * 2:]

    def needs_legacy_import(self) -> bool:
        with self.lock:
            return not bool(self.data.get("legacy_import_done", False))

    def mark_legacy_import_done(self) -> None:
        with self.lock:
            self.data["legacy_import_done"] = True
            self._save_locked()

    def has_messages(self) -> bool:
        with self.lock:
            return any(c.get("messages") for c in self.data.get("chats", []))


class InMemoryChatStore(ChatStore):
    """Non-persistent store for tests/embedding fallbacks."""
    def __init__(self):
        self.path = Path("<memory>")
        self.lock = threading.RLock()
        self.warning = ""
        self.data = {"version": CHAT_STORE_VERSION, "active_chat_id": "", "legacy_import_done": True, "chats": []}
        chat = self._create_chat_locked("New chat")
        self.data["active_chat_id"] = chat["id"]

    def _save_locked(self) -> None:
        return
