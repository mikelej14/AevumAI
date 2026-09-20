from __future__ import annotations

import copy
import pathlib
import sys
import tempfile
import tkinter as tk
from pathlib import Path
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.chats import ChatStore
from core.config import DEFAULT_CONFIG
from ui.main_window import MainWindow


class DummyPool:
    model_configs = {}
    def statuses(self): return {r: {"loaded": False} for r in ("executive", "qwen1", "qwen2")}
    def stop_all(self): pass

class DummyMemory:
    def __init__(self): self.online = True; self.last_error = ""
    def stats(self): return {"records": 0, "terms": 0, "size_bytes": 0, "path": "probe.rmem", "available": self.online, "error": self.last_error}
    def latest(self, count=30, types=None): return []
    def validate(self): return {"ok": self.online, "records_checked": 0, "valid_bytes": 0, "error": self.last_error if not self.online else None}
    def retrieve(self, *a, **k): return []
    def reopen(self): self.online = True; self.last_error = ""; return True

class DummyState:
    def snapshot(self): return {"affect": {"curiosity": 0.5}}

class DummyBrowser:
    config = {}

class DummyScheduler:
    state_engine = DummyState(); browser = DummyBrowser(); config = None
    def background_pending_count(self): return 0
    def cancel(self): pass
    def shutdown(self): pass


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        chats = ChatStore(Path(td) / "chats.json")
        first = chats.active_chat_id()
        chats.add_message(first, "user", "Alpha conversation")
        chats.add_message(first, "assistant", "Alpha answer", thinking="alpha thought", name="Nova")
        second = chats.create_chat("Beta chat")["id"]
        chats.add_message(second, "user", "Beta conversation")
        chats.set_active(first)

        cfg = copy.deepcopy(DEFAULT_CONFIG)
        cfg["identity"]["name"] = "Nova"
        root = tk.Tk(); root.geometry("1180x760+0+0")
        try: root.attributes("-alpha", 0.0)
        except tk.TclError: pass
        app = MainWindow(root, cfg, DummyPool(), DummyMemory(), DummyScheduler(), chats)
        root.update_idletasks(); root.update()

        assert app.current_chat_id == first
        assert "Alpha" in app.chat_title_var.get()
        assert len(app.chat.assistant_views) == 1
        assert app.chat.assistant_views[0].name.cget("text") == "Nova"
        assert app.chat.assistant_views[0].avatar.cget("text") == "N"

        app.switch_chat(second); root.update_idletasks(); root.update()
        assert app.current_chat_id == second
        assert "Beta" in app.chat_title_var.get()
        assert len(app.chat.assistant_views) == 0

        created = chats.create_chat("Third chat")["id"]
        app.current_chat_id = created
        app._refresh_chat_list(); app._load_active_chat(); root.update_idletasks(); root.update()
        assert app.current_chat_id == created
        assert "Third chat" == app.chat_title_var.get()

        # Long sidebar titles must never push the delete control out of the card.
        long_name = "look up the current weather in manchester, nh and compare tomorrow too"
        assert chats.rename_chat(second, long_name)
        app._refresh_chat_list(); root.update_idletasks(); root.update()
        matched = []
        for row, title_widget, delete_widget in app.chat_row_widgets:
            if title_widget.cget("text").startswith("look up the current"):
                matched.append((row, title_widget, delete_widget))
        assert len(matched) == 1
        row, title_widget, delete_widget = matched[0]
        assert title_widget.cget("text").endswith("…")
        assert delete_widget.winfo_ismapped()
        assert delete_widget.winfo_x() + delete_widget.winfo_width() <= row.winfo_width()
        stored_title = chats.get_chat(second)["title"]
        assert stored_title.startswith("look up the current weather")
        assert len(stored_title) > len(title_widget.cget("text"))  # sidebar truncation is extra display-only compaction

        # Delete without dialog interaction and verify another chat becomes active.
        with patch("ui.main_window.messagebox.askyesno", return_value=True):
            app.delete_chat(created)
        root.update_idletasks(); root.update()
        assert chats.get_chat(created) is None
        assert app.current_chat_id != created
        assert chats.get_chat(app.current_chat_id) is not None

        # Resize exercises the left rail and transcript layout.
        root.geometry("1080x700+0+0"); root.update_idletasks(); root.update()
        assert app.sidebar.winfo_width() >= 250
        assert app.chat.canvas.winfo_width() > 500
        print("UI dark-shell/multi-chat probe: PASS")
        root.destroy()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
