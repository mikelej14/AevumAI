from __future__ import annotations

import copy
import pathlib
import sys
import tempfile
import tkinter as tk
from pathlib import Path

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.chats import ChatStore
from core.config import DEFAULT_CONFIG
from ui.main_window import MainWindow
from ui.theme import ElevatedButton


class DummyPool:
    model_configs = {}
    def statuses(self): return {r: {"loaded": False} for r in ("executive", "qwen1", "qwen2")}
    def stop_all(self): pass
    def start(self): return {}

class DummyMemory:
    def __init__(self): self.online=True; self.last_error=""
    def stats(self): return {"records":0,"terms":0,"size_bytes":0,"path":"probe.rmem","available":True,"error":""}
    def latest(self, count=30, types=None): return []
    def validate(self): return {"ok":True,"records_checked":0,"valid_bytes":0}
    def retrieve(self,*a,**k): return []
    def reopen(self): return True

class DummyState:
    def snapshot(self): return {"affect": {}}
class DummyBrowser: config={}
class DummyScheduler:
    state_engine=DummyState(); browser=DummyBrowser(); config=None
    def background_pending_count(self): return 0
    def cancel(self): pass
    def shutdown(self): pass


def _walk(widget):
    yield widget
    for child in widget.winfo_children():
        yield from _walk(child)


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        chats=ChatStore(Path(td)/"chats.json")
        cfg=copy.deepcopy(DEFAULT_CONFIG)
        cfg["identity"]["name"]="Dr. Vale"
        root=tk.Tk(); root.geometry("1180x760+0+0")
        try: root.attributes("-alpha", 0.0)
        except tk.TclError: pass
        app=MainWindow(root,cfg,DummyPool(),DummyMemory(),DummyScheduler(),chats)
        root.update_idletasks(); root.update()

        # Runtime queue code depends on these exact live controls.
        assert isinstance(app.new_chat_btn, ElevatedButton) and callable(app.new_chat_btn.command)
        assert isinstance(app.send_btn, ElevatedButton) and callable(app.send_btn.command)
        assert isinstance(app.stop_btn, ElevatedButton) and callable(app.stop_btn.command)
        assert isinstance(app.start_models_btn, ElevatedButton) and callable(app.start_models_btn.command)
        assert isinstance(app.stop_models_btn, ElevatedButton) and callable(app.stop_models_btn.command)
        assert isinstance(app.rename_btn, ElevatedButton) and callable(app.rename_btn.command)

        # Enabled/disabled states must remain writable because the async runtime toggles them.
        app.send_btn.configure(state="disabled"); assert app.send_btn.cget("state") == "disabled"
        app.send_btn.configure(state="normal"); assert app.send_btn.cget("state") == "normal"
        app.stop_btn.configure(state="normal"); assert app.stop_btn.cget("state") == "normal"
        app.stop_btn.configure(state="disabled"); assert app.stop_btn.cget("state") == "disabled"

        # Every classic Tk/ttk button created by the redesigned UI must retain a callback.
        disconnected=[]
        for w in _walk(root):
            if isinstance(w, (tk.Button,)):
                if not str(w.cget("command")):
                    disconnected.append(str(w))
        assert not disconnected, f"Disconnected Tk buttons: {disconnected}"

        # Navigation remains connected after visual replacement.
        for key, button in app.nav_buttons.items():
            button.invoke(); root.update_idletasks(); root.update()
            assert app.pages[key].winfo_ismapped()

        # Assistant avatar visual identity is derived from the displayed name.
        chat_id=app.current_chat_id
        chats.add_message(chat_id,"assistant","Hello",name="Dr. Vale")
        app._load_active_chat(); root.update_idletasks(); root.update()
        assert app.chat.assistant_views[-1].avatar.cget("text") == "DV"

        print("UI control wiring / dynamic identity probe: PASS")
        root.destroy()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
