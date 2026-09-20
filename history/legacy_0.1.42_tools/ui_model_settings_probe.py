from __future__ import annotations

import copy
import pathlib
import sys
import tkinter as tk

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import DEFAULT_CONFIG
import ui.main_window as mw
from ui.main_window import MainWindow


class DummyPool:
    model_configs = {}
    def statuses(self):
        return {r: {"loaded": False} for r in ("executive", "qwen1", "qwen2")}
    def stop_all(self): pass


class DummyMemory:
    def stats(self):
        return {"records": 0, "terms": 0, "size_bytes": 0, "path": "probe.rmem"}
    def latest(self, count=30, types=None): return []
    def validate(self): return {"ok": True, "records_checked": 0, "valid_bytes": 0}
    def retrieve(self, *a, **k): return []


class DummyState:
    def snapshot(self): return {"affect": {}}


class DummyBrowser:
    config = {}


class DummyScheduler:
    state_engine = DummyState()
    browser = DummyBrowser()
    config = None
    def history(self, limit=200): return []
    def background_pending_count(self): return 0
    def shutdown(self): pass


def main() -> int:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["models"]["executive"]["path"] = r"C:\\models\\whatever-model.gguf"
    root = tk.Tk(); root.geometry("1000x700+0+0")
    try: root.attributes("-alpha", 0.0)
    except tk.TclError: pass

    old_save = mw.save_config
    old_info = mw.messagebox.showinfo
    old_warn = mw.messagebox.showwarning
    old_err = mw.messagebox.showerror
    mw.save_config = lambda _cfg: None
    mw.messagebox.showinfo = lambda *a, **k: None
    mw.messagebox.showwarning = lambda *a, **k: None
    mw.messagebox.showerror = lambda *a, **k: None
    try:
        app = MainWindow(root, cfg, DummyPool(), DummyMemory(), DummyScheduler())
        root.update_idletasks(); root.update()
        app.save_settings(reload_models=False)
        executive = cfg["models"]["executive"]
        assert executive["path"].endswith("whatever-model.gguf")
        assert "required_architecture" not in executive
        assert "allowed_architectures" not in executive
        assert executive["sampler_profile"] in {"auto", "custom"}
        assert cfg["models"]["qwen1"]["required_architecture"] == "qwen35"
        assert cfg["models"]["qwen2"]["required_architecture"] == "qwen35"
        assert cfg["runtime"]["browser_enabled"] is True
        assert cfg["runtime"]["browser_search_provider"] == "auto"
        assert app.scheduler.browser.config is cfg["runtime"]

        # Auto preset fields follow the selected Executive filename without turning
        # the filename into a permission gate. Runtime metadata can refine this later.
        app.model_entries["executive"].set(r"C:\models\Qwen3.5-2B-Q8_0.gguf")
        app._reset_executive_sampler_fields()
        assert app.executive_sampler_var.get() == "auto"
        assert float(app.setting_vars["executive_temp"].get()) == 1.0
        assert int(app.setting_vars["executive_top_k"].get()) == 20
        assert "Qwen 3.5" in app.executive_detected_var.get()

        app.model_entries["executive"].set(r"C:\models\gemma-2-2b-it-q8_0.gguf")
        app._reset_executive_sampler_fields()
        assert "Gemma 2" in app.executive_detected_var.get()
        assert float(app.setting_vars["executive_repeat"].get()) == 1.1

        print("UI model-settings adaptive Executive preset probe: PASS")
    finally:
        mw.save_config = old_save
        mw.messagebox.showinfo = old_info
        mw.messagebox.showwarning = old_warn
        mw.messagebox.showerror = old_err
        root.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
