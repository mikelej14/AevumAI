from __future__ import annotations

import multiprocessing as mp
import queue
import threading
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import messagebox

from ui.assets import AssetBank
from ui.splash import SplashScreen

STARTUP_LOG = Path(__file__).resolve().parent / "startup_error.log"


def _bootstrap(outq: "queue.Queue[tuple[str, object]]") -> None:
    try:
        outq.put(("status", ("Loading configuration", "Reading Aevum runtime and model settings…")))
        from core.config import CHATS_PATH, NEURAL_MEMORY_DIR, load_config
        cfg = load_config()

        outq.put(("status", ("Opening neural memory", "Loading compact persistent neural engrams…")))
        from core.neural_memory_runtime import NeuralMemoryRuntime
        memory = NeuralMemoryRuntime(NEURAL_MEMORY_DIR, chunk_tokens=int((cfg.get("memory") or {}).get("chunk_tokens", 128)))

        outq.put(("status", ("Restoring conversations", "Loading persistent chat history…")))
        from core.chats import ChatStore
        chats = ChatStore(CHATS_PATH)

        outq.put(("status", ("Preparing cognition", "Creating Granite runtime and optional semantic sidecar…")))
        from core.local_agent_runtime import LocalAgentRuntime
        from core.runtime_adapters import AevumNeuralScheduler, LocalPoolAdapter
        local = LocalAgentRuntime(memory, cfg, data_dir=Path(NEURAL_MEMORY_DIR).parent)
        pool = LocalPoolAdapter(local, cfg)
        scheduler = AevumNeuralScheduler(cfg, local, memory, chats)

        outq.put(("status", ("Building interface", "Finalizing Aevum AI neural memory…")))
        from ui.main_window import MainWindow
        outq.put(("done", (MainWindow, cfg, pool, memory, scheduler, chats)))
    except Exception as exc:
        try:
            STARTUP_LOG.write_text(traceback.format_exc(), encoding="utf-8")
        except Exception:
            pass
        outq.put(("error", exc))


def main() -> None:
    mp.freeze_support()
    root = tk.Tk()
    root.withdraw()
    assets = AssetBank(root)
    splash = SplashScreen(root, assets)
    bootq: "queue.Queue[tuple[str, object]]" = queue.Queue()
    threading.Thread(target=_bootstrap, args=(bootq,), name="aevum-bootstrap", daemon=True).start()

    def poll_boot() -> None:
        try:
            while True:
                kind, payload = bootq.get_nowait()
                if kind == "status":
                    title, detail = payload
                    splash.set_status(str(title), str(detail))
                elif kind == "done":
                    MainWindow, cfg, pool, memory, scheduler, chats = payload
                    MainWindow(root, cfg, pool, memory, scheduler, chats, assets=assets)
                    splash.close()
                    root.deiconify()
                    root.lift()
                    return
                elif kind == "error":
                    splash.close()
                    messagebox.showerror(
                        "Aevum AI startup failed",
                        f"Aevum AI could not finish starting.\n\n{payload}\n\nDetails: {STARTUP_LOG}",
                    )
                    root.destroy()
                    return
        except queue.Empty:
            root.after(40, poll_boot)

    root.after(10, poll_boot)
    root.mainloop()


if __name__ == "__main__":
    main()
