from __future__ import annotations
import tkinter as tk
from .assets import AssetBank

class SplashScreen:
    def __init__(self, root: tk.Tk, assets: AssetBank):
        self.root = root
        self.assets = assets
        self.closed = False
        self.win = tk.Toplevel(root)
        self.win.overrideredirect(True)
        self.win.configure(bg="#020503")
        self.win.attributes("-topmost", True)

        frame = tk.Frame(self.win, bg="#050a07", highlightthickness=1, highlightbackground="#255b3b")
        frame.pack(fill="both", expand=True)
        self.logo = assets.image("AevumAI-SPLASH_250x250.png")
        tk.Label(frame, image=self.logo, bg="#050a07").pack(pady=(28, 8))
        tk.Label(frame, text="AEVUM AI", bg="#050a07", fg="#eff8f1", font=("Segoe UI Semibold", 20)).pack()
        tk.Label(frame, text="MEMORY · CONTEXT · CONTINUITY", bg="#050a07", fg="#7da98b", font=("Segoe UI Semibold", 8)).pack(pady=(4, 18))

        status = tk.Frame(frame, bg="#09110c", highlightthickness=1, highlightbackground="#14311f")
        status.pack(fill="x", padx=24, pady=(0, 24))
        self.ind = assets.image("AevumAI-Indicator-Load_40x40.png")
        tk.Label(status, image=self.ind, bg="#09110c").pack(side="left", padx=(12, 8), pady=10)
        self.status_var = tk.StringVar(value="Starting Aevum AI…")
        self.detail_var = tk.StringVar(value="Preparing the cognitive runtime")
        tw = tk.Frame(status, bg="#09110c")
        tw.pack(side="left", fill="x", expand=True, pady=10)
        tk.Label(tw, textvariable=self.status_var, bg="#09110c", fg="#eaf7ee", font=("Segoe UI Semibold", 10), anchor="w").pack(anchor="w")
        tk.Label(tw, textvariable=self.detail_var, bg="#09110c", fg="#789483", font=("Segoe UI", 8), anchor="w").pack(anchor="w", pady=(2,0))

        self._center(390, 455)
        self.win.update_idletasks()

    def _center(self, w, h):
        sw, sh = self.win.winfo_screenwidth(), self.win.winfo_screenheight()
        self.win.geometry(f"{w}x{h}+{max(0,(sw-w)//2)}+{max(0,(sh-h)//2)}")

    def set_status(self, title: str, detail: str = ""):
        if self.closed: return
        self.status_var.set(title)
        if detail: self.detail_var.set(detail)
        self.win.update_idletasks()

    def close(self):
        if self.closed: return
        self.closed = True
        try: self.win.destroy()
        except Exception: pass
