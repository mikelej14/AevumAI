from __future__ import annotations
from pathlib import Path
import tkinter as tk

RUNTIME_DIR = Path(__file__).resolve().parent.parent / "assets" / "runtime"

class AssetBank:
    def __init__(self, root: tk.Misc):
        self.root = root
        self._cache: dict[str, tk.PhotoImage] = {}

    def image(self, filename: str) -> tk.PhotoImage:
        img = self._cache.get(filename)
        if img is None:
            img = tk.PhotoImage(master=self.root, file=str(RUNTIME_DIR / filename))
            self._cache[filename] = img
        return img
