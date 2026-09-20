from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import ttk
import webbrowser

from .theme import BG, SURFACE, TEXT, TEXT_MUTED, ACCENT, ERROR, FONT_SMALL, FONT_SECTION, ElevatedButton

GRANITE_URL = "https://huggingface.co/ibm-granite/granite-4.2-3b-GGUF/tree/main"
QWEN_URL = "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF"


def model_file_error(value: str, label: str = "Executive") -> str:
    path = Path(value.strip()).expanduser()
    if not value.strip():
        return f"Choose your {label} .gguf file with Browse."
    if path.suffix.lower() != ".gguf":
        return f"Select a .gguf model file for {label}, not a folder or download page."
    try:
        with path.open("rb") as model:
            if model.read(4) != b"GGUF":
                return f"The selected {label} file is not a GGUF model. Check that the download finished."
    except (OSError, ValueError):
        return f"The selected {label} file could not be opened. Browse to its current location."
    return ""


def needs_model_setup(config: dict) -> bool:
    if config.get("chat", {}).get("provider", "local_gguf") != "local_gguf":
        return False
    path = str(config.get("models", {}).get("executive", {}).get("path", ""))
    return bool(model_file_error(path))


class QuickStartPanel(tk.Frame):
    """A short local-model setup flow sharing the existing Settings variables."""

    def __init__(self, parent, *, model_vars, annotator_var, browse, load, chat, settings):
        super().__init__(parent, bg=BG)
        self.status = tk.StringVar(value="Choose an Executive model to get started.")
        self._ready = False

        footer = tk.Frame(self, bg=SURFACE, padx=18, pady=12)
        footer.pack(side="bottom", fill="x", padx=22, pady=(0, 18))
        self.status_label = tk.Label(footer, textvariable=self.status, bg=SURFACE,
                                    fg=TEXT_MUTED, font=FONT_SMALL, anchor="w",
                                    justify="left", wraplength=680)
        self.status_label.pack(fill="x", pady=(0, 9))
        buttons = tk.Frame(footer, bg=SURFACE)
        buttons.pack(fill="x")
        self.load_button = ElevatedButton(buttons, text="Save & load models", command=load, variant="primary")
        self.load_button.pack(side="left", padx=(0, 8))
        self.chat_button = ElevatedButton(buttons, text="Open chat", command=chat, variant="secondary", state="disabled")
        self.chat_button.pack(side="left")
        ElevatedButton(buttons, text="Advanced settings", command=settings, variant="ghost").pack(side="right")

        canvas = tk.Canvas(self, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y", padx=(0, 8), pady=18)
        canvas.pack(fill="both", expand=True, padx=(22, 0), pady=18)
        body = tk.Frame(canvas, bg=BG)
        window = canvas.create_window((0, 0), window=body, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        body.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        self._wrapping_labels = [self.status_label]

        def resize(event):
            canvas.itemconfigure(window, width=event.width)
            for label in self._wrapping_labels:
                label.configure(wraplength=max(220, event.width - 40))
        canvas.bind("<Configure>", resize)

        def card(title, description):
            frame = tk.Frame(body, bg=SURFACE, padx=18, pady=14)
            frame.pack(fill="x", pady=(0, 12))
            tk.Label(frame, text=title, bg=SURFACE, fg=TEXT, font=FONT_SECTION).pack(anchor="w")
            label = tk.Label(frame, text=description, bg=SURFACE, fg=TEXT_MUTED,
                             font=FONT_SMALL, justify="left", wraplength=680)
            label.pack(anchor="w", fill="x", pady=(5, 10))
            self._wrapping_labels.append(label)
            return frame

        executive = card("1  Choose your main model",
                         "Download IBM Granite 4.2 3B Q6_K, then Browse to granite-4.2-3b-Q6_K.gguf. "
                         "Allow roughly 6 GB of VRAM; 8 GB or more gives headroom. CPU inference uses system RAM.")
        ElevatedButton(executive, text="Get Granite on Hugging Face",
                       command=lambda: webbrowser.open(GRANITE_URL), variant="secondary").pack(anchor="w", pady=(0, 10))
        self._model_row(executive, model_vars["executive"], lambda: browse("executive"))

        annotator = card("2  Add Qwen (optional)",
                         "Qwen3.5-2B adds topic and sentiment annotations. Leave it off for the simplest first run; "
                         "you can add it later. It uses additional system RAM.")
        ElevatedButton(annotator, text="Get Qwen GGUF on Hugging Face",
                       command=lambda: webbrowser.open(QWEN_URL), variant="secondary").pack(anchor="w", pady=(0, 10))
        self._model_row(annotator, model_vars["annotator"], lambda: browse("annotator"))
        ttk.Checkbutton(annotator, text="Enable optional Qwen annotator",
                        variable=annotator_var).pack(anchor="w", pady=(10, 0))

        card("3  Save, load, then chat",
             "Click Save & load models below. Keep the app open while it loads. "
             "When the status says Ready, click Open chat and send your first message. "
             "Your choices are also available in Settings.")

    @staticmethod
    def _model_row(parent, variable, browse):
        row = tk.Frame(parent, bg=SURFACE)
        row.pack(fill="x")
        ElevatedButton(row, text="Browse", command=browse, variant="secondary").pack(side="right", padx=(8, 0))
        ttk.Entry(row, textvariable=variable).pack(side="left", fill="x", expand=True)

    def set_loading(self, loading: bool):
        self.load_button.configure(state="disabled" if loading else "normal")
        self.chat_button.configure(state="disabled" if loading or not self._ready else "normal")
        if loading:
            self.status.set("Loading models… This may take a while. You can follow details in Activity.")
            self.status_label.configure(fg=TEXT_MUTED)

    def set_result(self, ready: bool, error: str = "", annotator_unavailable: bool = False):
        self._ready = ready
        self.set_loading(False)
        if ready:
            note = " Optional Qwen is unavailable; you can still chat." if annotator_unavailable else ""
            self.status.set("Ready — your Executive is loaded. Click Open chat." + note)
            self.status_label.configure(fg=ACCENT)
        else:
            detail = error if len(error) <= 300 else error[:300] + "… See Activity for details."
            self.status.set(detail or "Model stopped. Click Save & load models when you are ready.")
            self.status_label.configure(fg=ERROR if error else TEXT_MUTED)
