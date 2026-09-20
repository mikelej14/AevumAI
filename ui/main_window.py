from __future__ import annotations

import json
import queue
import threading
import time
from datetime import datetime, timezone
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any, Dict

from core.chats import InMemoryChatStore
from core.config import normalize_identity_name, save_config
from core.model_presets import GENERIC_TEXT as GENERIC_EXECUTIVE_TEXT
from core.model_presets import detect_preset_from_filename, preset_by_key, preset_summary, sampler_profile
from core.qwen35_profile import apply_sampler_profile
from .assets import AssetBank
from .chat_widgets import AssistantTurnView, ChatTranscript
from .neural_minimap import NeuralMiniMap
from .quickstart import QuickStartPanel, model_file_error, needs_model_setup
from .theme import (
    ACCENT, ACCENT_DARK, ACCENT_GLOW, BG, BORDER, BORDER_BRIGHT, BORDER_SOFT, ERROR, FONT, FONT_MONO, FONT_SECTION, FONT_SMALL, FONT_TITLE,
    SIDEBAR, SIDEBAR_2, SHADOW, SHADOW_SOFT, SUCCESS, SURFACE, SURFACE_2, SURFACE_3, SURFACE_4, TEXT, TEXT_DIM, TEXT_MUTED,
    WARNING, ElevatedButton, SurfaceCard, apply_theme, assistant_initials, style_text,
)


VERSION = "0.2.4"


def _commit_final_answer(view: AssistantTurnView | None, payload: Dict[str, Any] | None) -> str:
    """Commit the scheduler's authoritative final answer into a chat card.

    Token callbacks are a live preview only. A successful turn may legally buffer or
    recover its answer, so completion must always synchronize from assistant_text.
    """
    if view is None:
        return ""
    data = payload if isinstance(payload, dict) else {}
    final_answer = str(data.get("assistant_text", "") or "")
    view.reconcile_answer(final_answer)
    view.finish()
    return final_answer


class MainWindow:
    def __init__(self, root: tk.Tk, config: Dict[str, Any], pool, memory, scheduler, chats=None, assets: AssetBank | None = None):
        self.root = root
        self.config = config
        self.pool = pool
        self.memory = memory
        self.scheduler = scheduler
        self.chats = chats or InMemoryChatStore()
        self.assets = assets or AssetBank(root)
        self._img: Dict[str, tk.PhotoImage] = {}
        self.uiq: queue.Queue = queue.Queue()
        self.busy = False
        self.models_loading = False
        self.streaming = False
        self.current_assistant: AssistantTurnView | None = None
        self.current_chat_id = self.chats.active_chat_id()
        self.model_entries: Dict[str, tk.StringVar] = {}
        self.setting_vars: Dict[str, tk.StringVar] = {}
        self.pages: Dict[str, tk.Frame] = {}
        self.nav_buttons: Dict[str, tk.Button] = {}
        self.chat_row_widgets = []

        apply_theme(root)
        root.title(f"Aevum AI {VERSION} — {normalize_identity_name(self.config['identity'].get('name', 'Assistant'))}")
        root.geometry("1380x860")
        root.minsize(1080, 700)
        try:
            root.iconphoto(True, self.assets.image("AevumAI-BadgeLOGO_64x64.png"))
        except Exception:
            pass
        self._build()
        self.root.bind("<Control-n>", lambda _e: self.new_chat())
        self.root.bind("<Control-comma>", lambda _e: self.show_page("settings"))
        self._refresh_chat_list()
        self._load_active_chat()
        self._refresh_state()
        self._refresh_memory()
        self._refresh_status()
        if needs_model_setup(self.config):
            self.show_page("quickstart")
        self.root.after(80, self._poll_uiq)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        if getattr(self.chats, "warning", ""):
            self.root.after(250, lambda: messagebox.showwarning("Chat recovery", self.chats.warning))
        self.root.after(450, self._reconcile_chat_memory)
        if self.config["runtime"].get("auto_start_models") and not needs_model_setup(self.config):
            self.start_models()

    # ---------- shell / theme ----------
    def _build(self):
        shell = tk.Frame(self.root, bg=BG)
        shell.pack(fill="both", expand=True)

        self.sidebar = tk.Frame(shell, bg=SIDEBAR, width=278)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        self.main = tk.Frame(shell, bg=BG)
        self.main.pack(side="left", fill="both", expand=True)

        self._build_sidebar()
        self._build_main_header()
        self.content = tk.Frame(self.main, bg=BG)
        self.content.pack(fill="both", expand=True)

        for key in ("chat", "state", "memory", "activity", "settings", "quickstart"):
            frame = tk.Frame(self.content, bg=BG)
            self.pages[key] = frame
            frame.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.chat_tab = self.pages["chat"]
        self.state_tab = self.pages["state"]
        self.memory_tab = self.pages["memory"]
        self.activity_tab = self.pages["activity"]
        self.settings_tab = self.pages["settings"]

        self._build_chat()
        self._build_state()
        self._build_memory()
        self._build_activity()
        self._build_settings()
        self.quickstart = QuickStartPanel(
            self.pages["quickstart"], model_vars=self.model_entries,
            annotator_var=self.annotator_enabled_var, browse=self._browse_model,
            load=self._quickstart_load, chat=lambda: self.show_page("chat"),
            settings=lambda: self.show_page("settings"),
        )
        self.quickstart.pack(fill="both", expand=True)
        self.show_page("chat")

    def _build_sidebar(self):
        brand_shell = tk.Frame(self.sidebar, bg=SHADOW_SOFT)
        brand_shell.pack(fill="x", padx=10, pady=(10, 8))
        brand = tk.Frame(brand_shell, bg=SIDEBAR_2, highlightthickness=1, highlightbackground=BORDER)
        brand.pack(fill="x", padx=(0, 1), pady=(0, 2))
        self._img["brand"] = self.assets.image("AevumAI-BadgeLOGO_48x48.png")
        tk.Label(brand, image=self._img["brand"], bg=SIDEBAR_2).pack(side="left", padx=(11, 10), pady=10)
        brand_text = tk.Frame(brand, bg=SIDEBAR_2)
        brand_text.pack(side="left", fill="x", expand=True, pady=9)
        tk.Label(brand_text, text="Aevum AI", bg=SIDEBAR_2, fg=TEXT, font=("Segoe UI Semibold", 13)).pack(anchor="w")
        tk.Label(brand_text, text="MEMORY · CONTEXT · CONTINUITY", bg=SIDEBAR_2, fg=TEXT_DIM,
                 font=("Segoe UI Semibold", 7)).pack(anchor="w", pady=(2, 0))
        tk.Label(brand_text, text=f"v{VERSION}", bg=SIDEBAR_2, fg=ACCENT, font=("Segoe UI Semibold", 8)).pack(anchor="w", pady=(3, 0))

        self._img["new_chat"] = self.assets.image("AevumAI-Chat_18x18.png")
        self.new_chat_btn = ElevatedButton(self.sidebar, text="New chat", image=self._img["new_chat"],
                                           command=self.new_chat, variant="primary", pady=8, anchor="center")
        self.new_chat_btn.pack(fill="x", padx=14, pady=(4, 10))

        search_shell = tk.Frame(self.sidebar, bg=BORDER_SOFT)
        search_shell.pack(fill="x", padx=14, pady=(0, 10))
        search_inner = tk.Frame(search_shell, bg=SURFACE, highlightthickness=0)
        search_inner.pack(fill="x", padx=1, pady=1)
        tk.Label(search_inner, text="⌕", bg=SURFACE, fg=TEXT_DIM, font=("Segoe UI", 12)).pack(side="left", padx=(9, 4))
        self.chat_filter_var = tk.StringVar(value="")
        self.chat_filter_entry = tk.Entry(search_inner, textvariable=self.chat_filter_var, bg=SURFACE, fg=TEXT,
                                          insertbackground=TEXT, relief="flat", bd=0, highlightthickness=0,
                                          font=("Segoe UI", 9))
        self.chat_filter_entry.pack(side="left", fill="x", expand=True, padx=(0, 8), pady=7)
        self.chat_filter_entry.bind("<KeyRelease>", lambda _e: self._refresh_chat_list())

        tk.Label(self.sidebar, text="CHATS", bg=SIDEBAR, fg=TEXT_DIM, font=("Segoe UI Semibold", 8), anchor="w").pack(fill="x", padx=18, pady=(1, 5))
        chat_wrap = tk.Frame(self.sidebar, bg=SIDEBAR)
        chat_wrap.pack(fill="both", expand=True, padx=(8, 4))
        self.chat_list_canvas = tk.Canvas(chat_wrap, bg=SIDEBAR, highlightthickness=0, borderwidth=0)
        self.chat_list_scroll = ttk.Scrollbar(chat_wrap, orient="vertical", command=self.chat_list_canvas.yview)
        self.chat_list_inner = tk.Frame(self.chat_list_canvas, bg=SIDEBAR)
        self.chat_list_window = self.chat_list_canvas.create_window((0, 0), window=self.chat_list_inner, anchor="nw")
        self.chat_list_canvas.configure(yscrollcommand=self.chat_list_scroll.set)
        self.chat_list_canvas.pack(side="left", fill="both", expand=True)
        self.chat_list_scroll.pack(side="right", fill="y")
        self.chat_list_inner.bind("<Configure>", lambda _e: self.chat_list_canvas.configure(scrollregion=self.chat_list_canvas.bbox("all")))
        self.chat_list_canvas.bind("<Configure>", lambda e: self.chat_list_canvas.itemconfigure(self.chat_list_window, width=e.width))
        self.chat_list_canvas.bind("<MouseWheel>", self._sidebar_wheel)
        self.chat_list_inner.bind("<MouseWheel>", self._sidebar_wheel)

        nav_sep = tk.Frame(self.sidebar, bg=BORDER_SOFT, height=1)
        nav_sep.pack(fill="x", padx=12, pady=(6, 7))
        nav = tk.Frame(self.sidebar, bg=SIDEBAR)
        nav.pack(fill="x", padx=8, pady=(0, 8))
        nav_items = (
            ("quickstart", "AevumAI-Settings_20x20.png", "Quick start"),
            ("chat", "AevumAI-Chat_20x20.png", "Chat"),
            ("state", "AevumAI-State_20x20.png", "Cognitive state"),
            ("memory", "AevumAI-Memory_20x20.png", "Memory"),
            ("activity", "AevumAI-Config_20x20.png", "Activity"),
            ("settings", "AevumAI-Settings_20x20.png", "Settings"),
        )
        for key, icon_file, label in nav_items:
            self._img[f"nav_{key}"] = self.assets.image(icon_file)
            b = tk.Button(nav, text=f"  {label}", image=self._img[f"nav_{key}"], compound="left",
                          command=lambda k=key: self.show_page(k), bg=SIDEBAR, fg=TEXT_MUTED,
                          activebackground=SURFACE_2, activeforeground=TEXT, relief="flat", bd=0,
                          highlightthickness=0, font=("Segoe UI", 10), anchor="w", padx=8, pady=8, cursor="hand2")
            b.pack(fill="x", pady=1)
            self.nav_buttons[key] = b

        footer = tk.Frame(self.sidebar, bg=SIDEBAR)
        footer.pack(fill="x", padx=14, pady=(0, 12))
        self.memory_badge = tk.Label(footer, text="NEURAL", bg=SURFACE_2, fg=TEXT_MUTED,
                                     font=("Segoe UI Semibold", 8), padx=9, pady=4,
                                     highlightthickness=1, highlightbackground=BORDER_SOFT)
        self.memory_badge.pack(side="left")
        tk.Label(footer, text=f"v{VERSION}", bg=SIDEBAR, fg=TEXT_DIM, font=("Segoe UI", 8)).pack(side="right")

    def _sidebar_wheel(self, event):
        units = int(-event.delta / 120) if event.delta else 0
        if units:
            self.chat_list_canvas.yview_scroll(units * 3, "units")
        return "break"

    def _build_main_header(self):
        # Keep the Aevum 0.1.x header geometry/branding, with one compact neural
        # minimap added at the far right like an RTS status map.
        header_shadow = tk.Frame(self.main, bg=SHADOW, height=92)
        header_shadow.pack(fill="x", padx=16, pady=(14, 5))
        header_shadow.pack_propagate(False)
        header = tk.Frame(header_shadow, bg=SURFACE, highlightthickness=1, highlightbackground=BORDER_SOFT)
        header.pack(fill="both", expand=True, padx=(0, 1), pady=(0, 3))
        header.columnconfigure(0, weight=1)
        header.rowconfigure(0, weight=1)

        left = tk.Frame(header, bg=SURFACE)
        left.grid(row=0, column=0, sticky="nsew", padx=(18, 8))
        self.page_title_var = tk.StringVar(value="Chat")
        self.page_subtitle_var = tk.StringVar(value="")
        tk.Label(left, textvariable=self.page_title_var, bg=SURFACE, fg=TEXT,
                 font=("Segoe UI Semibold", 14), anchor="w").pack(anchor="w", pady=(16, 0))
        tk.Label(left, textvariable=self.page_subtitle_var, bg=SURFACE, fg=TEXT_MUTED,
                 font=FONT_SMALL, anchor="w", wraplength=410, justify="left").pack(anchor="w", pady=(2, 0))

        right = tk.Frame(header, bg=SURFACE)
        right.grid(row=0, column=1, sticky="e", padx=(8, 12), pady=8)
        status_pill = tk.Frame(right, bg=SURFACE_2, highlightthickness=1, highlightbackground=BORDER_SOFT)
        status_pill.pack(side="left", padx=(0, 8))
        self._img["model_status"] = self.assets.image("AevumAI-Indicator-OFF_18x18.png")
        self.status_dot = tk.Label(status_pill, image=self._img["model_status"], bg=SURFACE_2)
        self.status_dot.pack(side="left", padx=(8, 4), pady=6)
        self.status_var = tk.StringVar(value="Model stopped")
        tk.Label(status_pill, textvariable=self.status_var, bg=SURFACE_2, fg=TEXT_MUTED,
                 font=FONT_SMALL).pack(side="left", padx=(0, 8), pady=6)
        self.start_models_btn = ElevatedButton(right, text="Start model", command=self.start_models,
                                               variant="primary", pady=6, padx=10)
        self.start_models_btn.pack(side="left", padx=(0, 6))
        self.stop_models_btn = ElevatedButton(right, text="Stop", command=self.stop_models,
                                              variant="secondary", pady=6, padx=10)
        self.stop_models_btn.pack(side="left", padx=(0, 9))

        ui_cfg = self.config.get("ui") or {}
        self.neural_minimap = NeuralMiniMap(
            right,
            speed_ms=int(ui_cfg.get("neural_minimap_speed_ms", 50) or 50),
            max_frames=int(ui_cfg.get("neural_minimap_frames", 48) or 48),
            quality=str(ui_cfg.get("neural_minimap_quality", "smooth") or "smooth"),
            width=192,
            height=72,
        )
        self.neural_minimap.pack(side="left", fill="y")

    def show_page(self, key: str):
        if key not in self.pages:
            return
        self.pages[key].tkraise()
        self.current_page = key
        labels = {
            "quickstart": ("Welcome to Aevum", "Choose a model · Save and load · Start a conversation"),
            "chat": (self._active_chat_title(), "Persistent conversation with neural continuity"),
            "state": ("Cognitive state", "Optional semantic context layered over raw neural memory"),
            "memory": ("Neural memory", "Search and inspect persistent compact neural experience"),
            "activity": ("Activity", "Runtime events, memory recall, tools, and model activity"),
            "settings": ("Settings", "Granite GGUF, optional semantic Qwen, web, identity, and runtime"),
        }
        title, subtitle = labels[key]
        self.page_title_var.set(title)
        self.page_subtitle_var.set(subtitle)
        for k, b in self.nav_buttons.items():
            active = k == key
            b.configure(bg=SURFACE_2 if active else SIDEBAR, fg=TEXT if active else TEXT_MUTED,
                        font=("Segoe UI Semibold", 10) if active else ("Segoe UI", 10))
        if key == "state": self._refresh_state()
        elif key == "memory": self._refresh_memory()

    def _active_chat_title(self) -> str:
        chat = self.chats.get_chat(self.current_chat_id)
        return str(chat.get("title", "New chat")) if chat else "New chat"

    @staticmethod
    def _chat_age_label(updated_at) -> str:
        try:
            ts = float(updated_at or 0)
            if not ts:
                return ""
            now = time.time()
            age = max(0, now - ts)
            if age < 86400 and time.localtime(ts).tm_yday == time.localtime(now).tm_yday:
                return time.strftime("%I:%M %p", time.localtime(ts)).lstrip("0")
            if age < 7 * 86400:
                return time.strftime("%a", time.localtime(ts))
            return time.strftime("%b %d", time.localtime(ts)).replace(" 0", " " )
        except Exception:
            return ""

    @staticmethod
    def _sidebar_chat_title(text: str, max_chars: int = 30) -> str:
        """Compact a sidebar title without changing the stored chat name."""
        clean = " ".join(str(text or "New chat").split()) or "New chat"
        if len(clean) <= max_chars:
            return clean
        return clean[: max(1, max_chars - 1)].rstrip() + "…"

    def _refresh_chat_list(self):
        for w in list(self.chat_list_inner.winfo_children()):
            w.destroy()
        self.chat_row_widgets.clear()
        filter_text = str(getattr(self, "chat_filter_var", tk.StringVar(value="")).get() or "").strip().lower()
        for meta in self.chats.list_chats():
            title_text = str(meta.get("title", "New chat"))
            sidebar_title = self._sidebar_chat_title(title_text)
            if filter_text and filter_text not in title_text.lower():
                continue
            cid = meta["id"]
            active = cid == self.current_chat_id
            # A soft card + accent rail is cleaner than a bright full outline.
            shadow = tk.Frame(self.chat_list_inner, bg=SHADOW_SOFT if active else SIDEBAR, bd=0)
            shadow.pack(fill="x", padx=3, pady=3)
            row_bg = "#17223a" if active else SIDEBAR_2
            row = tk.Frame(shadow, bg=row_bg, highlightthickness=1 if active else 0,
                           highlightbackground="#2d3b64")
            row.pack(fill="x", padx=(0, 1), pady=(0, 2 if active else 0))
            # Grid the row so the delete control owns a permanent column.  Long
            # titles can never push it outside the visible card.
            row.grid_columnconfigure(1, weight=1)
            rail = tk.Frame(row, bg=ACCENT if active else row_bg)
            rail.grid(row=0, column=0, rowspan=2, sticky="nsw")
            info = tk.Frame(row, bg=row_bg)
            info.grid(row=0, column=1, rowspan=2, sticky="nsew", pady=2)
            delete = tk.Button(
                row, text="×", command=lambda c=cid: self.delete_chat(c), bg=row_bg, fg=TEXT_DIM,
                activebackground="#35212a", activeforeground="#ffbec6", relief="flat", bd=0,
                highlightthickness=0, width=2, padx=2, pady=4, font=("Segoe UI", 10), cursor="hand2",
            )
            delete.grid(row=0, column=2, rowspan=2, sticky="ne", padx=(2, 4), pady=(3, 0))
            title = tk.Button(
                info, text=sidebar_title, command=lambda c=cid: self.switch_chat(c),
                bg=row_bg, fg=TEXT if active else TEXT_MUTED,
                activebackground=SURFACE_3, activeforeground=TEXT, relief="flat", bd=0,
                highlightthickness=0, anchor="w", justify="left", padx=10, pady=3,
                font=("Segoe UI Semibold", 9) if active else ("Segoe UI", 9), cursor="hand2",
            )
            title.pack(fill="x")
            meta_line = tk.Frame(info, bg=row_bg)
            meta_line.pack(fill="x", padx=10, pady=(0, 5))
            count = int(meta.get("message_count", 0) or 0)
            subtitle = "New conversation" if not count else f"{count} message{'s' if count != 1 else ''}"
            tk.Label(meta_line, text=subtitle, bg=row_bg, fg=TEXT_DIM, font=("Segoe UI", 8)).pack(side="left")
            age = self._chat_age_label(meta.get("updated_at"))
            if age:
                tk.Label(meta_line, text=age, bg=row_bg, fg=TEXT_DIM, font=("Segoe UI", 8)).pack(side="right")
            for w in (shadow, row, rail, info, title, meta_line, delete, *meta_line.winfo_children()):
                w.bind("<MouseWheel>", self._sidebar_wheel, add="+")
            self.chat_row_widgets.append((row, title, delete))
        self.root.after_idle(lambda: self.chat_list_canvas.configure(scrollregion=self.chat_list_canvas.bbox("all")))

    def new_chat(self):
        if self.busy:
            messagebox.showinfo("Turn in progress", "Finish or stop the current response before starting another chat.")
            return
        chat = self.chats.create_chat("New chat")
        self.current_chat_id = chat["id"]
        self._refresh_chat_list()
        self._load_active_chat()
        self.show_page("chat")
        self.input.focus_set()

    def switch_chat(self, chat_id: str):
        if chat_id == self.current_chat_id:
            self.show_page("chat")
            return
        if self.busy:
            messagebox.showinfo("Turn in progress", "Finish or stop the current response before switching chats.")
            return
        if not self.chats.set_active(chat_id):
            return
        self.current_chat_id = chat_id
        self._refresh_chat_list()
        self._load_active_chat()
        self.show_page("chat")

    def delete_chat(self, chat_id: str):
        if self.busy and chat_id == self.current_chat_id:
            messagebox.showinfo("Turn in progress", "The active chat cannot be deleted while a response is running.")
            return
        chat = self.chats.get_chat(chat_id)
        if not chat:
            return
        if not messagebox.askyesno(
            "Delete chat",
            f"Delete \"{chat.get('title', 'New chat')}\"?\n\nThis removes the visible chat transcript. Persistent neural memory is a separate lived-experience store and is not erased by deleting the visible chat transcript.",
        ):
            return
        new_active = self.chats.delete_chat(chat_id)
        if chat_id == self.current_chat_id:
            self.current_chat_id = new_active or self.chats.active_chat_id()
            self._load_active_chat()
        self._refresh_chat_list()
        self.show_page("chat")

    def rename_active_chat(self):
        chat = self.chats.get_chat(self.current_chat_id)
        if not chat:
            return
        title = simpledialog.askstring("Rename chat", "Chat name:", initialvalue=chat.get("title", "New chat"), parent=self.root)
        if title and self.chats.rename_chat(self.current_chat_id, title):
            self._refresh_chat_list()
            self.show_page("chat")

    def _load_active_chat(self):
        self.chat.clear()
        messages = self.chats.messages(self.current_chat_id)
        for item in messages:
            role = item.get("role")
            content = str(item.get("content", "") or "")
            if role == "user":
                self.chat.add_user(content)
            elif role == "assistant":
                self.chat.add_assistant(
                    name=normalize_identity_name(item.get("name") or self.config["identity"].get("name", "Assistant")),
                    answer=content, thinking=str(item.get("thinking", "") or ""),
                    tools=list(item.get("tools", []) or []), complete=True,
                )
        if not messages:
            self.chat.show_empty()
        self.chat.scroll_to_bottom()
        self.chat_title_var.set(self._active_chat_title())
        self.page_title_var.set(self._active_chat_title())

    # ---------- pages ----------
    def _build_chat(self):
        top = tk.Frame(self.chat_tab, bg=BG)
        top.pack(fill="x", padx=22, pady=(12, 7))
        self.chat_title_var = tk.StringVar(value=self._active_chat_title())
        tk.Label(top, textvariable=self.chat_title_var, bg=BG, fg=TEXT,
                 font=("Segoe UI Semibold", 13)).pack(side="left")
        self.rename_btn = ElevatedButton(top, text="Rename", command=self.rename_active_chat,
                                         variant="ghost", pady=6, padx=11)
        self.rename_btn.pack(side="right")

        body_shadow = tk.Frame(self.chat_tab, bg=SHADOW)
        body_shadow.pack(fill="both", expand=True, padx=20, pady=(4, 12))
        body = tk.Frame(body_shadow, bg=SURFACE, highlightthickness=1, highlightbackground=BORDER_SOFT)
        body.pack(fill="both", expand=True, padx=(0, 1), pady=(0, 3))
        self._img["empty_logo"] = self.assets.image("AevumAI-SoloLOGO_64x64.png")
        self.chat = ChatTranscript(body, empty_icon=self._img["empty_logo"])
        self.chat.pack(fill="both", expand=True, padx=1, pady=1)

        composer_outer = tk.Frame(self.chat_tab, bg=BG)
        composer_outer.pack(fill="x", padx=20, pady=(0, 17))
        composer_shadow = tk.Frame(composer_outer, bg=SHADOW)
        composer_shadow.pack(fill="x")
        composer = tk.Frame(composer_shadow, bg=SURFACE_2, highlightthickness=1, highlightbackground=BORDER)
        composer.pack(fill="x", padx=(0, 1), pady=(0, 3))
        composer.columnconfigure(0, weight=1)
        composer.rowconfigure(0, weight=1)
        self.input = tk.Text(composer, height=4, wrap="word")
        style_text(self.input, surface=SURFACE_2, font=("Segoe UI", 10))
        self.input.configure(highlightthickness=0)
        self.input.grid(row=0, column=0, sticky="nsew", padx=(12, 8), pady=9)
        self.input.bind("<Control-Return>", lambda _e: self.send_message())
        self.input.bind("<Return>", self._enter_to_send)

        controls = tk.Frame(composer, bg=SURFACE_2)
        controls.grid(row=0, column=1, padx=(0, 10), pady=9, sticky="ns")
        self._img["send_cursor"] = self.assets.image("AevumAI-Cursor_18x18.png")
        self.send_btn = ElevatedButton(controls, text="Send", image=self._img["send_cursor"], command=self.send_message,
                                       variant="primary", pady=7, padx=16)
        self.send_btn.pack(side="top", fill="x", pady=(0, 6))
        self.stop_btn = ElevatedButton(controls, text="Stop", command=self.stop_turn,
                                       variant="secondary", pady=6, padx=16)
        self.stop_btn.configure(state="disabled")
        self.stop_btn.pack(side="top", fill="x")
        tk.Label(composer_outer, text="Shift+Enter for a new line  ·  Enter/Ctrl+Enter to send",
                 bg=BG, fg=TEXT_DIM, font=("Segoe UI", 8)).pack(anchor="e", pady=(5, 0))

    def _enter_to_send(self, event):
        if event.state & 0x0001:  # Shift
            return None
        self.send_message()
        return "break"

    def _section_header(self, parent, title: str, subtitle: str = ""):
        wrap = tk.Frame(parent, bg=BG)
        wrap.pack(fill="x", padx=24, pady=(20, 11))
        tk.Label(wrap, text=title, bg=BG, fg=TEXT, font=FONT_TITLE).pack(anchor="w")
        if subtitle:
            tk.Label(wrap, text=subtitle, bg=BG, fg=TEXT_MUTED, font=FONT_SMALL).pack(anchor="w", pady=(3, 0))
        return wrap

    def _build_state(self):
        head = self._section_header(self.state_tab, "Internal state", "Optional Qwen semantic annotations update this compact functional state after raw neural storage is complete.")
        ElevatedButton(head, text="↻  Refresh", command=self._refresh_state, variant="ghost", pady=6).pack(side="right")
        shadow = tk.Frame(self.state_tab, bg=SHADOW)
        shadow.pack(fill="both", expand=True, padx=24, pady=(0, 22))
        card = tk.Frame(shadow, bg=SURFACE, highlightthickness=1, highlightbackground=BORDER_SOFT)
        card.pack(fill="both", expand=True, padx=(0, 1), pady=(0, 3))
        self.state_text = ScrolledText(card, wrap="word", state="disabled")
        style_text(self.state_text, mono=True, surface=SURFACE)
        self.state_text.configure(highlightthickness=0)
        self.state_text.pack(fill="both", expand=True, padx=10, pady=10)

    def _build_memory(self):
        head = self._section_header(self.memory_tab, "Neural memory", "Messages and imported text are stored as compact neural engrams; searches decode the remembered document on demand.")
        self.mem_stats_var = tk.StringVar(value="")
        tk.Label(head, textvariable=self.mem_stats_var, bg=BG, fg=TEXT_MUTED, font=FONT_SMALL).pack(anchor="w", pady=(6, 0))
        actions = tk.Frame(self.memory_tab, bg=BG)
        actions.pack(fill="x", padx=24, pady=(0, 10))
        ElevatedButton(actions, text="↻  Refresh", command=self._refresh_memory, variant="ghost", pady=6).pack(side="left", padx=(0, 7))
        ElevatedButton(actions, text="Import text / Markdown", command=self._import_memory_file, variant="secondary", pady=6).pack(side="left", padx=(0, 7))
        ElevatedButton(actions, text="✓  Validate", command=self._validate_memory, variant="secondary", pady=6).pack(side="left", padx=(0, 7))

        search_shadow = tk.Frame(self.memory_tab, bg=SHADOW_SOFT)
        search_shadow.pack(fill="x", padx=24, pady=(0, 10))
        search = tk.Frame(search_shadow, bg=SURFACE_2, highlightthickness=1, highlightbackground=BORDER_SOFT)
        search.pack(fill="x", padx=(0, 1), pady=(0, 2))
        self.mem_query = tk.StringVar()
        entry = ttk.Entry(search, textvariable=self.mem_query)
        entry.pack(side="left", fill="x", expand=True, padx=(8, 0), pady=8)
        entry.bind("<Return>", lambda _e: self._search_memory())
        ElevatedButton(search, text="Search", command=self._search_memory, variant="primary", pady=6).pack(side="left", padx=8, pady=6)

        shadow = tk.Frame(self.memory_tab, bg=SHADOW)
        shadow.pack(fill="both", expand=True, padx=24, pady=(0, 22))
        card = tk.Frame(shadow, bg=SURFACE, highlightthickness=1, highlightbackground=BORDER_SOFT)
        card.pack(fill="both", expand=True, padx=(0, 1), pady=(0, 3))
        self.memory_text = ScrolledText(card, wrap="word", state="disabled")
        style_text(self.memory_text, mono=True, surface=SURFACE)
        self.memory_text.configure(highlightthickness=0)
        self.memory_text.pack(fill="both", expand=True, padx=10, pady=10)

    def _build_activity(self):
        head = self._section_header(self.activity_tab, "Runtime activity", "Detailed pipeline events stay here instead of cluttering the chat.")
        ElevatedButton(head, text="Clear", command=lambda: self._set_text(self.activity, ""), variant="ghost", pady=6).pack(side="right")
        shadow = tk.Frame(self.activity_tab, bg=SHADOW)
        shadow.pack(fill="both", expand=True, padx=24, pady=(0, 22))
        card = tk.Frame(shadow, bg=SURFACE, highlightthickness=1, highlightbackground=BORDER_SOFT)
        card.pack(fill="both", expand=True, padx=(0, 1), pady=(0, 3))
        self.activity = ScrolledText(card, wrap="word", state="disabled")
        style_text(self.activity, mono=True, surface=SURFACE)
        self.activity.configure(highlightthickness=0)
        self.activity.pack(fill="both", expand=True, padx=10, pady=10)

    def _settings_card(self, inner, title: str, subtitle: str = ""):
        shadow = tk.Frame(inner, bg=SHADOW_SOFT)
        shadow.pack(fill="x", pady=(0, 15))
        card = tk.Frame(shadow, bg=SURFACE, highlightthickness=1, highlightbackground=BORDER_SOFT)
        card.pack(fill="x", padx=(0, 1), pady=(0, 3))
        tk.Label(card, text=title, bg=SURFACE, fg=TEXT, font=FONT_SECTION).pack(anchor="w", padx=17, pady=(15, 0))
        if subtitle:
            tk.Label(card, text=subtitle, bg=SURFACE, fg=TEXT_MUTED, font=FONT_SMALL,
                     wraplength=900, justify="left").pack(anchor="w", padx=17, pady=(4, 11))
        body = tk.Frame(card, bg=SURFACE)
        body.pack(fill="x", padx=17, pady=(8 if not subtitle else 0, 17))
        return body

    def _build_settings(self):
        canvas = tk.Canvas(self.settings_tab, highlightthickness=0, bg=BG)
        sb = ttk.Scrollbar(self.settings_tab, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=BG)
        win = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win, width=max(760, e.width - 8)))
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True, padx=(22, 0), pady=18)
        sb.pack(side="right", fill="y", pady=18, padx=(0, 8))
        canvas.bind("<MouseWheel>", lambda e: (canvas.yview_scroll(int(-e.delta / 120) * 3, "units"), "break")[1])

        body = self._settings_card(inner, "Models", "Granite/chat GGUF is the only required model. Qwen3.5 ~2B is an optional CPU semantic annotator and never blocks chat or memory.")
        labels = {"executive": "Executive · Granite/chat GGUF", "annotator": "Semantic annotator · optional Qwen3.5"}
        row = 0
        for role in ("executive", "annotator"):
            tk.Label(body, text=labels[role], bg=SURFACE, fg=TEXT_MUTED, font=FONT_SMALL).grid(row=row, column=0, sticky="w", pady=5)
            var = tk.StringVar(value=str(self.config["models"][role].get("path", "")))
            self.model_entries[role] = var
            ttk.Entry(body, textvariable=var).grid(row=row, column=1, sticky="ew", padx=8, pady=5)
            ElevatedButton(body, text="Browse", command=lambda r=role: self._browse_model(r), variant="secondary", pady=5).grid(row=row, column=2, pady=5)
            row += 1
        body.columnconfigure(1, weight=1)
        self.annotator_enabled_var = tk.BooleanVar(value=bool(self.config["models"]["annotator"].get("enabled", False)))
        ttk.Checkbutton(body, text="Enable semantic annotator", variable=self.annotator_enabled_var).grid(row=row, column=0, columnspan=2, sticky="w", pady=(7, 2)); row += 1
        initial_preset = detect_preset_from_filename(self.config["models"]["executive"].get("path", ""))
        self.executive_detected_var = tk.StringVar(value=f"Executive preset: {preset_summary(initial_preset)} (filename estimate until loaded)")
        tk.Label(body, textvariable=self.executive_detected_var, bg=SURFACE, fg=TEXT_MUTED, font=FONT_SMALL).grid(row=row, column=0, columnspan=3, sticky="w", pady=(6, 0))

        body = self._settings_card(inner, "Runtime & tools", "Recent chat stays directly in context. Older neural matches preload as metadata-only hints; full memories are opened only when the model needs them.")
        r = 0
        tk.Label(body, text="Chat provider", bg=SURFACE, fg=TEXT_MUTED).grid(row=r, column=0, sticky="w", pady=4)
        self.provider_var = tk.StringVar(value=str((self.config.get("chat") or {}).get("provider", "local_gguf")))
        ttk.Combobox(body, textvariable=self.provider_var, state="readonly", values=("local_gguf", "openai")).grid(row=r, column=1, sticky="w", padx=8); r += 1
        self.browser_enabled_var = tk.BooleanVar(value=bool((self.config.get("browser") or {}).get("enabled", True)))
        ttk.Checkbutton(body, text="Enable web search / fetch", variable=self.browser_enabled_var).grid(row=r, column=0, columnspan=2, sticky="w", pady=4); r += 1
        tk.Label(body, text="Web search provider", bg=SURFACE, fg=TEXT_MUTED).grid(row=r, column=0, sticky="w", pady=4)
        self.browser_provider_var = tk.StringVar(value=str((self.config.get("browser") or {}).get("search_provider", "auto")))
        ttk.Combobox(body, textvariable=self.browser_provider_var, state="readonly", values=("auto", "duckduckgo", "bing")).grid(row=r, column=1, sticky="w", padx=8); r += 1
        tk.Label(body, text="Executive sampler", bg=SURFACE, fg=TEXT_MUTED).grid(row=r, column=0, sticky="w", pady=4)
        self.executive_sampler_var = tk.StringVar(value=str(self.config["models"]["executive"].get("sampler_profile", "auto")))
        ttk.Combobox(body, textvariable=self.executive_sampler_var, state="readonly", values=("auto", "custom")).grid(row=r, column=1, sticky="w", padx=8); r += 1
        tk.Label(body, text="Neural HUD quality", bg=SURFACE, fg=TEXT_MUTED).grid(row=r, column=0, sticky="w", pady=4)
        self.neural_hud_quality_var = tk.StringVar(value=str((self.config.get("ui") or {}).get("neural_minimap_quality", "smooth")))
        ttk.Combobox(body, textvariable=self.neural_hud_quality_var, state="readonly", values=("smooth", "balanced", "detailed")).grid(row=r, column=1, sticky="w", padx=8); r += 1

        fields = [
            ("executive_ctx", "Executive context", self.config["models"]["executive"]["n_ctx"]),
            ("executive_gpu", "Executive GPU layers (-1 = all)", self.config["models"]["executive"]["n_gpu_layers"]),
            ("executive_threads", "Executive CPU threads", self.config["models"]["executive"]["n_threads"]),
            ("executive_max_tokens", "Executive max output tokens", self.config["models"]["executive"]["max_tokens"]),
            ("executive_temp", "Executive temperature", self.config["models"]["executive"].get("temperature", GENERIC_EXECUTIVE_TEXT["temperature"])),
            ("executive_top_p", "Executive top-p", self.config["models"]["executive"].get("top_p", GENERIC_EXECUTIVE_TEXT["top_p"])),
            ("executive_top_k", "Executive top-k", self.config["models"]["executive"].get("top_k", GENERIC_EXECUTIVE_TEXT["top_k"])),
            ("executive_min_p", "Executive min-p", self.config["models"]["executive"].get("min_p", GENERIC_EXECUTIVE_TEXT["min_p"])),
            ("executive_presence", "Executive presence penalty", self.config["models"]["executive"].get("presence_penalty", GENERIC_EXECUTIVE_TEXT["presence_penalty"])),
            ("executive_frequency", "Executive frequency penalty", self.config["models"]["executive"].get("frequency_penalty", GENERIC_EXECUTIVE_TEXT["frequency_penalty"])),
            ("executive_repeat", "Executive repetition penalty", self.config["models"]["executive"].get("repeat_penalty", GENERIC_EXECUTIVE_TEXT["repeat_penalty"])),
            ("thinking_max_chars", "Thinking guard max characters", self.config["models"]["executive"].get("thinking_max_chars", 5000)),
            ("annotator_ctx", "Semantic Qwen context", self.config["models"]["annotator"]["n_ctx"]),
            ("annotator_threads", "Semantic Qwen CPU threads", self.config["models"]["annotator"]["n_threads"]),
            ("recent_turns", "Recent turns in fast context", self.config["runtime"].get("recent_turns", 16)),
            ("memory_hints", "Automatic neural memory hints", self.config["runtime"].get("auto_memory_hints", 6)),
            ("tool_rounds", "Executive private tool rounds", self.config["runtime"].get("tool_rounds", 4)),
            ("browser_timeout", "Web request timeout (s)", self.config["browser"].get("timeout_seconds", 12.0)),
            ("browser_results", "Web search result limit", self.config["browser"].get("search_results", 8)),
            ("chunk_tokens", "Neural storage chunk tokens", self.config["memory"].get("chunk_tokens", 12)),
        ]
        for key, label, val in fields:
            tk.Label(body, text=label, bg=SURFACE, fg=TEXT_MUTED).grid(row=r, column=0, sticky="w", pady=4)
            var = tk.StringVar(value=str(val)); self.setting_vars[key] = var
            ttk.Entry(body, textvariable=var).grid(row=r, column=1, sticky="w", padx=8, pady=4)
            r += 1
        ElevatedButton(body, text="Reset Executive to Auto", command=self._reset_executive_sampler_fields, variant="ghost", pady=6).grid(row=r, column=0, columnspan=2, sticky="w", pady=(10, 0))

        body = self._settings_card(inner, "Identity", "Names are attached as provenance to every neurally stored chat message.")
        tk.Label(body, text="User name", bg=SURFACE, fg=TEXT_MUTED).grid(row=0, column=0, sticky="w", pady=4)
        self.user_name_var = tk.StringVar(value=str(self.config["identity"].get("user_name", "User")))
        ttk.Entry(body, textvariable=self.user_name_var).grid(row=0, column=1, sticky="w", padx=8, pady=4)
        tk.Label(body, text="Assistant name", bg=SURFACE, fg=TEXT_MUTED).grid(row=1, column=0, sticky="w", pady=4)
        self.identity_name_var = tk.StringVar(value=normalize_identity_name(self.config["identity"].get("name", "Assistant")))
        ttk.Entry(body, textvariable=self.identity_name_var).grid(row=1, column=1, sticky="w", padx=8, pady=4)

        body = self._settings_card(inner, "OpenAI fallback", "Optional. API key is read from OPENAI_API_KEY and is never written to config.")
        tk.Label(body, text="OpenAI model", bg=SURFACE, fg=TEXT_MUTED).grid(row=0, column=0, sticky="w")
        self.openai_model_var = tk.StringVar(value=str((self.config.get("openai") or {}).get("model", "gpt-5.6-luna")))
        ttk.Entry(body, textvariable=self.openai_model_var).grid(row=0, column=1, sticky="ew", padx=8)
        body.columnconfigure(1, weight=1)

        body = self._settings_card(inner, "System prompt", "Operating rules, memory behavior, tool policy, and hard constraints.")
        self.system_prompt = tk.Text(body, width=100, height=13, wrap="word")
        style_text(self.system_prompt, surface=SURFACE_2)
        self.system_prompt.insert("1.0", self.config["identity"].get("system_prompt", ""))
        self.system_prompt.pack(fill="x")

        body = self._settings_card(inner, "Personality prompt", "Voice and conversational character; the optional semantic state is supplied separately.")
        self.personality_prompt = tk.Text(body, width=100, height=7, wrap="word")
        style_text(self.personality_prompt, surface=SURFACE_2)
        self.personality_prompt.insert("1.0", self.config["identity"].get("personality_prompt", ""))
        self.personality_prompt.pack(fill="x")

        buttons = tk.Frame(inner, bg=BG)
        buttons.pack(fill="x", pady=(0, 28))
        ElevatedButton(buttons, text="Save settings", command=self.save_settings, variant="secondary", pady=7).pack(side="left", padx=(0, 8))
        ElevatedButton(buttons, text="Save + reload model", command=lambda: self.save_settings(reload_models=True), variant="primary", pady=7).pack(side="left")

    def _apply_executive_preset_fields(self, preset) -> None:
        values = sampler_profile(preset, structured=False)
        mapping = {
            "executive_temp": values["temperature"], "executive_top_p": values["top_p"],
            "executive_top_k": values["top_k"], "executive_min_p": values["min_p"],
            "executive_presence": values["presence_penalty"], "executive_frequency": values["frequency_penalty"],
            "executive_repeat": values["repeat_penalty"],
        }
        for key, value in mapping.items():
            if key in self.setting_vars: self.setting_vars[key].set(str(value))

    def _reset_executive_sampler_fields(self):
        self.executive_sampler_var.set("auto")
        preset = detect_preset_from_filename(self.model_entries.get("executive", tk.StringVar()).get())
        self._apply_executive_preset_fields(preset)
        self.executive_detected_var.set(f"Executive preset: {preset_summary(preset)} (filename estimate until loaded)")
        self.status_var.set(f"Executive Auto preset: {preset.label}")

    def _browse_model(self, role: str):
        p = filedialog.askopenfilename(title=f"Select {role} GGUF", filetypes=[("GGUF models", "*.gguf"), ("All files", "*.*")])
        if not p: return
        self.model_entries[role].set(p)
        if role == "executive":
            preset = detect_preset_from_filename(p)
            self.executive_detected_var.set(f"Executive preset: {preset_summary(preset)} (filename estimate until loaded)")
            if self.executive_sampler_var.get() == "auto": self._apply_executive_preset_fields(preset)
            self.status_var.set(f"Selected {Path(p).name} · {preset.label}")

    # ---------- chat execution ----------
    def send_message(self):
        if self.busy: return
        if self.models_loading:
            self.show_page("quickstart")
            return
        text = self.input.get("1.0", "end").strip()
        if not text: return
        provider = str((self.config.get("chat") or {}).get("provider", "local_gguf") or "local_gguf")
        if provider == "local_gguf":
            statuses = self.pool.statuses()
            if not bool((statuses.get("executive") or {}).get("loaded")):
                messagebox.showwarning(
                    "Executive not loaded",
                    "Load the Executive GGUF first. The semantic annotator is optional and is never required for chat.",
                )
                return
        elif provider == "openai":
            from core.config import api_key_from_environment
            if not api_key_from_environment():
                messagebox.showwarning(
                    "OpenAI API key missing",
                    "Set OPENAI_API_KEY, or switch Chat provider back to local_gguf in Settings.",
                )
                return
        try:
            chat_user_message_id = self.chats.add_message(self.current_chat_id, "user", text)
        except Exception as exc:
            messagebox.showerror("Chat save error", f"Could not save the message to this chat:\n\n{exc}")
            return
        self.input.delete("1.0", "end")
        self.chat.add_user(text)
        self._refresh_chat_list()
        self.chat_title_var.set(self._active_chat_title())
        self.page_title_var.set(self._active_chat_title())
        self.current_assistant = self.chat.add_assistant(name=normalize_identity_name(self.config["identity"].get("name", "Assistant")), answer="", thinking="", complete=False)
        self.busy = True; self.streaming = True
        self.send_btn.configure(state="disabled"); self.stop_btn.configure(state="normal")
        self.new_chat_btn.configure(state="disabled")
        self.status_var.set("Thinking…"); self._img["model_status"] = self.assets.image("AevumAI-Indicator-Load_18x18.png"); self.status_dot.configure(image=self._img["model_status"])
        turn_chat_id = self.current_chat_id

        def token_cb(tok: str): self.uiq.put(("token", tok))
        def thought_cb(tok: str): self.uiq.put(("thought", tok))
        def event_cb(name: str, data: Dict[str, Any]): self.uiq.put(("event", (name, data)))
        def run():
            try:
                result = self.scheduler.run_turn(
                    text, on_token=token_cb, on_thought=thought_cb, on_event=event_cb,
                    chat_id=turn_chat_id, chat_user_message_id=chat_user_message_id,
                )
                self.uiq.put(("turn_done", result))
            except Exception as exc:
                self.uiq.put(("turn_error", str(exc)))
        threading.Thread(target=run, name="cognitive-turn", daemon=True).start()

    def stop_turn(self):
        if self.busy:
            self.scheduler.cancel(); self.status_var.set("Stopping…")

    # ---------- model lifecycle ----------
    def _quickstart_load(self):
        if self.busy or self.models_loading:
            return
        self.provider_var.set("local_gguf")
        self.save_settings(reload_models=True)

    def _set_models_loading(self, loading: bool):
        self.models_loading = loading
        self.start_models_btn.configure(state="disabled" if loading else "normal")
        self.stop_models_btn.configure(state="disabled" if loading else "normal")
        self.quickstart.set_loading(loading)

    def _load_models_async(self, operation):
        self._set_models_loading(True)
        self.status_var.set("Loading model…")
        self._img["model_status"] = self.assets.image("AevumAI-Indicator-Load_18x18.png")
        self.status_dot.configure(image=self._img["model_status"])
        def run():
            try:
                errors = operation()
            except Exception as exc:
                errors = {"executive": str(exc)}
            self.uiq.put(("models_started", errors))
        threading.Thread(target=run, name="model-loader", daemon=True).start()

    def start_models(self):
        if self.busy or self.models_loading: return
        if needs_model_setup(self.config):
            self.show_page("quickstart")
            self.quickstart.set_result(False, "Choose your Executive GGUF, then click Save & load models.")
            return
        self._activity("Starting Granite Executive and optional semantic annotator…")
        self._load_models_async(self.pool.start)

    def stop_models(self):
        if self.models_loading: return
        self.pool.stop_all()
        self.quickstart.set_result(False)
        self._refresh_status()
        self._activity("Model workers stopped.")

    def _refresh_status(self):
        statuses = self.pool.statuses()
        if self.busy or self.models_loading:
            return
        ex = statuses.get("executive", {})
        ann = statuses.get("annotator", {})
        ready = bool(ex.get("loaded"))
        ann_cfg = self.config.get("models", {}).get("annotator", {})
        if ready:
            suffix = " · semantic on" if ann.get("loaded") else (" · semantic off" if not ann_cfg.get("enabled") else " · semantic unavailable")
            self.status_var.set("Ready" + suffix)
        else:
            self.status_var.set("Model stopped")
        self._img["model_status"] = self.assets.image("AevumAI-Indicator-ON_18x18.png" if ready else "AevumAI-Indicator-OFF_18x18.png")
        self.status_dot.configure(image=self._img["model_status"])
        try:
            stats = self.memory.stats()
            self.memory_badge.configure(text=f"NEURAL · {int(stats.get('documents',0))}", fg=SUCCESS)
        except Exception:
            self.memory_badge.configure(text="NEURAL ERROR", fg=ERROR)

    @staticmethod
    def _set_text(widget: ScrolledText, text: str):
        widget.configure(state="normal"); widget.delete("1.0", "end"); widget.insert("1.0", text); widget.configure(state="disabled")

    def _activity(self, line: str):
        self.activity.configure(state="normal"); self.activity.insert("end", line.rstrip() + "\n"); self.activity.see("end"); self.activity.configure(state="disabled")

    def _refresh_state(self):
        try:
            snap = self.scheduler.state_engine.snapshot()
            statuses = self.pool.statuses()
            payload = {
                "semantic_annotator": "loaded" if statuses.get("annotator", {}).get("loaded") else "disabled / unavailable",
                "functional_state": snap,
                "note": "Raw chat remains in neural memory. This state is optional semantic context layered on top.",
            }
            self._set_text(self.state_text, json.dumps(payload, indent=2, ensure_ascii=False))
        except Exception as exc:
            self._set_text(self.state_text, f"State unavailable: {exc}")

    def _refresh_memory(self):
        """Refresh the memory dashboard without decoding historical engrams.

        A large lifetime store can contain manuals and thousands of messages. Merely
        visiting the Memory page must therefore remain metadata-only; full neural
        decoding happens only for an explicit search/open operation.
        """
        try:
            stats = self.memory.stats()
            self.memory_badge.configure(text=f"NEURAL · {int(stats.get('documents',0))}", fg=SUCCESS)
            self.mem_stats_var.set(
                f"{int(stats.get('documents',0)):,} documents · {int(stats.get('neural_chunks',0)):,} engrams · "
                f"{int(stats.get('engram_bytes',0))/1024:.1f} KiB compact · scales {stats.get('scales_ms', [])} ms"
            )
            docs = sorted(
                (dict(d) for d in list(getattr(self.memory.brain, "neural_documents", []) or [])),
                key=lambda d: (str(d.get("timestamp_utc") or ""), int(d.get("id", -1))),
                reverse=True,
            )[:24]
            blocks = []
            for d in docs:
                semantic = d.get("semantic") if isinstance(d.get("semantic"), dict) else {}
                topics = ", ".join(str(x) for x in list(semantic.get("topics") or [])[:4])
                source = str(d.get("source_name") or d.get("kind") or "memory")
                extra = f" · {topics}" if topics else ""
                blocks.append(
                    f"DOC {d.get('id')}  {d.get('timestamp_utc','')}  {d.get('speaker','')} [{d.get('role','')}]\n"
                    f"{source}{extra}"
                )
            note = "Recent neural documents (metadata only; search to decode full content):\n\n" if blocks else ""
            self._set_text(self.memory_text, note + "\n\n".join(blocks) if blocks else "Neural memory is empty.")
        except Exception as exc:
            self.memory_badge.configure(text="NEURAL ERROR", fg=ERROR)
            self.mem_stats_var.set("Neural memory unavailable")
            self._set_text(self.memory_text, str(exc))

    def _reopen_memory(self):
        self._refresh_memory()
        messagebox.showinfo("Neural memory", "Neural memory is embedded in the Aevum runtime and is loaded automatically at startup.")

    def _import_memory_file(self):
        p = filedialog.askopenfilename(
            title="Import text into neural memory",
            filetypes=[("Text / Markdown", "*.txt *.md *.markdown"), ("Text files", "*.txt"), ("Markdown", "*.md *.markdown"), ("All files", "*.*")],
        )
        if not p:
            return
        path = Path(p)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            messagebox.showerror("Import failed", f"Could not read {path.name}:\n\n{exc}")
            return
        if not text.strip():
            messagebox.showwarning("Import", "That file contains no text to remember.")
            return
        self.status_var.set(f"Remembering {path.name}…")
        self._activity(f"Importing {path.name} into neural memory ({len(text):,} characters)…")

        def progress(i, n, info):
            try:
                blob = self.memory.memory_blob(int(info.get("id")))
                self.uiq.put(("event", ("neural_activity", {"mode": "REMEMBER", "label": path.name, "blob": blob})))
            except Exception:
                pass
            if i == 0 or (i + 1) == n or (i + 1) % 10 == 0:
                self.uiq.put(("memory_import_progress", path.name, i + 1, n))

        def run():
            try:
                info = self.memory.store_text(text, source_name=path.name, speaker="Imported", progress=progress)
                self.uiq.put(("memory_import_done", path.name, info))
            except Exception as exc:
                self.uiq.put(("memory_import_error", path.name, str(exc)))
        threading.Thread(target=run, name="neural-memory-import", daemon=True).start()

    def _validate_memory(self):
        def run():
            try:
                from compact_engram import header
                checked = 0
                bad = []
                for rec in list(self.memory.brain.neural_memories):
                    try:
                        blob = self.memory.memory_blob(int(rec["id"]))
                        h = header(blob)
                        if int(h.get("version", 0)) != 2:
                            raise ValueError("unexpected engram version")
                        checked += 1
                    except Exception as exc:
                        bad.append(f"memory {rec.get('id')}: {exc}")
                self.uiq.put(("memory_validation", checked, bad))
            except Exception as exc:
                self.uiq.put(("turn_error", f"Memory validation failed: {exc}"))
        threading.Thread(target=run, name="neural-memory-validate", daemon=True).start()

    def _search_memory(self):
        q = self.mem_query.get().strip()
        if not q:
            return
        self._set_text(self.memory_text, "Searching neural memory…")
        def run():
            try:
                hits = self.memory.search_full_documents(q, top_k=int((self.config.get("memory") or {}).get("search_limit", 20)))
                self.uiq.put(("memory_search_results", q, hits))
            except Exception as exc:
                self.uiq.put(("turn_error", f"Memory search failed: {exc}"))
        threading.Thread(target=run, name="neural-memory-search", daemon=True).start()

    def save_settings(self, reload_models: bool = False):
        if self.busy or self.models_loading:
            return
        try:
            if reload_models and self.provider_var.get() == "local_gguf":
                error = model_file_error(self.model_entries["executive"].get())
                if not error and self.annotator_enabled_var.get():
                    error = model_file_error(self.model_entries["annotator"].get(), "optional Qwen")
                if error:
                    self.show_page("quickstart")
                    self.quickstart.set_result(False, error)
                    return
            for role, var in self.model_entries.items():
                raw_path = var.get().strip()
                self.config["models"][role]["path"] = str(Path(raw_path).expanduser()) if raw_path else ""
            ex = self.config["models"]["executive"]
            ex["n_ctx"] = int(self.setting_vars["executive_ctx"].get())
            ex["n_gpu_layers"] = int(self.setting_vars["executive_gpu"].get())
            ex["n_threads"] = int(self.setting_vars["executive_threads"].get())
            ex["offload_kqv"] = True
            ex["max_tokens"] = max(128, int(self.setting_vars["executive_max_tokens"].get()))
            ex["temperature"] = float(self.setting_vars["executive_temp"].get())
            ex["top_p"] = float(self.setting_vars["executive_top_p"].get())
            ex["top_k"] = int(self.setting_vars["executive_top_k"].get())
            ex["min_p"] = float(self.setting_vars["executive_min_p"].get())
            ex["presence_penalty"] = float(self.setting_vars["executive_presence"].get())
            ex["frequency_penalty"] = float(self.setting_vars["executive_frequency"].get())
            ex["repeat_penalty"] = float(self.setting_vars["executive_repeat"].get())
            ex["thinking_max_chars"] = max(1000, int(self.setting_vars["thinking_max_chars"].get()))
            sampler_mode = str(self.executive_sampler_var.get() or "auto").strip().lower()
            ex["sampler_profile"] = sampler_mode if sampler_mode in {"auto", "custom"} else "auto"
            ex["enable_thinking"] = True

            ann = self.config["models"]["annotator"]
            ann["enabled"] = bool(self.annotator_enabled_var.get())
            ann["n_ctx"] = int(self.setting_vars["annotator_ctx"].get())
            ann["n_threads"] = int(self.setting_vars["annotator_threads"].get())
            ann["n_gpu_layers"] = 0; ann["offload_kqv"] = False; ann["enable_thinking"] = False; ann["required_architecture"] = "qwen35"

            self.config["chat"]["provider"] = str(self.provider_var.get() or "local_gguf")
            rt = self.config["runtime"]
            rt["recent_turns"] = max(1, int(self.setting_vars["recent_turns"].get()))
            rt["auto_memory_hints"] = max(0, min(12, int(self.setting_vars["memory_hints"].get())))
            rt["tool_rounds"] = max(1, min(6, int(self.setting_vars["tool_rounds"].get())))
            self.config["browser"]["enabled"] = bool(self.browser_enabled_var.get())
            self.config["browser"]["search_provider"] = str(self.browser_provider_var.get() or "auto")
            self.config["browser"]["timeout_seconds"] = max(3.0, min(45.0, float(self.setting_vars["browser_timeout"].get())))
            self.config["browser"]["search_results"] = max(1, min(12, int(self.setting_vars["browser_results"].get())))
            self.config["memory"]["chunk_tokens"] = max(1, min(64, int(self.setting_vars["chunk_tokens"].get())))
            quality = str(self.neural_hud_quality_var.get() or "smooth").strip().lower()
            if quality not in {"smooth", "balanced", "detailed"}: quality = "smooth"
            self.config.setdefault("ui", {})["neural_minimap_quality"] = quality
            self.neural_minimap.set_quality(quality)
            self.config["openai"]["model"] = self.openai_model_var.get().strip() or "gpt-5.6-luna"

            self.config["identity"]["user_name"] = " ".join(self.user_name_var.get().split())[:64] or "User"
            self.config["identity"]["name"] = normalize_identity_name(self.identity_name_var.get())
            self.identity_name_var.set(self.config["identity"]["name"])
            self.config["identity"]["system_prompt"] = self.system_prompt.get("1.0", "end").strip()
            self.config["identity"]["personality_prompt"] = self.personality_prompt.get("1.0", "end").strip()
            self.root.title(f"Aevum AI {VERSION} — {self.config['identity']['name']}")

            save_config(self.config)
            self.scheduler.config = self.config
            self.scheduler.reconfigure()
            self.pool.config = self.config
            self.pool.model_configs = self.config["models"]
            self.memory.chunk_tokens = self.config["memory"]["chunk_tokens"]
            if reload_models:
                self._load_models_async(lambda: self.pool.reload(self.config["models"]))
            else:
                messagebox.showinfo("Settings", "Settings saved. Reload the model for GGUF/runtime model changes to take effect.")
        except Exception as exc:
            if self.current_page == "quickstart":
                self.quickstart.set_result(False, str(exc))
            messagebox.showerror("Settings error", str(exc))

    def _poll_uiq(self):
        try:
            for _ in range(250):
                item = self.uiq.get_nowait()
                kind = item[0]
                if kind == "token":
                    payload = item[1]
                    if self.current_assistant: self.current_assistant.append_answer(payload)
                elif kind == "thought":
                    payload = item[1]
                    if self.current_assistant: self.current_assistant.append_thought(payload)
                elif kind == "event":
                    name, data = item[1]
                    if name == "neural_activity":
                        blob = data.get("blob")
                        self.neural_minimap.play_blob(blob, mode=str(data.get("mode") or "ACTIVE"), label=str(data.get("label") or ""))
                        # The minimap is the live neural telemetry. Avoid appending
                        # every animation event to the Tk activity text widget;
                        # high-volume remembering should not cause layout/scroll churn.
                    elif name in {"executive_tool_start", "executive_tool"}:
                        if self.current_assistant:
                            self.current_assistant.append_tool_event(data)
                        action = str(data.get("action", "tool")).replace("_", " ")
                        self.status_var.set(f"Using {action}…")
                        safe = {k:v for k,v in data.items() if k != "blob"}
                        short = json.dumps(safe, ensure_ascii=False, default=str)
                        self._activity(f"{name}: {short[:900]}{'…' if len(short) > 900 else ''}")
                    elif name == "visible_complete":
                        self.streaming = False
                        _commit_final_answer(self.current_assistant, data)
                        pending = int(data.get("background_pending", 0) or 0)
                        self.status_var.set("Answer complete" + (f" · semantic {pending}" if pending else ""))
                    elif name == "turn_mode":
                        self.status_var.set("NEURAL CONTEXT")
                        self._activity(f"turn_mode: {json.dumps(data, ensure_ascii=False, default=str)}")
                    elif name == "executive_start":
                        secs = float(data.get("pre_exec_seconds", 0.0) or 0.0)
                        self.status_var.set(f"Executive running · pre-model {secs:.2f}s")
                        self._activity(f"executive_start: {json.dumps(data, ensure_ascii=False, default=str)}")
                    elif name == "background_complete":
                        pending = int(data.get("pending", 0) or 0)
                        self._activity(f"background_complete: {json.dumps(data, ensure_ascii=False, default=str)}")
                        self._refresh_state()
                        if not self.busy:
                            self.status_var.set("Ready" if pending == 0 else f"Ready · semantic {pending}")
                    elif name == "error":
                        self._activity(f"ERROR: {data.get('message')}")
                    elif name == "warning":
                        self._activity(f"WARNING: {data.get('message')}")
                    else:
                        short = json.dumps(data, ensure_ascii=False, default=str)
                        self._activity(f"{name}: {short[:900]}{'…' if len(short) > 900 else ''}")
                elif kind == "turn_done":
                    payload = item[1]
                    _commit_final_answer(self.current_assistant, payload if isinstance(payload, dict) else {})
                    self.busy = False; self.streaming = False
                    self.send_btn.configure(state="normal"); self.stop_btn.configure(state="disabled"); self.new_chat_btn.configure(state="normal")
                    self.current_assistant = None
                    self._refresh_chat_list(); self.chat_title_var.set(self._active_chat_title()); self.page_title_var.set(self._active_chat_title())
                    self._refresh_state(); self._refresh_status()
                elif kind == "turn_error":
                    payload = item[1]
                    if self.current_assistant:
                        self.current_assistant.append_error(f"\n\n[Turn failed: {payload}]")
                        self.current_assistant.finish()
                    self.busy = False; self.streaming = False
                    self.send_btn.configure(state="normal"); self.stop_btn.configure(state="disabled"); self.new_chat_btn.configure(state="normal")
                    self.status_var.set("Turn failed")
                    self._img["model_status"] = self.assets.image("AevumAI-Indicator-OFF_18x18.png")
                    self.status_dot.configure(image=self._img["model_status"])
                    self.current_assistant = None
                    self._activity(f"TURN ERROR: {payload}")
                elif kind == "models_started":
                    errors = item[1] or {}
                    self._set_models_loading(False)
                    if errors:
                        self._activity("Model startup issues: " + json.dumps(errors, ensure_ascii=False, default=str))
                        # Executive failure matters; annotator failure is a warning only.
                        if "executive" in errors:
                            messagebox.showwarning("Model startup", "Executive model did not load:\n\n" + str(errors["executive"]))
                        elif "annotator" in errors:
                            self._activity("Semantic annotator unavailable; chat and neural memory remain fully functional.")
                    statuses = self.pool.statuses()
                    for role, st in statuses.items():
                        self._activity(f"{role}: loaded={bool(st.get('loaded'))} · arch={st.get('architecture') or '?'} · preset={st.get('preset_label') or '?'} · ctx={st.get('context') or '?'} · gpu_layers={st.get('gpu_layers')}")
                    est = statuses.get("executive", {})
                    self.quickstart.set_result(
                        bool(est.get("loaded")), str(errors.get("executive", "")),
                        annotator_unavailable="annotator" in errors,
                    )
                    if est.get("loaded") and hasattr(self, "executive_detected_var"):
                        self.executive_detected_var.set(f"Executive preset: {est.get('preset_label') or 'Generic GGUF'} · metadata architecture: {est.get('architecture') or '?'}")
                        if self.executive_sampler_var.get() == "auto":
                            self._apply_executive_preset_fields(preset_by_key(est.get("preset_key", "generic")))
                    self._refresh_status()
                elif kind == "memory_search_results":
                    _kind, q, hits = item
                    blocks = []
                    for h in hits:
                        blocks.append(
                            f"DOC {h.get('document_id')}  {h.get('timestamp_utc','')}  {h.get('speaker','')} [{h.get('role','')}]\n"
                            f"MATCH: {', '.join(h.get('match_tokens') or [])}  score={float(h.get('neural_score',0)):.3f}\n"
                            f"{h.get('text','')}"
                        )
                        mids = list(h.get("matched_memory_ids") or [])
                        try:
                            blob = self.memory.memory_blob(int(mids[0])) if mids else self.memory.document_first_blob(int(h.get("document_id")))
                            self.neural_minimap.play_blob(blob, mode="RECALL", label=q)
                        except Exception:
                            pass
                    self._set_text(self.memory_text, "\n\n".join(blocks) or f"No neural memory matched {q!r}.")
                elif kind == "memory_import_progress":
                    _kind, name, done, total = item
                    self.status_var.set(f"Remembering {name} · {done}/{total}")
                elif kind == "memory_import_done":
                    _kind, name, info = item
                    self.status_var.set(f"Remembered {name}")
                    self._activity(f"Imported {name}: document {info.get('document_id')} · {len(info.get('memory_ids') or [])} neural chunk(s)")
                    self._refresh_memory()
                elif kind == "memory_import_error":
                    _kind, name, error = item
                    self.status_var.set("Import failed")
                    messagebox.showerror("Import failed", f"Could not remember {name}:\n\n{error}")
                elif kind == "memory_validation":
                    _kind, checked, bad = item
                    if bad:
                        messagebox.showerror("Neural memory validation", f"Checked {checked} engrams.\n\n" + "\n".join(bad[:20]))
                    else:
                        messagebox.showinfo("Neural memory validation", f"Valid. {checked} compact neural engrams checked.")
                elif kind == "reconcile_done":
                    _kind, count = item
                    if count:
                        self._activity(f"Recovered {count} transcript message(s) into neural memory after startup.")
                    self._refresh_status()
        except queue.Empty:
            pass
        self.root.after(80, self._poll_uiq)

    def _reconcile_chat_memory(self):
        """Crash-safe archive of visible chat messages missing from neural memory."""
        def run():
            count = 0
            try:
                known = self.memory.chat_message_ids()
                uname = str((self.config.get("identity") or {}).get("user_name", "User"))
                aname = str((self.config.get("identity") or {}).get("name", "Assistant"))
                for c in self.chats.list_chats():
                    cid = c["id"]
                    for m in self.chats.messages(cid):
                        mid = str(m.get("id", ""))
                        text = str(m.get("content", "") or "").strip()
                        if not mid or mid in known or not text:
                            continue
                        role = str(m.get("role") or "user")
                        speaker = str(m.get("name") or (uname if role == "user" else aname))
                        try:
                            ts = datetime.fromtimestamp(float(m.get("timestamp", 0) or 0), timezone.utc).isoformat()
                        except Exception:
                            ts = datetime.now(timezone.utc).isoformat()
                        def progress(_i, _n, info):
                            try:
                                blob = self.memory.memory_blob(int(info.get("id")))
                                self.uiq.put(("event", ("neural_activity", {"mode":"REMEMBER", "label":"recovery", "blob":blob})))
                            except Exception:
                                pass
                        self.memory.store_message(text, role=role, speaker=speaker, chat_id=cid, message_id=mid, timestamp_utc=ts, progress=progress)
                        known.add(mid)
                        count += 1
            except Exception as exc:
                self.uiq.put(("event", ("warning", {"message":f"Startup neural reconciliation skipped: {exc}"})))
            self.uiq.put(("reconcile_done", count))
        threading.Thread(target=run, name="neural-chat-reconcile", daemon=True).start()

    def _on_close(self):
        if self.busy and not messagebox.askyesno("Exit", "A turn is still running. Stop it and exit?"):
            return
        try:
            self.scheduler.cancel(); self.scheduler.shutdown(); self.pool.stop_all()
        finally:
            self.root.destroy()
