from __future__ import annotations

import re
import tkinter as tk
from tkinter import ttk

# Aevum AI — restrained obsidian + neon green.  Green is an accent, not a fill-everything color.
BG = "#030604"
BG_DEEP = "#010302"
SIDEBAR = "#060b08"
SIDEBAR_2 = "#0a110d"
SURFACE = "#0b120e"
SURFACE_2 = "#101813"
SURFACE_3 = "#162019"
SURFACE_4 = "#1b2a20"
BORDER = "#1b3d29"
BORDER_SOFT = "#102a1b"
BORDER_BRIGHT = "#2a5c3d"
TEXT = "#eef7f1"
TEXT_MUTED = "#9cb2a3"
TEXT_DIM = "#607568"
ACCENT = "#42ff7a"
ACCENT_2 = "#8affad"
ACCENT_HOVER = "#62ff91"
ACCENT_PRESSED = "#2cd867"
ACCENT_DARK = "#10341e"
ACCENT_GLOW = "#1b5f34"
USER_BUBBLE = "#0f2017"
USER_BORDER = "#236f43"
ASSISTANT_BUBBLE = "#0c1510"
THINK_BG = "#08100b"
TOOL_BG = "#08130c"
TOOL_BORDER = "#215438"
TOOL_TEXT = "#bee7ca"
SUCCESS = "#42ff7a"
WARNING = "#ffd24f"
ERROR = "#ff5f66"
SHADOW = "#010201"
SHADOW_SOFT = "#020403"

FONT = ("Segoe UI", 10)
FONT_SMALL = ("Segoe UI", 9)
FONT_TINY = ("Segoe UI", 8)
FONT_TITLE = ("Segoe UI Semibold", 16)
FONT_SECTION = ("Segoe UI Semibold", 11)
FONT_MONO = ("Cascadia Mono", 9)


def assistant_initials(name: str) -> str:
    """Return a compact dynamic avatar monogram from an assistant name."""
    words = [w for w in re.findall(r"[A-Za-z0-9]+", str(name or "")) if w]
    if not words:
        return "AI"
    if len(words) == 1:
        return words[0][:1].upper()
    return (words[0][0] + words[1][0]).upper()[:2]


class ElevatedButton(tk.Button):
    """Normal Tk button with Aevum styling.

    Deliberately uses Tk's own geometry/text/image layout so controls size correctly
    across DPI/font settings instead of manually painting text inside a Canvas.
    """

    _VARIANTS = {
        "primary": dict(bg="#0e1711", fg=ACCENT, activebackground="#15231a", activeforeground="#9bffb8", highlightbackground="#2a7047"),
        "secondary": dict(bg=SURFACE_3, fg=TEXT, activebackground=SURFACE_4, activeforeground=TEXT, highlightbackground=BORDER),
        "ghost": dict(bg=SURFACE_2, fg=TEXT_MUTED, activebackground=SURFACE_3, activeforeground=TEXT, highlightbackground=BORDER_SOFT),
        "danger": dict(bg="#321317", fg="#ffd7da", activebackground="#4b1c22", activeforeground="#ffffff", highlightbackground="#6e2a33"),
    }

    def __init__(self, parent, text="", command=None, variant="secondary", *, font=None,
                 padx=14, pady=8, anchor="center", cursor="hand2", image=None, **kwargs):
        self.variant = variant if variant in self._VARIANTS else "secondary"
        self.command = command
        self._image_ref = image
        colors = self._VARIANTS[self.variant]
        # Legacy callers passed pixel widths to the old Canvas control.  Real Tk
        # buttons size naturally; ignore those stale width/height hints.
        kwargs.pop("width", None)
        kwargs.pop("height", None)
        super().__init__(
            parent,
            text=text,
            command=command,
            image=image,
            compound="left" if image is not None else "none",
            bg=colors["bg"],
            fg=colors["fg"],
            activebackground=colors["activebackground"],
            activeforeground=colors["activeforeground"],
            disabledforeground=TEXT_DIM,
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=colors["highlightbackground"],
            highlightcolor=ACCENT,
            font=font or ("Segoe UI Semibold", 9),
            padx=padx,
            pady=pady,
            anchor=anchor,
            cursor=cursor,
            takefocus=True,
            **kwargs,
        )

    def configure(self, cnf=None, **kwargs):  # noqa: A003
        opts = dict(cnf or {}) if isinstance(cnf, dict) else {}
        opts.update(kwargs)
        if "image" in opts:
            self._image_ref = opts["image"]
            opts.setdefault("compound", "left")
        if "command" in opts:
            self.command = opts["command"]
        opts.pop("width", None)
        opts.pop("height", None)
        return super().configure(**opts)

    config = configure


class SurfaceCard(tk.Frame):
    """Nested surface with a subtle one-pixel lifted edge."""
    def __init__(self, parent, *, bg=SURFACE_2, shadow=True, border=BORDER_SOFT, **kwargs):
        outer_bg = SHADOW_SOFT if shadow else bg
        super().__init__(parent, bg=outer_bg, bd=0, highlightthickness=0, **kwargs)
        self.body = tk.Frame(self, bg=bg, bd=0, highlightthickness=1, highlightbackground=border)
        self.body.pack(fill="both", expand=True, padx=(0, 1), pady=(0, 2 if shadow else 0))


def apply_theme(root: tk.Tk) -> ttk.Style:
    root.configure(bg=BG)
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    style.configure(".", background=BG, foreground=TEXT, fieldbackground=SURFACE_2, bordercolor=BORDER, font=FONT)
    style.configure("TFrame", background=BG)
    style.configure("Surface.TFrame", background=SURFACE)
    style.configure("Sidebar.TFrame", background=SIDEBAR)
    style.configure("Card.TFrame", background=SURFACE_2, borderwidth=0)
    style.configure("TLabel", background=BG, foreground=TEXT)
    style.configure("Surface.TLabel", background=SURFACE, foreground=TEXT)
    style.configure("Sidebar.TLabel", background=SIDEBAR, foreground=TEXT)
    style.configure("Muted.TLabel", background=BG, foreground=TEXT_MUTED)
    style.configure("SurfaceMuted.TLabel", background=SURFACE, foreground=TEXT_MUTED)
    style.configure("Title.TLabel", background=BG, foreground=TEXT, font=FONT_TITLE)
    style.configure("SurfaceTitle.TLabel", background=SURFACE, foreground=TEXT, font=FONT_TITLE)
    style.configure("Section.TLabel", background=BG, foreground=TEXT, font=FONT_SECTION)
    style.configure("SurfaceSection.TLabel", background=SURFACE, foreground=TEXT, font=FONT_SECTION)

    style.configure("TButton", background=SURFACE_3, foreground=TEXT, borderwidth=0, focusthickness=0, padding=(12, 8))
    style.map("TButton", background=[("active", SURFACE_4), ("pressed", SURFACE_2), ("disabled", SURFACE)], foreground=[("disabled", TEXT_DIM)])
    style.configure("Accent.TButton", background=ACCENT, foreground="#ffffff", borderwidth=0, padding=(14, 9), font=("Segoe UI Semibold", 10))
    style.map("Accent.TButton", background=[("active", ACCENT_HOVER), ("pressed", ACCENT_PRESSED), ("disabled", ACCENT_DARK)], foreground=[("disabled", "#aab0dc")])
    style.configure("Ghost.TButton", background=SURFACE_2, foreground=TEXT_MUTED, borderwidth=0, padding=(10, 7))
    style.map("Ghost.TButton", background=[("active", SURFACE_3), ("pressed", SURFACE)], foreground=[("active", TEXT)])
    style.configure("Danger.TButton", background="#40232a", foreground="#f3b2b8", borderwidth=0, padding=(9, 7))
    style.map("Danger.TButton", background=[("active", "#5a2c36")], foreground=[("active", "#ffd3d7")])

    style.configure("TEntry", fieldbackground=SURFACE_2, foreground=TEXT, insertcolor=TEXT, bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER, padding=8)
    style.map("TEntry", bordercolor=[("focus", ACCENT)])
    style.configure("TCombobox", fieldbackground=SURFACE_2, background=SURFACE_2, foreground=TEXT, arrowcolor=TEXT_MUTED, bordercolor=BORDER, padding=7)
    style.map("TCombobox", fieldbackground=[("readonly", SURFACE_2)], foreground=[("readonly", TEXT)], selectbackground=[("readonly", SURFACE_3)], selectforeground=[("readonly", TEXT)])
    style.configure("TCheckbutton", background=BG, foreground=TEXT, indicatorcolor=SURFACE_2, indicatorbackground=SURFACE_2)
    style.map("TCheckbutton", background=[("active", BG)], foreground=[("active", TEXT)])
    style.configure("TScrollbar", background=SURFACE_3, troughcolor=BG_DEEP, bordercolor=BG_DEEP, arrowcolor=TEXT_MUTED, gripcount=0)
    style.map("TScrollbar", background=[("active", SURFACE_4)])
    style.configure("Horizontal.TSeparator", background=BORDER_SOFT)
    return style


def style_text(widget: tk.Text, *, mono: bool = False, surface: str = SURFACE, font=None) -> None:
    widget.configure(
        bg=surface,
        fg=TEXT,
        insertbackground=TEXT,
        selectbackground=ACCENT_DARK,
        selectforeground=TEXT,
        relief="flat",
        borderwidth=0,
        highlightthickness=1,
        highlightbackground=BORDER,
        highlightcolor=ACCENT,
        font=font or (FONT_MONO if mono else FONT),
    )
