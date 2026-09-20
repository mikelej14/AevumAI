from __future__ import annotations

import re
import webbrowser
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any, Dict, List, Optional

from .theme import (
    ACCENT, ACCENT_DARK, ASSISTANT_BUBBLE, BG, BORDER, BORDER_BRIGHT, BORDER_SOFT,
    FONT, FONT_MONO, FONT_SMALL, SHADOW_SOFT, SURFACE, SURFACE_2, SURFACE_3,
    TEXT, TEXT_DIM, TEXT_MUTED, THINK_BG, TOOL_BG, TOOL_BORDER, TOOL_TEXT,
    USER_BORDER, USER_BUBBLE,
    assistant_initials,
)


_MARKDOWN_LINK_RE = re.compile(r"\[([^\]\n]+)\]\s*\((https?://[^)\s]+)\)", re.I)
_BARE_URL_RE = re.compile(r"https?://[^\s<>()\[\]{}]+", re.I)
_INLINE_TOKEN_RE = re.compile(r"(\*\*[^*\n]+\*\*|`[^`\n]+`)")


def _inline_segments(text: str):
    pos = 0
    for match in _INLINE_TOKEN_RE.finditer(text):
        if match.start() > pos:
            yield ("plain", text[pos:match.start()], "")
        token = match.group(0)
        if token.startswith("**"):
            yield ("bold", token[2:-2], "")
        else:
            yield ("code", token[1:-1], "")
        pos = match.end()
    if pos < len(text):
        yield ("plain", text[pos:], "")


def _plain_segments(text: str):
    pos = 0
    for match in _BARE_URL_RE.finditer(text):
        if match.start() > pos:
            yield from _inline_segments(text[pos:match.start()])
        raw = match.group(0)
        url = raw.rstrip(".,;:!?")
        trailing = raw[len(url):]
        yield ("link", url, url)
        if trailing:
            yield ("plain", trailing, "")
        pos = match.end()
    if pos < len(text):
        yield from _inline_segments(text[pos:])


def _markdown_segments(text: str):
    """Yield lightweight rich-chat segments: Markdown links, URLs, bold and code."""
    text = text or ""
    pos = 0
    for match in _MARKDOWN_LINK_RE.finditer(text):
        if match.start() > pos:
            yield from _plain_segments(text[pos:match.start()])
        yield ("link", match.group(1), match.group(2).rstrip(".,;:!?"))
        pos = match.end()
    if pos < len(text):
        yield from _plain_segments(text[pos:])


def _reconcile_answer_text(current: str, final: str) -> tuple[str, str, bool]:
    """Reconcile streamed UI text with the scheduler's authoritative final answer.

    Returns (mode, payload, changed):
      - ("none", "", False) when already exact
      - ("append", suffix, True) when the stream is a prefix of final
      - ("replace", final, True) when recovery/cleanup changed the answer

    The scheduler return value is authoritative because some valid runtime paths
    (reasoning recovery, buffered self-state/tool answers, citation cleanup) do not
    necessarily emit every final character through the live token callback.
    """
    current = str(current or "")
    final = str(final or "")
    if current == final:
        return "none", "", False
    if final.startswith(current):
        return "append", final[len(current):], True
    return "replace", final, True


class AssistantTurnView:
    def __init__(self, transcript: "ChatTranscript", parent: tk.Widget, *, name: str = "Assistant",
                 answer: str = "", thinking: str = "", tools: Optional[List[Dict[str, Any]]] = None,
                 complete: bool = False):
        self.transcript = transcript
        self.frame = tk.Frame(parent, bg=BG, bd=0, highlightthickness=0)
        self.frame.pack(fill="x", padx=(26, 70), pady=(9, 12))

        # Avatar sits outside the elevated message card.
        left = tk.Frame(self.frame, bg=BG)
        left.pack(side="left", anchor="n", padx=(0, 10), pady=(2, 0))
        avatar_shadow = tk.Frame(left, bg="#0b1d11", bd=0)
        avatar_shadow.pack()
        self.avatar = tk.Label(
            avatar_shadow, text=assistant_initials(name), bg=ACCENT, fg="#031008",
            width=3, height=1, font=("Segoe UI Semibold", 10), anchor="center",
            padx=2, pady=7,
        )
        self.avatar.pack(padx=(0, 1), pady=(0, 2))

        shadow = tk.Frame(self.frame, bg=SHADOW_SOFT, bd=0)
        shadow.pack(side="left", fill="x", expand=True, anchor="n")
        self.card = tk.Frame(shadow, bg=ASSISTANT_BUBBLE, bd=0, highlightthickness=1, highlightbackground=BORDER_SOFT)
        self.card.pack(fill="x", expand=True, padx=(0, 1), pady=(0, 3))

        head = tk.Frame(self.card, bg=ASSISTANT_BUBBLE)
        head.pack(fill="x", padx=14, pady=(11, 4))
        self.name = tk.Label(head, text=name, bg=ASSISTANT_BUBBLE, fg=TEXT,
                             font=("Segoe UI Semibold", 10), anchor="w")
        self.name.pack(side="left")
        self.meta = tk.Label(head, text="", bg=ASSISTANT_BUBBLE, fg=TEXT_DIM,
                             font=("Segoe UI", 8), anchor="w")
        self.meta.pack(side="left", padx=(8, 0))

        self.thought_count = 0
        self.thinking_live = not complete
        self.expanded = False
        self.thought_button = tk.Button(
            self.card, text="Thinking…" if not complete else "Thoughts",
            command=self.toggle_thoughts, bg=SURFACE, fg=TEXT_MUTED,
            activebackground=SURFACE_3, activeforeground=TEXT, relief="flat", bd=0,
            highlightthickness=1, highlightbackground=BORDER_SOFT,
            font=("Segoe UI", 9), anchor="w", padx=11, pady=7, cursor="hand2",
        )

        self.tool_activity: List[Dict[str, Any]] = []
        self.tool_count = 0
        self.tool_failures = 0
        self.tools_live = not complete
        self.tools_expanded = False
        self.tool_button = tk.Button(
            self.card, text="", command=self.toggle_tools, bg="#102321", fg=TOOL_TEXT,
            activebackground="#17312e", activeforeground=TEXT, relief="flat", bd=0,
            highlightthickness=1, highlightbackground=TOOL_BORDER,
            font=("Segoe UI Semibold", 9), anchor="w", padx=11, pady=7, cursor="hand2",
        )

        self._answer_raw = str(answer or "")
        self._link_serial = 0
        self.answer_text = tk.Text(
            self.card, height=1, wrap="word", bg=ASSISTANT_BUBBLE, fg=TEXT,
            relief="flat", bd=0, highlightthickness=0, font=("Segoe UI", 10),
            insertbackground=TEXT, selectbackground="#244b32", selectforeground=TEXT,
            cursor="arrow", takefocus=0, padx=0, pady=0,
        )
        self.answer_text.tag_configure("bold", font=("Segoe UI Semibold", 10))
        self.answer_text.tag_configure("code", font=FONT_MONO, background=SURFACE)
        self.answer_text.tag_configure("link", foreground="#75e99a", underline=True)
        self.answer_text.pack(fill="x", padx=14, pady=(4, 9))
        self.answer_text.bind("<Configure>", lambda _e: self.answer_text.after_idle(self._sync_answer_height), add="+")
        self.thought_button.pack(fill="x", padx=14, pady=(0, 12))
        if self._answer_raw:
            if complete:
                self._render_answer_markdown()
            else:
                self._append_answer_raw(self._answer_raw)

        self.thought_wrap = tk.Frame(self.card, bg=THINK_BG, highlightthickness=1, highlightbackground=BORDER)
        self.thought_text = ScrolledText(
            self.thought_wrap, height=8, wrap="word", font=FONT_MONO, state="disabled",
            relief="flat", borderwidth=0, bg=THINK_BG, fg="#bcc7d8", insertbackground=TEXT,
            selectbackground="#1f3c2a", selectforeground=TEXT,
        )
        self.thought_text.pack(fill="both", expand=True, padx=11, pady=10)
        self.thought_text.bind("<MouseWheel>", self._thought_mousewheel, add="+")
        self.thought_text.bind("<Button-4>", self._thought_mousewheel_linux, add="+")
        self.thought_text.bind("<Button-5>", self._thought_mousewheel_linux, add="+")

        self.tool_wrap = tk.Frame(self.card, bg=TOOL_BG, highlightthickness=1, highlightbackground=TOOL_BORDER)
        self.tool_text = tk.Text(
            self.tool_wrap, height=7, wrap="word", font=FONT_MONO, state="disabled",
            relief="flat", borderwidth=0, bg=TOOL_BG, fg=TOOL_TEXT, insertbackground=TEXT,
            selectbackground="#21452f", selectforeground=TEXT, cursor="arrow", takefocus=0,
        )
        self.tool_text.tag_configure("tool_head", foreground="#72e1c1", font=("Cascadia Mono Semibold", 9))
        self.tool_text.tag_configure("tool_fail", foreground="#ff9ca8", font=("Cascadia Mono Semibold", 9))
        self.tool_text.tag_configure("tool_dim", foreground="#789b95")
        self.tool_text.tag_configure("tool_link", foreground="#75e99a", underline=True)
        self.tool_text.pack(fill="both", expand=True, padx=11, pady=10)
        self.tool_text.bind("<MouseWheel>", self._tool_mousewheel, add="+")
        self.tool_text.bind("<Button-4>", self._tool_mousewheel_linux, add="+")
        self.tool_text.bind("<Button-5>", self._tool_mousewheel_linux, add="+")

        for w in (self.frame, left, avatar_shadow, self.avatar, shadow, self.card, head,
                  self.name, self.meta, self.thought_button, self.tool_button, self.answer_text):
            self.transcript.bind_chat_wheel(w)

        if thinking:
            self._append_thought_raw(thinking)
            self.thought_count = len(thinking)
        for item in tools or []:
            self.append_tool_event(dict(item), restored=True)
        if complete:
            self.finish()

    def _tool_mousewheel(self, event):
        units = int(-event.delta / 120) if event.delta else 0
        if units:
            self.tool_text.yview_scroll(units * 3, "units")
        return "break"

    def _tool_mousewheel_linux(self, event):
        self.tool_text.yview_scroll(-3 if event.num == 4 else 3, "units")
        return "break"

    def _thought_mousewheel(self, event):
        units = int(-event.delta / 120) if event.delta else 0
        if units:
            self.thought_text.yview_scroll(units * 3, "units")
        return "break"

    def _thought_mousewheel_linux(self, event):
        self.thought_text.yview_scroll(-3 if event.num == 4 else 3, "units")
        return "break"

    def toggle_thoughts(self):
        was_bottom = self.transcript.at_bottom()
        self.expanded = not self.expanded
        if self.expanded:
            self.thought_wrap.pack(fill="both", expand=False, padx=14, pady=(0, 9), before=self.thought_button)
            self.thought_text.see("end")
        else:
            self.thought_wrap.pack_forget()
        self._update_thought_button()
        self.transcript.after_layout_change(was_bottom)

    def toggle_tools(self):
        was_bottom = self.transcript.at_bottom()
        self.tools_expanded = not self.tools_expanded
        if self.tools_expanded:
            self.tool_wrap.pack(fill="both", expand=False, padx=14, pady=(0, 7), before=self.tool_button)
            self.tool_text.see("end")
        else:
            self.tool_wrap.pack_forget()
        self._update_tool_button()
        self.transcript.after_layout_change(was_bottom)

    def _show_tool_button(self):
        if not self.tool_button.winfo_manager():
            self.tool_button.pack(fill="x", padx=14, pady=(0, 7), before=self.thought_button)

    def _update_tool_button(self):
        arrow = "⌄" if self.tools_expanded else "›"
        if not self.tool_count:
            label = "Using tools…" if self.tools_live else "No tools used"
        elif self.tools_live:
            label = f"Tools · {self.tool_count} completed · working…"
        elif self.tool_failures:
            label = f"Tools · {self.tool_count} completed · {self.tool_failures} failed"
        else:
            label = f"Tools · {self.tool_count} verified"
        self.tool_button.configure(text=f"  {arrow}   {label}")

    @staticmethod
    def _tool_title(action: str) -> str:
        return {
            "current_weather": "CURRENT WEATHER",
            "web_search": "WEB SEARCH",
            "web_fetch": "WEB PAGE",
            "memory_search": "MEMORY SEARCH",
            "memory_get": "MEMORY RECORD",
        }.get(action, action.replace("_", " ").upper() or "TOOL")

    @staticmethod
    def _format_number(value: Any, suffix: str = "") -> str:
        if value is None or value == "":
            return ""
        try:
            number = float(value)
            shown = f"{number:.1f}".rstrip("0").rstrip(".")
        except Exception:
            shown = str(value)
        return shown + suffix

    def _insert_tool_link(self, label: str, url: str):
        tag = f"tool_url_{len(self.tool_text.tag_names())}_{self.tool_count}"
        self.tool_text.insert("end", label, ("tool_link", tag))
        self.tool_text.tag_bind(tag, "<Button-1>", lambda _e, u=url: self._open_link(u))
        self.tool_text.tag_bind(tag, "<Enter>", lambda _e: self.tool_text.configure(cursor="hand2"))
        self.tool_text.tag_bind(tag, "<Leave>", lambda _e: self.tool_text.configure(cursor="arrow"))

    def _append_tool_result(self, item: Dict[str, Any]):
        action = str(item.get("action", "") or "")
        ok = bool(item.get("ok", False))
        elapsed = item.get("elapsed_ms")
        timing = f" · {float(elapsed) / 1000:.2f}s" if isinstance(elapsed, (int, float)) else ""
        status = "✓" if ok else "✕"
        self.tool_text.insert("end", f"{status} {self._tool_title(action)}{timing}\n", "tool_head" if ok else "tool_fail")

        rows: List[tuple[str, str]] = []
        if action == "current_weather":
            rows.extend([
                ("Location", str(item.get("resolved_location") or item.get("location") or "")),
                ("Coordinates", ", ".join(x for x in (
                    self._format_number(item.get("latitude")), self._format_number(item.get("longitude"))
                ) if x)),
                ("Observed", str(item.get("observed_at") or "") + (f" · {item.get('timezone')}" if item.get("timezone") else "")),
                ("Condition", str(item.get("condition") or "")),
                ("Temperature", self._format_number(item.get("temperature_f"), " °F")),
                ("Feels like", self._format_number(item.get("apparent_temperature_f"), " °F")),
                ("Humidity", self._format_number(item.get("relative_humidity_percent"), "%")),
                ("Wind", self._format_number(item.get("wind_speed_mph"), " mph")),
            ])
        elif action == "web_search":
            rows.extend([
                ("Query", str(item.get("query") or "")),
                ("Provider", str(item.get("provider") or "")),
                ("Results", str(item.get("count", ""))),
            ])
        elif action == "web_fetch":
            rows.extend([
                ("Page", str(item.get("title") or "")),
                ("Retrieved", f"{int(item.get('chars', 0) or 0):,} characters"),
            ])
        elif action == "memory_search":
            rows.extend([("Query", str(item.get("query") or "")), ("Matches", str(item.get("count", "")))])
        elif action == "memory_get":
            rows.append(("Record", str(item.get("record_id") or "")))

        for key, value in rows:
            if value:
                self.tool_text.insert("end", f"  {key}: ", "tool_dim")
                self.tool_text.insert("end", value + "\n")
        error = str(item.get("error") or "").strip()
        if error:
            self.tool_text.insert("end", "  Error: ", "tool_dim")
            self.tool_text.insert("end", error + "\n", "tool_fail")
        source = str(item.get("source") or "").strip()
        url = str(item.get("source_url") or item.get("url") or "").strip()
        if url:
            self.tool_text.insert("end", "  Source: ", "tool_dim")
            self._insert_tool_link(source or url, url)
            self.tool_text.insert("end", "\n")
        for result in item.get("results", []) or []:
            result_url = str(result.get("url") or "").strip()
            title = str(result.get("title") or result_url).strip()
            if result_url and title:
                self.tool_text.insert("end", "  • ", "tool_dim")
                self._insert_tool_link(title[:120], result_url)
                self.tool_text.insert("end", "\n")
        self.tool_text.insert("end", "\n")

    def append_tool_event(self, item: Dict[str, Any], *, restored: bool = False):
        if not isinstance(item, dict):
            return
        was_bottom = self.transcript.at_bottom()
        self._show_tool_button()
        if item.get("stage") == "started":
            action = str(item.get("action", "") or "")
            subject = str(item.get("location") or item.get("query") or item.get("url") or "").strip()
            self.tool_text.configure(state="normal")
            self.tool_text.insert("end", f"○ {self._tool_title(action)}\n", "tool_head")
            self.tool_text.insert("end", f"  Retrieving {subject or 'evidence'}…\n\n", "tool_dim")
            self.tool_text.configure(state="disabled")
            if not self.tools_expanded:
                self.tools_expanded = True
                self.tool_wrap.pack(fill="both", expand=False, padx=14, pady=(0, 7), before=self.tool_button)
        else:
            self.tool_activity.append(dict(item))
            self.tool_count += 1
            if not bool(item.get("ok", False)):
                self.tool_failures += 1
            self.tool_text.configure(state="normal")
            self._append_tool_result(item)
            self.tool_text.configure(state="disabled")
            if restored and not self.tools_expanded:
                self.tools_expanded = True
                self.tool_wrap.pack(fill="both", expand=False, padx=14, pady=(0, 7), before=self.tool_button)
        self._update_tool_button()
        self.tool_text.see("end")
        self.transcript.after_layout_change(was_bottom)

    def _update_thought_button(self):
        arrow = "⌄" if self.expanded else "›"
        if self.thinking_live:
            label = "Thinking…"
        elif self.thought_count:
            label = f"Thinking · {self.thought_count:,} chars"
        else:
            label = "No separate thoughts"
        self.thought_button.configure(text=f"  {arrow}   {label}")

    def _append_thought_raw(self, text: str):
        at_bottom = self.thought_text.yview()[1] >= 0.995
        self.thought_text.configure(state="normal")
        self.thought_text.insert("end", text)
        self.thought_text.configure(state="disabled")
        if at_bottom:
            self.thought_text.see("end")

    def append_thought(self, text: str):
        if not text:
            return
        was_chat_bottom = self.transcript.at_bottom()
        self._append_thought_raw(text)
        self.thought_count += len(text)
        self._update_thought_button()
        if self.expanded:
            self.transcript.after_layout_change(was_chat_bottom)

    def _append_answer_raw(self, text: str):
        self.answer_text.configure(state="normal")
        self.answer_text.insert("end", text)
        self.answer_text.configure(state="disabled")
        self._sync_answer_height()

    def append_answer(self, text: str):
        if not text:
            return
        was_bottom = self.transcript.at_bottom()
        self._answer_raw += text
        self._append_answer_raw(text)
        self.transcript.after_layout_change(was_bottom)

    def reconcile_answer(self, final_text: str) -> bool:
        """Make the visible card match the scheduler's authoritative final answer.

        Live token callbacks are an optimization, not the source of truth. This closes
        the failure mode where a recovered/buffered answer exists in run_turn() and in
        the persisted chat transcript but never reached the Tk token queue.
        """
        mode, payload, changed = _reconcile_answer_text(self._answer_raw, final_text)
        if not changed:
            return False
        was_bottom = self.transcript.at_bottom()
        if mode == "append":
            self._answer_raw += payload
            self._append_answer_raw(payload)
        else:
            self._answer_raw = payload
            self._render_answer_markdown()
        self.transcript.after_layout_change(was_bottom)
        return True

    def _open_link(self, url: str):
        try:
            webbrowser.open(url, new=2)
        except Exception:
            pass

    def _render_answer_markdown(self):
        was_bottom = self.transcript.at_bottom()
        self.answer_text.configure(state="normal")
        self.answer_text.delete("1.0", "end")
        for tag in list(self.answer_text.tag_names()):
            if tag.startswith("url_"):
                self.answer_text.tag_delete(tag)
        self._link_serial = 0
        for kind, visible, url in _markdown_segments(self._answer_raw):
            if not visible:
                continue
            if kind == "link":
                self._link_serial += 1
                tag = f"url_{self._link_serial}"
                self.answer_text.insert("end", visible, ("link", tag))
                self.answer_text.tag_bind(tag, "<Button-1>", lambda _e, u=url: self._open_link(u))
                self.answer_text.tag_bind(tag, "<Enter>", lambda _e: self.answer_text.configure(cursor="hand2"))
                self.answer_text.tag_bind(tag, "<Leave>", lambda _e: self.answer_text.configure(cursor="arrow"))
            elif kind == "bold":
                self.answer_text.insert("end", visible, ("bold",))
            elif kind == "code":
                self.answer_text.insert("end", visible, ("code",))
            else:
                self.answer_text.insert("end", visible)
        self.answer_text.configure(state="disabled")
        self._sync_answer_height()
        self.transcript.after_layout_change(was_bottom)

    def _sync_answer_height(self):
        try:
            self.answer_text.update_idletasks()
            counted = self.answer_text.count("1.0", "end-1c", "displaylines")
            lines = int(counted[0]) if counted else max(1, self._answer_raw.count("\n") + 1)
        except Exception:
            lines = max(1, self._answer_raw.count("\n") + 1)
        self.answer_text.configure(height=max(1, lines))

    def append_error(self, text: str):
        self.append_answer(text)

    def finish(self):
        self.thinking_live = False
        self.tools_live = False
        self._render_answer_markdown()
        self._update_thought_button()
        if self.tool_count:
            self._update_tool_button()

    def set_wraplength(self, width: int):
        self.answer_text.after_idle(self._sync_answer_height)


class ChatTranscript(ttk.Frame):
    """Premium dark transcript with isolated thought scrolling."""

    def __init__(self, parent, empty_icon=None):
        super().__init__(parent, style="Surface.TFrame")
        self.empty_icon = empty_icon
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0, bg=BG)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = tk.Frame(self.canvas, bg=BG)
        self.window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.assistant_views: List[AssistantTurnView] = []
        self._empty = None

        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.bind_chat_wheel(self.canvas)
        self.bind_chat_wheel(self.inner)

    def _on_inner_configure(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self.window_id, width=event.width)
        for view in self.assistant_views:
            view.set_wraplength(event.width)

    def bind_chat_wheel(self, widget):
        widget.bind("<MouseWheel>", self._chat_mousewheel, add="+")
        widget.bind("<Button-4>", self._chat_mousewheel_linux, add="+")
        widget.bind("<Button-5>", self._chat_mousewheel_linux, add="+")

    def _chat_mousewheel(self, event):
        units = int(-event.delta / 120) if event.delta else 0
        if units:
            self.canvas.yview_scroll(units * 3, "units")
        return "break"

    def _chat_mousewheel_linux(self, event):
        self.canvas.yview_scroll(-3 if event.num == 4 else 3, "units")
        return "break"

    def at_bottom(self) -> bool:
        try:
            return self.canvas.yview()[1] >= 0.995
        except Exception:
            return True

    def after_layout_change(self, follow_if_was_bottom: bool):
        self.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        if follow_if_was_bottom:
            self.canvas.yview_moveto(1.0)

    def scroll_to_bottom(self):
        self.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.canvas.yview_moveto(1.0)

    def clear(self):
        for child in list(self.inner.winfo_children()):
            child.destroy()
        self.assistant_views.clear()
        self._empty = None
        self.canvas.yview_moveto(0.0)

    def show_empty(self, title: str = "Start a conversation",
                   subtitle: str = "Memory and subconscious processing will work quietly in the background."):
        if self.inner.winfo_children():
            return
        box = tk.Frame(self.inner, bg=BG)
        box.pack(fill="both", expand=True, padx=60, pady=(140, 60))
        halo = tk.Frame(box, bg=BG, bd=0)
        halo.pack()
        if self.empty_icon is not None:
            icon = tk.Label(halo, image=self.empty_icon, bg=BG)
        else:
            icon = tk.Label(halo, text="A", bg=ACCENT_DARK, fg=ACCENT, font=("Segoe UI Semibold", 18), padx=14, pady=10)
        icon.pack(padx=1, pady=1)
        tk.Label(box, text=title, bg=BG, fg=TEXT, font=("Segoe UI Semibold", 18)).pack(pady=(14, 5))
        tk.Label(box, text=subtitle, bg=BG, fg=TEXT_MUTED, font=FONT_SMALL,
                 wraplength=560, justify="center").pack()
        for w in (box, halo, icon):
            self.bind_chat_wheel(w)
        self._empty = box

    def _remove_empty(self):
        if self._empty is not None:
            try:
                self._empty.destroy()
            except Exception:
                pass
            self._empty = None

    def add_user(self, text: str):
        self._remove_empty()
        outer = tk.Frame(self.inner, bg=BG)
        outer.pack(fill="x", padx=(210, 26), pady=(8, 8))
        shadow = tk.Frame(outer, bg=SHADOW_SOFT)
        shadow.pack(side="right", fill="x")
        card = tk.Frame(shadow, bg=USER_BUBBLE, highlightthickness=1, highlightbackground=USER_BORDER)
        card.pack(fill="x", padx=(0, 1), pady=(0, 3))
        name = tk.Label(card, text="You", bg=USER_BUBBLE, fg="#d3f8df",
                        font=("Segoe UI Semibold", 9), anchor="w")
        name.pack(fill="x", padx=14, pady=(10, 2))
        msg = tk.Label(card, text=text, bg=USER_BUBBLE, fg=TEXT, justify="left",
                       anchor="nw", font=FONT, wraplength=760)
        msg.pack(fill="x", padx=14, pady=(2, 11))
        for w in (outer, shadow, card, name, msg):
            self.bind_chat_wheel(w)
        self.scroll_to_bottom()

    def add_assistant(self, *, name: str = "Assistant", answer: str = "", thinking: str = "",
                      tools: Optional[List[Dict[str, Any]]] = None,
                      complete: bool = False) -> AssistantTurnView:
        self._remove_empty()
        view = AssistantTurnView(
            self, self.inner, name=name, answer=answer, thinking=thinking, tools=tools, complete=complete
        )
        self.assistant_views.append(view)
        try:
            width = max(500, self.canvas.winfo_width())
            view.set_wraplength(width)
        except Exception:
            pass
        self.scroll_to_bottom()
        return view
