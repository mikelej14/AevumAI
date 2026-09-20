from __future__ import annotations

import pathlib
import sys
import tkinter as tk

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.chat_widgets import ChatTranscript


def main() -> int:
    root = tk.Tk()
    root.geometry("900x600+0+0")
    try:
        root.attributes("-alpha", 0.0)
    except tk.TclError:
        pass

    chat = ChatTranscript(root)
    chat.pack(fill="both", expand=True)
    for i in range(18):
        chat.add_user(f"older user message {i} " + ("x " * 20))
        chat.add_assistant(answer=f"older answer {i} " + ("y " * 30), complete=True)
    view = chat.add_assistant(name="Nova", answer="", complete=False)
    assert view.name.cget("text") == "Nova"
    view.toggle_thoughts()
    for i in range(100):
        view.append_thought(f"thought line {i}\n")
    view.append_answer("visible start " + ("z " * 80))
    root.update_idletasks(); root.update()

    # Manual transcript position is not stolen by streamed answer content.
    chat.canvas.yview_moveto(0.35); root.update()
    before = chat.canvas.yview()[0]
    view.append_answer("more streamed visible answer " * 10)
    root.update_idletasks(); root.update()
    assert abs(chat.canvas.yview()[0] - before) < 0.015

    # Manual thought position is not stolen by streamed thought content.
    view.thought_text.yview_moveto(0.30); root.update()
    before = view.thought_text.yview()[0]
    view.append_thought("new streamed thought that should not steal scroll\n" * 5)
    root.update_idletasks(); root.update()
    assert abs(view.thought_text.yview()[0] - before) < 0.02

    # Pointer over thought: only thought scrolls.
    chat.canvas.yview_moveto(0.40)
    view.thought_text.yview_moveto(0.40)
    root.update()
    chat_before = chat.canvas.yview()[0]
    thought_before = view.thought_text.yview()[0]
    view.thought_text.event_generate("<MouseWheel>", delta=-120)
    root.update()
    assert abs(chat.canvas.yview()[0] - chat_before) < 0.002
    assert view.thought_text.yview()[0] > thought_before

    # Pointer over normal chat: only transcript scrolls.
    chat.canvas.yview_moveto(0.40)
    view.thought_text.yview_moveto(0.40)
    root.update()
    chat_before = chat.canvas.yview()[0]
    thought_before = view.thought_text.yview()[0]
    view.answer_text.event_generate("<MouseWheel>", delta=-120)
    root.update()
    assert chat.canvas.yview()[0] > chat_before
    assert abs(view.thought_text.yview()[0] - thought_before) < 0.002

    # Each viewport follows its own live tail if it was already there.
    chat.scroll_to_bottom(); view.thought_text.see("end"); root.update()
    view.append_thought("tail thought\n")
    view.append_answer(" tail answer")
    root.update_idletasks(); root.update()
    assert chat.canvas.yview()[1] >= 0.995
    assert view.thought_text.yview()[1] >= 0.995

    print("UI identity/scroll/stream probe: PASS")
    root.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
