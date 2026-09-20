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
    root.geometry("900x600")
    transcript = ChatTranscript(root)
    transcript.pack(fill="both", expand=True)
    view = transcript.add_assistant(name="Assistant", answer="", thinking="", complete=False)
    view.toggle_thoughts()

    checkpoints = []

    def first_chunk():
        view.append_thought("First constraint. ")
        root.update_idletasks()

    def check_first():
        text = view.thought_text.get("1.0", "end-1c")
        assert text == "First constraint. ", repr(text)
        assert view.thinking_live
        checkpoints.append("first-visible-before-final")

    def second_chunk():
        view.append_thought("Second constraint.")
        root.update_idletasks()

    def check_second_and_finish():
        text = view.thought_text.get("1.0", "end-1c")
        assert text == "First constraint. Second constraint.", repr(text)
        checkpoints.append("second-visible-before-final")
        # Native Granite must deliver its own final answer through the normal chat
        # token path. Reconciliation remains a safety net, not the answer generator.
        view.append_answer("D, B, E, C, A")
        root.update_idletasks()
        assert view.answer_text.get("1.0", "end-1c") == "D, B, E, C, A"
        checkpoints.append("answer-visible-through-normal-chat-path")
        view.finish()
        root.update_idletasks()
        root.destroy()

    root.after(40, first_chunk)
    root.after(100, check_first)
    root.after(160, second_chunk)
    root.after(240, check_second_and_finish)
    root.mainloop()

    assert checkpoints == [
        "first-visible-before-final",
        "second-visible-before-final",
        "answer-visible-through-normal-chat-path",
    ], checkpoints
    print("UI live-thinking + normal-answer probe: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
