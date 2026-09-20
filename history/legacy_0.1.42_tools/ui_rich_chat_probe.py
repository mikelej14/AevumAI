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
    root.geometry("1000x720")
    transcript = ChatTranscript(root)
    transcript.pack(fill="both", expand=True)
    answer = (
        "Current conditions are mild. See [NOAA] (https://weather.gov/example) "
        "and https://example.com/details for the source.\n\n" + "detail line\n" * 95
    )
    view = transcript.add_assistant(
        name="Nova", answer=answer, thinking="private thought",
        tools=[{
            "stage": "completed", "action": "current_weather", "ok": True,
            "elapsed_ms": 242.0, "resolved_location": "Manchester, New Hampshire, United States",
            "latitude": 42.9956, "longitude": -71.4548, "timezone": "America/New_York",
            "observed_at": "2026-09-16T01:00", "condition": "clear sky",
            "temperature_f": 59.0, "apparent_temperature_f": 56.7,
            "relative_humidity_percent": 70, "wind_speed_mph": 5.1,
            "source": "Open-Meteo", "source_url": "https://api.open-meteo.com/v1/forecast",
        }], complete=True,
    )
    root.update_idletasks(); root.update()

    rendered = view.answer_text.get("1.0", "end-1c")
    assert "[NOAA]" not in rendered, rendered
    assert "NOAA" in rendered
    assert "https://example.com/details" in rendered
    link_tags = [t for t in view.answer_text.tag_names() if t.startswith("url_")]
    assert len(link_tags) >= 2, link_tags
    for tag in link_tags:
        assert view.answer_text.tag_ranges(tag), tag
        assert root.tk.call(view.answer_text._w, "tag", "bind", tag, "<Button-1>"), tag
    # Long messages must expand into the outer chat scroll, not clip inside a fixed-height text box.
    assert int(view.answer_text.cget("height")) > 80, view.answer_text.cget("height")
    tool_receipt = view.tool_text.get("1.0", "end-1c")
    assert "CURRENT WEATHER" in tool_receipt, tool_receipt
    assert "Manchester, New Hampshire" in tool_receipt, tool_receipt
    assert "Observed: 2026-09-16T01:00" in tool_receipt, tool_receipt
    assert "Temperature: 59 °F" in tool_receipt, tool_receipt
    assert "Open-Meteo" in tool_receipt, tool_receipt
    assert "Tools · 1 verified" in view.tool_button.cget("text"), view.tool_button.cget("text")
    tool_links = [t for t in view.tool_text.tag_names() if t.startswith("tool_url_")]
    assert tool_links, view.tool_text.tag_names()
    assert view.tools_expanded

    root.destroy()
    print("UI rich-answer/link probe: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
