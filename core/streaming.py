from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class StreamPiece:
    channel: str  # "content" or "reasoning"
    text: str


class ThinkTagSplitter:
    """Incrementally separates <think>...</think> from visible content.

    Qwen3.5's embedded chat template can emit native thinking inside these tags.
    The parser keeps enough tail bytes to recognize tags split across streaming
    chunks and never redraws or rewrites already-emitted text.
    """

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self, *, start_in_think: bool = False) -> None:
        # Qwen3.5 native thinking is slightly unusual: with enable_thinking=True
        # its embedded GGUF template writes the opening <think> token into the
        # PROMPT.  The generated completion therefore commonly begins with raw
        # reasoning text and later emits only </think> before the visible answer.
        # start_in_think=True models that exact transport contract.
        self.in_think = bool(start_in_think)
        self.pending = ""
        self.saw_tag = bool(start_in_think)

    def feed(self, text: str) -> List[StreamPiece]:
        if not text:
            return []
        self.pending += text
        out: List[StreamPiece] = []
        while self.pending:
            marker = self.CLOSE if self.in_think else self.OPEN
            lower = self.pending.lower()
            idx = lower.find(marker)
            if idx >= 0:
                if idx:
                    out.append(StreamPiece("reasoning" if self.in_think else "content", self.pending[:idx]))
                self.pending = self.pending[idx + len(marker):]
                self.in_think = not self.in_think
                self.saw_tag = True
                continue

            # A marker may be split between chunks. Keep only the smallest suffix
            # that could still become the current marker and emit everything else.
            keep = 0
            max_prefix = min(len(marker) - 1, len(self.pending))
            pending_lower = self.pending.lower()
            for n in range(max_prefix, 0, -1):
                if pending_lower.endswith(marker[:n]):
                    keep = n
                    break
            flush_len = len(self.pending) - keep
            if flush_len:
                out.append(StreamPiece("reasoning" if self.in_think else "content", self.pending[:flush_len]))
                self.pending = self.pending[flush_len:]
            break
        return [p for p in out if p.text]

    def finish(self) -> List[StreamPiece]:
        if not self.pending:
            return []
        piece = StreamPiece("reasoning" if self.in_think else "content", self.pending)
        self.pending = ""
        return [piece] if piece.text else []


class ReasoningLoopGuard:
    """Detect obviously runaway/repeating hidden reasoning without judging content.

    Qwen3.5-2B's model card specifically warns that thinking mode can enter loops.
    This guard is intentionally conservative: it triggers only after a sizeable
    reasoning stream and either a hard character budget or repeated trailing word
    windows.  It never inspects or suppresses visible answer text.
    """

    def __init__(self, *, max_chars: int = 4000, window_words: int = 48, repeat_hits: int = 3) -> None:
        self.max_chars = max(1000, int(max_chars))
        self.window_words = max(16, int(window_words))
        self.repeat_hits = max(2, int(repeat_hits))
        self._text = ""
        self.triggered = False
        self.reason = ""

    def feed(self, text: str) -> bool:
        if self.triggered or not text:
            return self.triggered
        self._text += text
        if len(self._text) >= self.max_chars:
            self.triggered = True
            self.reason = f"reasoning exceeded {self.max_chars} characters"
            return True

        # Wait for enough material to avoid flagging legitimate short repetition.
        words = re.findall(r"\S+", self._text.lower())
        n = self.window_words
        if len(words) < n * (self.repeat_hits + 1):
            return False
        tail = words[-n:]
        prior = words[:-n]
        matches = 0
        # Count non-overlapping copies of the exact trailing window in prior text.
        i = 0
        while i <= len(prior) - n:
            if prior[i:i + n] == tail:
                matches += 1
                if matches >= self.repeat_hits:
                    self.triggered = True
                    self.reason = f"repeated {n}-word reasoning window"
                    return True
                i += n
            else:
                i += 1
        return False


def split_complete_thinking(text: str, *, start_in_think: bool = False) -> Tuple[str, str]:
    """Return (visible_content, reasoning) for a completed Qwen response.

    With native Qwen3.5 thinking, pass start_in_think=True because the embedded
    template injects the opening <think> in the prompt rather than the completion.
    """
    parser = ThinkTagSplitter(start_in_think=start_in_think)
    pieces = parser.feed(text) + parser.finish()
    visible = "".join(p.text for p in pieces if p.channel == "content")
    reasoning = "".join(p.text for p in pieces if p.channel == "reasoning")
    return visible, reasoning
