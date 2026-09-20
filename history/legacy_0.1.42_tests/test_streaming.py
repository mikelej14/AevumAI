import unittest

from core.streaming import ReasoningLoopGuard, ThinkTagSplitter, split_complete_thinking


class StreamingTests(unittest.TestCase):
    def test_fragmented_think_tags_are_separated_without_leaking(self):
        splitter = ThinkTagSplitter()
        chunks = ["<thi", "nk>plan ", "carefully</thi", "nk>Final ", "answer"]
        content = []
        reasoning = []
        for chunk in chunks:
            for piece in splitter.feed(chunk):
                (reasoning if piece.channel == "reasoning" else content).append(piece.text)
        for piece in splitter.finish():
            (reasoning if piece.channel == "reasoning" else content).append(piece.text)
        self.assertEqual("".join(reasoning), "plan carefully")
        self.assertEqual("".join(content), "Final answer")

    def test_complete_split(self):
        visible, thinking = split_complete_thinking("<think>hidden</think>visible")
        self.assertEqual(thinking, "hidden")
        self.assertEqual(visible, "visible")

    def test_plain_response_stays_visible(self):
        visible, thinking = split_complete_thinking("ordinary answer")
        self.assertEqual(visible, "ordinary answer")
        self.assertEqual(thinking, "")


    def test_qwen35_prompt_injected_open_think_routes_initial_text_to_reasoning(self):
        # Native Qwen3.5 enable_thinking=True puts <think> in the prompt, so the
        # completion can start directly with reasoning and only emit </think>.
        splitter = ThinkTagSplitter(start_in_think=True)
        chunks = ["native ", "thought</thi", "nk>Visible ", "answer"]
        content = []
        reasoning = []
        for chunk in chunks:
            for piece in splitter.feed(chunk):
                (reasoning if piece.channel == "reasoning" else content).append(piece.text)
        for piece in splitter.finish():
            (reasoning if piece.channel == "reasoning" else content).append(piece.text)
        self.assertEqual("".join(reasoning), "native thought")
        self.assertEqual("".join(content), "Visible answer")

    def test_complete_qwen35_prompt_injected_open_think(self):
        visible, thinking = split_complete_thinking(
            "native thought</think>visible", start_in_think=True
        )
        self.assertEqual(thinking, "native thought")
        self.assertEqual(visible, "visible")

    def test_multiple_think_answer_phases_are_routed_independently(self):
        splitter = ThinkTagSplitter(start_in_think=True)
        chunks = ["first thought</think>first answer<think>second thought</think>final answer"]
        content, reasoning = [], []
        for chunk in chunks:
            for piece in splitter.feed(chunk):
                (reasoning if piece.channel == "reasoning" else content).append(piece.text)
        for piece in splitter.finish():
            (reasoning if piece.channel == "reasoning" else content).append(piece.text)
        self.assertEqual("".join(reasoning), "first thoughtsecond thought")
        self.assertEqual("".join(content), "first answerfinal answer")

    def test_reasoning_loop_guard_detects_repeated_tail(self):
        guard = ReasoningLoopGuard(max_chars=50000, window_words=16, repeat_hits=2)
        block = " ".join(f"w{i}" for i in range(16)) + " "
        self.assertFalse(guard.feed("intro words " * 20 + block))
        self.assertFalse(guard.feed(block))
        self.assertTrue(guard.feed(block))
        self.assertTrue(guard.triggered)
        self.assertIn("repeated", guard.reason)

    def test_reasoning_loop_guard_hard_cap(self):
        guard = ReasoningLoopGuard(max_chars=1000, window_words=16, repeat_hits=3)
        self.assertTrue(guard.feed("x" * 1000))
        self.assertIn("exceeded", guard.reason)


if __name__ == "__main__":
    unittest.main()
