import unittest
from ui.chat_widgets import _markdown_segments, _reconcile_answer_text
from ui.main_window import _commit_final_answer


class RichChatTests(unittest.TestCase):
    def test_markdown_link_with_optional_space_is_clickable_segment(self):
        parts = list(_markdown_segments("Use [NOAA] (https://weather.gov/test) for details."))
        self.assertIn(("link", "NOAA", "https://weather.gov/test"), parts)
        visible = "".join(p[1] for p in parts)
        self.assertEqual(visible, "Use NOAA for details.")


    def test_final_answer_reconciliation_surfaces_unstreamed_result(self):
        self.assertEqual(_reconcile_answer_text("", "Recovered final answer."),
                         ("append", "Recovered final answer.", True))

    def test_final_answer_reconciliation_only_appends_missing_suffix(self):
        self.assertEqual(_reconcile_answer_text("Hello", "Hello world"),
                         ("append", " world", True))

    def test_final_answer_reconciliation_replaces_repaired_draft(self):
        self.assertEqual(_reconcile_answer_text("draft", "final answer"),
                         ("replace", "final answer", True))


    def test_turn_completion_commits_authoritative_answer_to_view(self):
        class FakeView:
            def __init__(self):
                self.reconciled = None
                self.finished = False
            def reconcile_answer(self, text):
                self.reconciled = text
            def finish(self):
                self.finished = True

        view = FakeView()
        committed = _commit_final_answer(view, {"assistant_text": "The final answer."})
        self.assertEqual(committed, "The final answer.")
        self.assertEqual(view.reconciled, "The final answer.")
        self.assertTrue(view.finished)

    def test_bare_url_and_basic_inline_formatting(self):
        parts = list(_markdown_segments("See https://example.com and **important** `code`."))
        self.assertIn(("link", "https://example.com", "https://example.com"), parts)
        self.assertIn(("bold", "important", ""), parts)
        self.assertIn(("code", "code", ""), parts)


if __name__ == "__main__":
    unittest.main()
