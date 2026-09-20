import copy
from pathlib import Path
import tempfile
import threading
import time
import tkinter as tk
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from core.config import DEFAULT_CONFIG
from ui.main_window import MainWindow
from ui.quickstart import model_file_error, needs_model_setup


class ModelSelectionTests(unittest.TestCase):
    def test_missing_wrong_type_and_incomplete_download(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.gguf"
            self.assertTrue(model_file_error(""))
            self.assertTrue(model_file_error(str(path)))
            path.write_bytes(b"version https://git-lfs.github.com/spec/v1")
            self.assertIn("not a GGUF", model_file_error(str(path)))
            path.write_bytes(b"GGUF" + b"test header")
            self.assertEqual(model_file_error(str(path)), "")

    def test_hosted_provider_does_not_require_local_setup(self):
        self.assertFalse(needs_model_setup({"chat": {"provider": "openai"}}))


class QuickStartIntegrationTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Tk display required: {exc}")
        self.root.withdraw()
        self.temp = tempfile.TemporaryDirectory(prefix="aevum model ")
        self.model = Path(self.temp.name) / "granite.gguf"
        self.model.write_bytes(b"GGUF" + b"test header")
        self.loaded = False
        self.pool = Mock()
        self.pool.statuses.side_effect = lambda: {"executive": {"loaded": self.loaded}}
        self.pool.reload.side_effect = self.reload
        self.pool.start.side_effect = lambda: self.reload(None)
        self.memory = Mock()
        self.memory.stats.return_value = {}
        self.memory.brain = SimpleNamespace(neural_documents=[])
        self.memory.chat_message_ids.return_value = set()
        self.scheduler = Mock()
        self.scheduler.state_engine.snapshot.return_value = {}
        self.save_patch = patch("ui.main_window.save_config")
        self.save_config = self.save_patch.start()
        self.dialog_patch = patch("ui.main_window.messagebox")
        self.dialogs = self.dialog_patch.start()

    def reload(self, _config):
        self.loaded = True
        return {}

    def window(self, config=None):
        return MainWindow(self.root, config or copy.deepcopy(DEFAULT_CONFIG),
                          self.pool, self.memory, self.scheduler)

    def wait_loaded(self, window):
        deadline = time.monotonic() + 3
        while window.models_loading and time.monotonic() < deadline:
            self.root.update()
            time.sleep(0.01)
        self.assertFalse(window.models_loading, "loader did not finish")

    def tearDown(self):
        if not hasattr(self, "temp"):
            return
        for timer in self.root.tk.call("after", "info"):
            self.root.tk.call("after", "cancel", timer)
        self.root.destroy()
        self.temp.cleanup()
        self.save_patch.stop()
        self.dialog_patch.stop()

    def test_first_launch_opens_quick_start(self):
        window = self.window()
        self.assertEqual(window.current_page, "quickstart")
        self.assertEqual(window.page_title_var.get(), "Welcome to Aevum")
        self.assertEqual(str(window.quickstart.chat_button["state"]), "disabled")

    def test_returning_user_keeps_chat_landing(self):
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["models"]["executive"]["path"] = str(self.model)
        window = self.window(config)
        self.assertEqual(window.current_page, "chat")

    def test_invalid_selection_does_not_save_or_start(self):
        window = self.window()
        window.quickstart.load_button.invoke()
        self.save_config.assert_not_called()
        self.pool.reload.assert_not_called()
        self.assertIn("Choose your Executive", window.quickstart.status.get())

    def test_save_load_ready_and_open_chat(self):
        window = self.window()
        window.model_entries["executive"].set(str(self.model))
        window.quickstart.load_button.invoke()
        self.assertTrue(window.models_loading)
        self.assertEqual(str(window.quickstart.load_button["state"]), "disabled")
        self.wait_loaded(window)
        self.save_config.assert_called_once()
        self.assertEqual(window.config["models"]["executive"]["path"], str(self.model))
        self.assertTrue(window.status_var.get().startswith("Ready"))
        window.quickstart.chat_button.invoke()
        self.assertEqual(window.current_page, "chat")

    def test_optional_model_failure_keeps_chat_available(self):
        def partial(_):
            self.loaded = True
            return {"annotator": "out of memory"}
        self.pool.reload.side_effect = partial
        window = self.window()
        window.model_entries["executive"].set(str(self.model))
        window.quickstart.load_button.invoke()
        self.wait_loaded(window)
        self.assertEqual(str(window.quickstart.chat_button["state"]), "normal")
        self.assertIn("still chat", window.quickstart.status.get())

    def test_loader_exception_restores_controls_and_shows_error(self):
        self.pool.reload.side_effect = RuntimeError("Not enough memory")
        window = self.window()
        window.model_entries["executive"].set(str(self.model))
        window.quickstart.load_button.invoke()
        self.wait_loaded(window)
        self.assertEqual(str(window.quickstart.load_button["state"]), "normal")
        self.assertEqual(str(window.quickstart.chat_button["state"]), "disabled")
        self.assertIn("Not enough memory", window.quickstart.status.get())

    def test_duplicate_load_and_send_are_blocked_during_startup(self):
        gate = threading.Event()
        def slow_reload(_):
            gate.wait(3)
            return self.reload(None)
        self.pool.reload.side_effect = slow_reload
        window = self.window()
        window.model_entries["executive"].set(str(self.model))
        try:
            window.quickstart.load_button.invoke()
            window._quickstart_load()
            window.start_models()
            window.send_message()
            self.assertEqual(self.pool.reload.call_count, 1)
            self.pool.start.assert_not_called()
            self.scheduler.run_turn.assert_not_called()
        finally:
            gate.set()
            self.wait_loaded(window)

    def test_controls_fit_minimum_window_size(self):
        window = self.window()
        self.root.attributes("-alpha", 0)
        self.root.deiconify()
        self.root.geometry("1080x700")
        self.root.update()
        button = window.quickstart.load_button
        self.assertTrue(button.winfo_ismapped())
        bottom = button.winfo_rooty() + button.winfo_height()
        self.assertLessEqual(bottom, self.root.winfo_rooty() + self.root.winfo_height())
        self.assertGreater(button.winfo_width(), 80)


if __name__ == "__main__":
    unittest.main()
