import json
import tempfile
import unittest
from pathlib import Path

from core.chats import ChatStore


class ChatStoreTests(unittest.TestCase):
    def test_multichat_create_switch_delete_and_persist(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "chats.json"
            store = ChatStore(path)
            first = store.active_chat_id()
            m1 = store.add_message(first, "user", "First chat topic alpha")
            store.add_message(
                first, "assistant", "First answer", thinking="private",
                tools=[{"action": "current_weather", "ok": True, "source": "Open-Meteo"}],
            )
            second = store.create_chat("Second chat")["id"]
            store.add_message(second, "user", "Second chat topic beta")
            self.assertNotEqual(first, second)
            self.assertEqual(store.active_chat_id(), second)
            self.assertEqual([m["content"] for m in store.messages(first)], ["First chat topic alpha", "First answer"])
            self.assertEqual([m["content"] for m in store.messages(second)], ["Second chat topic beta"])
            self.assertEqual(store.recent_messages(first, turns=8)[0]["content"], "First chat topic alpha")
            self.assertTrue(m1.startswith("msg_"))

            reopened = ChatStore(path)
            self.assertEqual(reopened.active_chat_id(), second)
            self.assertEqual(len(reopened.list_chats()), 2)
            self.assertEqual(reopened.messages(first)[1]["tools"][0]["source"], "Open-Meteo")
            new_active = reopened.delete_chat(second)
            self.assertEqual(new_active, first)
            self.assertIsNone(reopened.get_chat(second))
            self.assertIsNotNone(reopened.get_chat(first))

    def test_corrupt_store_is_preserved_and_recovers(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "chats.json"
            path.write_text("{not json", encoding="utf-8")
            store = ChatStore(path)
            self.assertTrue(store.warning)
            self.assertEqual(len(store.list_chats()), 1)
            backups = list(Path(td).glob("chats.corrupt-*.json"))
            self.assertEqual(len(backups), 1)

    def test_before_message_scopes_recent_context(self):
        with tempfile.TemporaryDirectory() as td:
            store = ChatStore(Path(td) / "chats.json")
            cid = store.active_chat_id()
            store.add_message(cid, "user", "u1")
            store.add_message(cid, "assistant", "a1")
            current = store.add_message(cid, "user", "u2")
            hist = store.recent_messages(cid, turns=8, before_message_id=current)
            self.assertEqual(hist, [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}])


if __name__ == "__main__":
    unittest.main()

class ChatMigrationMarkerTests(unittest.TestCase):
    def test_legacy_import_marker_persists_and_prevents_reimport(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "chats.json"
            store = ChatStore(path)
            self.assertTrue(store.needs_legacy_import())
            store.mark_legacy_import_done()
            reopened = ChatStore(path)
            self.assertFalse(reopened.needs_legacy_import())
