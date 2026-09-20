import os
import tempfile
import threading
import unittest
from pathlib import Path

from core.memory import CognitiveRMEM, MAGIC


class MemoryTests(unittest.TestCase):
    def test_binary_append_reopen_retrieve_validate(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.rmem"
            m = CognitiveRMEM(path)
            r1 = m.append("episode", {"summary": "User prefers native M3 decoding inside the chat client", "salience": 0.9})
            r2 = m.append("lesson", {"lesson": "Avoid unnecessary localhost service layers", "salience": 0.8})
            self.assertEqual(path.read_bytes()[:8], MAGIC)
            self.assertNotEqual(path.read_bytes()[:1], b"{")
            self.assertTrue(m.validate()["ok"])
            hits = m.retrieve("native M3 client localhost", limit=5)
            self.assertTrue(any(h.record_id in {r1, r2} for h in hits))
            m2 = CognitiveRMEM(path)
            self.assertEqual(m2.stats()["records"], 2)
            self.assertTrue(m2.validate()["ok"])

    def test_concurrent_append_and_retrieve_are_safe(self):
        with tempfile.TemporaryDirectory() as td:
            mem = CognitiveRMEM(Path(td) / "concurrent.rmem")
            errors = []

            def writer():
                try:
                    for i in range(120):
                        mem.append("episode", {"summary": f"parallel cobalt record {i}", "salience": 0.6})
                except Exception as exc:
                    errors.append(exc)

            def reader():
                try:
                    for _ in range(180):
                        mem.retrieve("parallel cobalt", limit=8)
                        mem.latest(count=5)
                        mem.stats()
                except Exception as exc:
                    errors.append(exc)

            t1 = threading.Thread(target=writer)
            t2 = threading.Thread(target=reader)
            t1.start(); t2.start(); t1.join(); t2.join()
            self.assertEqual(errors, [])
            self.assertEqual(mem.stats()["records"], 120)
            self.assertTrue(mem.validate()["ok"])

    def test_corrupt_tail_preserves_valid_prefix(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.rmem"
            m = CognitiveRMEM(path)
            m.append("turn", {"role": "user", "content": "hello durable world"})
            valid_size = path.stat().st_size
            with path.open("ab") as f:
                f.write(b"BROKEN_TAIL")
            m2 = CognitiveRMEM(path)
            self.assertEqual(m2.stats()["records"], 1)
            result = m2.validate()
            self.assertFalse(result["ok"])
            self.assertEqual(result["valid_bytes"], valid_size)

    def test_exclude_current_record(self):
        with tempfile.TemporaryDirectory() as td:
            m = CognitiveRMEM(Path(td) / "m.rmem")
            old = m.append("turn", {"role":"user", "content":"executive memory architecture"})
            cur = m.append("turn", {"role":"user", "content":"executive memory architecture latest"})
            hits = m.retrieve("executive memory architecture", exclude_ids=[cur])
            ids = [h.record_id for h in hits]
            self.assertIn(old, ids)
            self.assertNotIn(cur, ids)

if __name__ == "__main__":
    unittest.main()

class ResilientMemoryTests(unittest.TestCase):
    def test_missing_runtime_pack_degrades_without_recreating_headerless_file(self):
        from core.memory import ResilientMemory, HEADER
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "experience.rmem"
            mem = ResilientMemory(path)
            rid = mem.append("turn", {"role": "user", "content": "hello"})
            self.assertGreater(rid, 0)
            path.unlink()
            self.assertEqual(mem.append("turn", {"role": "user", "content": "after loss"}), 0)
            self.assertFalse(path.exists())
            self.assertFalse(mem.stats()["available"])
            self.assertEqual(mem.retrieve("hello"), [])
            self.assertTrue(mem.reopen())
            self.assertTrue(path.exists())
            self.assertGreaterEqual(path.stat().st_size, HEADER.size)
            self.assertTrue(mem.validate()["ok"])
