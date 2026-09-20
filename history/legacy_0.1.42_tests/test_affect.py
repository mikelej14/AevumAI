import tempfile
import unittest
from pathlib import Path

from core.affect import StateEngine
from core.schemas import AppraisalPacket, ObserverPacket


class AffectTests(unittest.TestCase):
    def test_hostility_is_signal_not_direct_emotion(self):
        with tempfile.TemporaryDirectory() as td:
            e = StateEngine(Path(td) / "state.json")
            before = e.state.affect.anger
            obs = ObserverPacket(hostility=1.0, insult=1.0, directed_at_system=True, salience=0.8)
            app = AppraisalPacket(confidence=0.8, affect_deltas={"anger": 0.1})
            e.apply(obs, app)
            self.assertGreater(e.state.affect.anger, before)
            self.assertLess(e.state.affect.anger, 0.5)  # bounded/inertial, not 'insult = rage'
            self.assertTrue(Path(td, "state.json").exists())

    def test_deltas_clamped(self):
        app = AppraisalPacket.from_dict({"affect_deltas":{"anger":999,"valence":-999}, "confidence": 2})
        self.assertEqual(app.affect_deltas["anger"], 0.25)
        self.assertEqual(app.affect_deltas["valence"], -0.25)
        self.assertEqual(app.confidence, 1.0)

if __name__ == "__main__":
    unittest.main()
