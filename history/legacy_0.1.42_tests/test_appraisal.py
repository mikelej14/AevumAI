import unittest

from core.appraisal import DeterministicAppraiser
from core.schemas import MemoryPacket, ObserverPacket


class AppraisalTests(unittest.TestCase):
    def test_correction_creates_prediction_error_without_model(self):
        appraiser = DeterministicAppraiser()
        obs = ObserverPacket(
            focus="architecture", intent="correct design", goals=["fix architecture"],
            correction_detected=True, salience=0.9, confidence=0.9, frustration=0.4,
        )
        mem = MemoryPacket(summary="Previous design", contradictions=["old assumption conflicts"], confidence=0.8)
        packet = appraiser.pre(obs, mem, {"active_goals": ["fix architecture"]})
        self.assertGreater(packet.prediction_error, 0.6)
        self.assertGreater(packet.goal_relevance, 0.5)
        self.assertTrue(any("correction" in x for x in packet.state_bias))
        self.assertLessEqual(abs(packet.affect_deltas.get("confidence", 0.0)), 0.05)

    def test_social_signal_comes_from_observer_not_freeform_emotion(self):
        appraiser = DeterministicAppraiser()
        obs = ObserverPacket(hostility=0.9, insult=0.8, directed_at_system=True, salience=0.7, confidence=0.9)
        packet = appraiser.pre(obs, MemoryPacket(confidence=0.3), {})
        self.assertGreater(packet.social_negative, 0.75)
        self.assertEqual(packet.social_positive, 0.0)

    def test_post_appraisal_does_not_invent_user_approval(self):
        appraiser = DeterministicAppraiser()
        obs = ObserverPacket(salience=0.7, confidence=0.9)
        mem = MemoryPacket(confidence=0.8)
        pre = appraiser.pre(obs, mem, {})
        post = appraiser.post(obs, mem, pre, {
            "summary": "Assistant answered", "what_changed": ["answer produced"],
            "lesson": "", "salience": 0.5, "confidence": 0.9,
        })
        self.assertEqual(post.social_positive, 0.0)
        self.assertEqual(post.social_negative, 0.0)


if __name__ == "__main__":
    unittest.main()
