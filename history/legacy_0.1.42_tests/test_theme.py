import unittest

from ui.theme import assistant_initials


class ThemeIdentityTests(unittest.TestCase):
    def test_dynamic_assistant_avatar_initials(self):
        self.assertEqual(assistant_initials("Nova"), "N")
        self.assertEqual(assistant_initials("Atlas"), "A")
        self.assertEqual(assistant_initials("Dr. Vale"), "DV")
        self.assertEqual(assistant_initials("Project Alpha"), "PA")
        self.assertEqual(assistant_initials(""), "AI")
