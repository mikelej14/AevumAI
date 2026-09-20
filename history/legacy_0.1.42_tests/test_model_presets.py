import unittest

from core.model_presets import (
    GENERIC_PRESET,
    detect_preset,
    detect_preset_from_filename,
    sampler_profile,
    template_prompt_opens_thinking,
)


class ModelPresetTests(unittest.TestCase):
    def test_major_filename_families(self):
        cases = {
            "Qwen3.5-2B-Q8_0.gguf": "qwen35",
            "Qwen3-4B-Q4_K_M.gguf": "qwen3",
            "Qwen2.5-3B-Instruct-Q5_K_M.gguf": "qwen2",
            "granite-4.2-3b-Q5_K_S.gguf": "granite42",
            "gemma-2-2b-it-Q8_0.gguf": "gemma2",
            "gemma-3-4b-it-Q4_K_M.gguf": "gemma3",
            "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf": "mistral",
            "Mixtral-8x7B-Instruct-Q4_K_M.gguf": "mistral",
            "Llama-3.2-3B-Instruct-Q4_K_M.gguf": "llama",
            "Phi-4-mini-instruct-Q5_K_M.gguf": "phi",
            "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf": "deepseek_r1",
            "command-r7b-Q4_K_M.gguf": "commandr",
            "Yi-1.5-6B-Chat-Q4_K_M.gguf": "yi",
            "Falcon3-7B-Instruct-Q4_K_M.gguf": "falcon",
        }
        for filename, expected in cases.items():
            with self.subTest(filename=filename):
                self.assertEqual(detect_preset_from_filename(filename).key, expected)

    def test_metadata_wins_when_file_is_renamed(self):
        self.assertEqual(
            detect_preset("qwen35", "renamed-model", "mystery.gguf", "{{ enable_thinking }}<think>").key,
            "qwen35",
        )
        self.assertEqual(
            detect_preset("phi3", "renamed-model", "mystery.gguf", "{{ messages }}").key,
            "phi",
        )

    def test_granite_thinking_capability_identifies_42_contract_after_rename(self):
        p = detect_preset(
            "granite", "renamed-granite", "mystery.gguf",
            "{% if add_generation_prompt %}{% if enable_thinking %}<think>{% endif %}{% endif %}",
        )
        self.assertEqual(p.key, "granite42")
        self.assertTrue(p.thinking_default)
        self.assertEqual(sampler_profile(p)["temperature"], 1.0)
        self.assertEqual(sampler_profile(p)["top_p"], 0.95)

    def test_gemma_uses_private_primer_system_mode(self):
        self.assertEqual(detect_preset_from_filename("gemma-2-2b-it.gguf").system_mode, "primer")
        self.assertEqual(detect_preset_from_filename("gemma-3-4b-it.gguf").system_mode, "primer")

    def test_unknown_model_remains_allowed_generic(self):
        p = detect_preset("future_arch_99", "Future Model", "future-model.gguf", "{{ messages }}")
        self.assertEqual(p.key, GENERIC_PRESET.key)

    def test_prompt_open_thinking_detection(self):
        qwen_like = "{% if add_generation_prompt %}{% if enable_thinking %}<think>\\n{% endif %}{% endif %}"
        explicit_only = "{% for message in messages %}{{ message.content }}{% endfor %}"
        self.assertTrue(template_prompt_opens_thinking(qwen_like))
        self.assertFalse(template_prompt_opens_thinking(explicit_only))


if __name__ == "__main__":
    unittest.main()
