import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import core.config as config_module
from core.config import DEFAULT_CONFIG, _merge, normalize_identity_name
from core.qwen35_profile import QWEN35_SAMPLER_PROFILE_VERSION
from core.model_presets import GENERIC_TEXT as GENERIC_EXECUTIVE_TEXT
from core.model_worker import WorkerPool


class ConfigTests(unittest.TestCase):
    def test_custom_system_keeps_separate_personality_default(self):
        merged = _merge(DEFAULT_CONFIG, {"identity": {"system_prompt": "custom system"}})
        self.assertEqual(merged["identity"]["system_prompt"], "custom system")
        self.assertTrue(merged["identity"]["personality_prompt"])

    def test_default_system_documents_real_memory_tools(self):
        prompt = DEFAULT_CONFIG["identity"]["system_prompt"]
        self.assertNotIn("[memory_search]", prompt)
        self.assertNotIn("[web_search]", prompt)
        self.assertIn("native format only", prompt)
        self.assertIn("do not adopt any other assistant", prompt)
        self.assertIn("memory is useful context", prompt.lower())
        self.assertIn("maintained self-state", prompt.lower())
        self.assertIn("ai has no feelings", prompt.lower())
        self.assertIn("keep simple things simple", prompt.lower())
        self.assertNotIn("CORE RULES", prompt)
        self.assertLess(len(prompt), 1400)

    def test_identity_name_normalization(self):
        self.assertEqual(normalize_identity_name("  Nova   Ray  "), "Nova Ray")
        self.assertEqual(normalize_identity_name(""), "Assistant")
        self.assertLessEqual(len(normalize_identity_name("X" * 200)), 64)

    def test_exact_current_model_roles(self):
        self.assertEqual(tuple(DEFAULT_CONFIG["models"].keys()), ("executive", "qwen1", "qwen2"))
        self.assertEqual(WorkerPool.ROLES, ("executive", "qwen1", "qwen2"))
        self.assertNotIn("qwen3", DEFAULT_CONFIG["models"])

    def test_model_role_runtime_contract(self):
        executive = DEFAULT_CONFIG["models"]["executive"]
        q1 = DEFAULT_CONFIG["models"]["qwen1"]
        q2 = DEFAULT_CONFIG["models"]["qwen2"]

        # Executive has no architecture lock. The two subconscious workers do.
        self.assertNotIn("required_architecture", executive)
        self.assertNotIn("allowed_architectures", executive)
        self.assertEqual(executive["sampler_profile"], "auto")
        self.assertTrue(executive["enable_thinking"])
        for key, value in GENERIC_EXECUTIVE_TEXT.items():
            self.assertEqual(executive[key], value)

        self.assertEqual(q1["required_architecture"], "qwen35")
        self.assertEqual(q2["required_architecture"], "qwen35")
        self.assertFalse(q1["enable_thinking"])
        self.assertFalse(q2["enable_thinking"])
        for worker in (q1, q2):
            self.assertEqual(worker["temperature"], 1.0)
            self.assertEqual(worker["top_p"], 1.0)
            self.assertEqual(worker["top_k"], 20)
            self.assertEqual(worker["min_p"], 0.0)
            self.assertEqual(worker["presence_penalty"], 2.0)
            self.assertEqual(worker["frequency_penalty"], 0.0)
            self.assertEqual(worker["repeat_penalty"], 1.0)

        self.assertEqual(DEFAULT_CONFIG["runtime"]["qwen35_sampler_profile_version"], QWEN35_SAMPLER_PROFILE_VERSION)
        self.assertEqual(DEFAULT_CONFIG["runtime"]["executive_model_contract_version"], 3)

    def test_stale_qwen_locked_config_is_unlocked_without_touching_worker_contract(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config.json"
            cfg_path.write_text(json.dumps({
                "models": {
                    "executive": {
                        "temperature": 0.65,
                        "required_architecture": "qwen35",
                        "allowed_architectures": ["qwen35"],
                    },
                    "qwen1": {"temperature": 0.10, "top_p": 0.80, "presence_penalty": 0.0},
                    "qwen2": {"temperature": 0.15, "top_p": 0.85, "presence_penalty": 0.0},
                },
                "runtime": {},
            }), encoding="utf-8")
            with patch.object(config_module, "CONFIG_PATH", cfg_path):
                cfg = config_module.load_config()

            # Unknown old Executive lock keys are discarded by schema merge.
            self.assertNotIn("required_architecture", cfg["models"]["executive"])
            self.assertNotIn("allowed_architectures", cfg["models"]["executive"])
            self.assertEqual(cfg["models"]["executive"]["sampler_profile"], "auto")
            # Existing user sampler values may remain stored but Auto decides runtime defaults.
            self.assertEqual(cfg["models"]["executive"]["temperature"], 0.65)

            # CPU workers are repaired to their canonical Qwen profile.
            self.assertEqual(cfg["models"]["qwen1"]["top_p"], 1.0)
            self.assertEqual(cfg["models"]["qwen1"]["presence_penalty"], 2.0)
            self.assertEqual(cfg["models"]["qwen2"]["top_p"], 1.0)
            self.assertEqual(cfg["models"]["qwen2"]["presence_penalty"], 2.0)
            saved = json.loads(cfg_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["runtime"]["executive_model_contract_version"], 3)

    def test_obsolete_unknown_config_keys_are_discarded(self):
        merged = _merge(DEFAULT_CONFIG, {
            "models": {"qwen3": {"path": "old.gguf"}, "granite": {"path": "old2.gguf"}},
            "runtime": {"granite_tool_rounds": 99},
        })
        self.assertNotIn("qwen3", merged["models"])
        self.assertNotIn("granite", merged["models"])
        self.assertNotIn("granite_tool_rounds", merged["runtime"])

    def test_old_architecture_first_default_prompt_is_migrated_but_custom_prompt_is_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config.json"
            cfg_path.write_text(json.dumps({
                "identity": {
                    "prompt_schema_version": 7,
                    "system_prompt": "You are the waking conscious executive of a persistent local cognitive system. Old default text.",
                },
                "runtime": {"qwen35_sampler_profile_version": QWEN35_SAMPLER_PROFILE_VERSION},
            }), encoding="utf-8")
            with patch.object(config_module, "CONFIG_PATH", cfg_path):
                cfg = config_module.load_config()
            self.assertEqual(cfg["identity"]["prompt_schema_version"], 15)
            self.assertNotIn("QUICK REFERENCE", cfg["identity"]["system_prompt"])

        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config.json"
            cfg_path.write_text(json.dumps({
                "identity": {
                    "prompt_schema_version": 7,
                    "system_prompt": "My deliberately custom operating prompt.",
                },
                "runtime": {"qwen35_sampler_profile_version": QWEN35_SAMPLER_PROFILE_VERSION},
            }), encoding="utf-8")
            with patch.object(config_module, "CONFIG_PATH", cfg_path):
                cfg = config_module.load_config()
            self.assertEqual(cfg["identity"]["system_prompt"], "My deliberately custom operating prompt.")
            self.assertEqual(cfg["identity"]["prompt_schema_version"], 15)

    def test_v12_rule_list_default_migrates_to_compact_reference(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config.json"
            cfg_path.write_text(json.dumps({
                "identity": {
                    "prompt_schema_version": 12,
                    "system_prompt": "You are a persistent conversational assistant speaking directly with the user.\n\nCORE RULES\n1. Answer the user's actual message naturally and directly.\n2. old shipped defaults",
                },
                "runtime": {"qwen35_sampler_profile_version": QWEN35_SAMPLER_PROFILE_VERSION},
            }), encoding="utf-8")
            with patch.object(config_module, "CONFIG_PATH", cfg_path):
                cfg = config_module.load_config()
            self.assertEqual(cfg["identity"]["prompt_schema_version"], 15)
            self.assertNotIn("QUICK REFERENCE", cfg["identity"]["system_prompt"])
            self.assertNotIn("CORE RULES", cfg["identity"]["system_prompt"])

    def test_v13_bracket_tool_prompt_is_replaced_without_touching_custom_prompts(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config.json"
            cfg_path.write_text(json.dumps({
                "identity": {
                    "prompt_schema_version": 13,
                    "system_prompt": (
                        "You are a persistent conversational assistant speaking directly with the user.\n\n"
                        "QUICK REFERENCE\n[memory_search] Search memory.\n"
                        "[web_search] Search the web.\n[web_fetch] Fetch a page."
                    ),
                },
                "runtime": {"qwen35_sampler_profile_version": QWEN35_SAMPLER_PROFILE_VERSION},
            }), encoding="utf-8")
            with patch.object(config_module, "CONFIG_PATH", cfg_path):
                cfg = config_module.load_config()
            self.assertEqual(cfg["identity"]["prompt_schema_version"], 15)
            self.assertNotIn("[web_search]", cfg["identity"]["system_prompt"])
            self.assertIn("native format only", cfg["identity"]["system_prompt"])

    def test_v14_shipped_self_state_prompt_migrates_to_stronger_v15_contract(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config.json"
            cfg_path.write_text(json.dumps({
                "identity": {
                    "prompt_schema_version": 14,
                    "system_prompt": (
                        "You are a persistent local conversational assistant speaking directly with the user. "
                        "Internal state is your own quiet functional state; let it influence tone and judgement naturally without reporting scores or diagnostics."
                    ),
                },
                "runtime": {"qwen35_sampler_profile_version": QWEN35_SAMPLER_PROFILE_VERSION},
            }), encoding="utf-8")
            with patch.object(config_module, "CONFIG_PATH", cfg_path):
                cfg = config_module.load_config()
            self.assertEqual(cfg["identity"]["prompt_schema_version"], 15)
            self.assertIn("maintained SELF-STATE", cfg["identity"]["system_prompt"])
            self.assertIn("AI has no feelings", cfg["identity"]["system_prompt"])

    def test_v125_default_thinking_guard_migrates_but_custom_value_is_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config.json"
            cfg_path.write_text(json.dumps({
                "models": {"executive": {"thinking_max_chars": 12000}},
                "runtime": {
                    "qwen35_sampler_profile_version": QWEN35_SAMPLER_PROFILE_VERSION,
                    "executive_model_contract_version": 3,
                },
                "identity": {"prompt_schema_version": 15, "system_prompt": "custom"},
            }), encoding="utf-8")
            with patch.object(config_module, "CONFIG_PATH", cfg_path):
                cfg = config_module.load_config()
            self.assertEqual(cfg["models"]["executive"]["thinking_max_chars"], 4000)
            self.assertEqual(cfg["runtime"]["thinking_guard_contract_version"], 2)

        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config.json"
            cfg_path.write_text(json.dumps({
                "models": {"executive": {"thinking_max_chars": 6500}},
                "runtime": {
                    "qwen35_sampler_profile_version": QWEN35_SAMPLER_PROFILE_VERSION,
                    "executive_model_contract_version": 3,
                },
                "identity": {"prompt_schema_version": 15, "system_prompt": "custom"},
            }), encoding="utf-8")
            with patch.object(config_module, "CONFIG_PATH", cfg_path):
                cfg = config_module.load_config()
            self.assertEqual(cfg["models"]["executive"]["thinking_max_chars"], 6500)
            self.assertEqual(cfg["runtime"]["thinking_guard_contract_version"], 2)


if __name__ == "__main__":
    unittest.main()
