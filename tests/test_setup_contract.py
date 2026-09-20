from pathlib import Path

from core.config import DEFAULT_CONFIG
from current_best_decoder import SCALES

ROOT = Path(__file__).resolve().parents[1]


def test_setup_contract_and_optional_annotator():
    assert tuple(SCALES) == (0.5, 1.0, 2.0, 4.0)
    assert DEFAULT_CONFIG["chat"]["provider"] == "local_gguf"
    assert DEFAULT_CONFIG["models"]["annotator"]["enabled"] is False
    assert DEFAULT_CONFIG["models"]["executive"]["n_gpu_layers"] == -1
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "llama-cpp-python==" not in req
    setup = (ROOT / "SETUP.bat").read_text(encoding="utf-8")
    assert "whl/vulkan" in setup
    assert "AEVUM_BUILD" in setup
    assert "tools\\setup_probe.py --require-runtime-deps" in setup
    assert (ROOT / "assets" / "runtime" / "AevumAI-SPLASH_250x250.png").is_file()
    assert (ROOT / "RUN.bat").is_file()
    assert (ROOT / "RUN_DEBUG.bat").is_file()
