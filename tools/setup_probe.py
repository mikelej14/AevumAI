from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_ASSETS = (
    "AevumAI-BadgeLOGO_48x48.png",
    "AevumAI-Chat_18x18.png",
    "AevumAI-Chat_20x20.png",
    "AevumAI-Config_20x20.png",
    "AevumAI-Indicator-Load_18x18.png",
    "AevumAI-Indicator-OFF_18x18.png",
    "AevumAI-Indicator-ON_18x18.png",
    "AevumAI-Memory_20x20.png",
    "AevumAI-SPLASH_250x250.png",
    "AevumAI-Settings_20x20.png",
    "AevumAI-State_20x20.png",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate an Aevum AI installation without loading a GGUF.")
    ap.add_argument("--require-runtime-deps", action="store_true", help="also require llama_cpp and openai imports")
    args = ap.parse_args()

    require(sys.version_info >= (3, 10), "Python 3.10+ is required")
    print(f"Python: {platform.python_version()} ({platform.architecture()[0]})")
    if sys.version_info >= (3, 13):
        print("NOTE: Python 3.11/3.12 usually has the broadest llama-cpp prebuilt-wheel coverage; 3.13+ may source-build.")

    import tkinter  # noqa: F401
    print("Tkinter: OK")

    sys.path.insert(0, str(ROOT))
    from core.config import CONFIG_PATH, LOGS_DIR, MODELS_DIR, NEURAL_MEMORY_DIR, load_config
    from current_best_decoder import SCALES

    cfg = load_config()
    require(tuple(float(x) for x in SCALES) == (0.5, 1.0, 2.0, 4.0), f"unexpected decoder scales: {SCALES!r}")
    require(CONFIG_PATH.exists(), "default config was not created")
    require(MODELS_DIR.exists(), "models directory was not created")
    require(LOGS_DIR.exists(), "logs directory was not created")
    require(NEURAL_MEMORY_DIR.exists(), "neural memory directory was not created")
    require("executive" in cfg.get("models", {}), "Executive model slot missing")
    require("annotator" in cfg.get("models", {}), "optional semantic annotator slot missing")
    require(cfg["models"]["annotator"].get("enabled") in (True, False), "annotator enabled flag invalid")

    asset_dir = ROOT / "assets" / "runtime"
    missing = [name for name in REQUIRED_ASSETS if not (asset_dir / name).is_file()]
    require(not missing, "missing runtime asset(s): " + ", ".join(missing))
    print(f"Aevum runtime assets: {len(REQUIRED_ASSETS)}/{len(REQUIRED_ASSETS)} OK")
    print("Neural decoder scales: 0.5 / 1 / 2 / 4 ms")

    if args.require_runtime_deps:
        for name in ("llama_cpp", "openai"):
            try:
                mod = importlib.import_module(name)
            except Exception as exc:
                raise RuntimeError(f"required runtime dependency {name!r} could not import: {exc}") from exc
            print(f"{name}: {getattr(mod, '__version__', 'imported')}")

    print("Aevum setup probe: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Aevum setup probe: FAIL — {exc}", file=sys.stderr)
        raise SystemExit(1)
