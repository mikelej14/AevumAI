#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
trap 'printf "\nSetup failed. Review the error above, then rerun macOS-Setup.command.\n" >&2' ERR

[[ "$(uname -s)" == "Darwin" ]] || fail "This installer is for macOS."
backend="auto"
case "${1:-}" in
  "") ;;
  --cpu) backend="cpu" ;;
  *) fail "Usage: bash macOS-Setup.command [--cpu]" ;;
esac

python_bin="${AEVUM_PYTHON:-}"
if [[ -z "$python_bin" ]]; then
  for candidate in python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      python_bin="$candidate"
      break
    fi
  done
fi
[[ -n "$python_bin" ]] || fail "Install Python 3.11 or 3.12 from python.org, including Tkinter."
"$python_bin" -c 'import sys, struct, tkinter; assert sys.version_info >= (3, 10), "Python 3.10+ is required"; assert struct.calcsize("P") == 8, "64-bit Python is required"'

if [[ ! -d .venv ]]; then
  "$python_bin" -m venv .venv
fi
[[ -x .venv/bin/python ]] || fail "The existing .venv is not a macOS environment. Move it aside and rerun setup."
py="$PWD/.venv/bin/python"
"$py" -c 'import sys, struct, tkinter; assert sys.version_info >= (3, 10); assert struct.calcsize("P") == 8'
python_arch="$("$py" -c 'import platform; print(platform.machine())')"
apple_silicon="$(sysctl -n hw.optional.arm64 2>/dev/null || true)"
if [[ "$apple_silicon" == "1" && "$python_arch" != "arm64" ]]; then
  fail "Python is running under Rosetta. Use native arm64 Python and recreate .venv."
fi
if [[ "$backend" == "auto" ]]; then
  if [[ "$python_arch" == "arm64" ]]; then backend="metal"; else backend="cpu"; fi
fi

printf 'Aevum AI macOS setup: %s backend (%s)\n' "$backend" "$python_arch"
"$py" -m pip install --upgrade pip setuptools wheel
if ! "$py" -m pip install --upgrade --force-reinstall --only-binary=:all: \
  --extra-index-url "https://abetlen.github.io/llama-cpp-python/whl/$backend" \
  llama-cpp-python==0.3.35; then
  printf 'No compatible wheel found; building the runtime locally.\n'
  xcode-select -p >/dev/null 2>&1 || fail "Install Apple Command Line Tools with: xcode-select --install"
  "$py" -m pip install --upgrade cmake ninja
  if [[ "$backend" == "metal" ]]; then metal="ON"; else metal="OFF"; fi
  CMAKE_ARGS="-DGGML_METAL=$metal" FORCE_CMAKE=1 \
    "$py" -m pip install --upgrade --force-reinstall --no-cache-dir \
    --no-binary=llama-cpp-python llama-cpp-python==0.3.35
fi
"$py" -m pip install -r requirements.txt
"$py" tools/setup_probe.py --require-runtime-deps
printf '\nSetup complete. Run: bash macOS-Run.command\nSelect your Granite GGUF and optional Qwen GGUF in Settings, then Start model.\n'
