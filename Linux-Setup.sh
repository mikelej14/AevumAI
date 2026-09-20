#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
trap 'printf "\nSetup failed. Review the error above, then rerun Linux-Setup.sh.\n" >&2' ERR
[[ "$(uname -s)" == "Linux" ]] || fail "This installer is for Linux."
[[ $# -eq 0 ]] || fail "Usage: bash Linux-Setup.sh"

python_bin="${AEVUM_PYTHON:-}"
if [[ -z "$python_bin" ]]; then
  for candidate in python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      python_bin="$candidate"
      break
    fi
  done
fi
[[ -n "$python_bin" ]] || fail "Install Python 3.10+ with venv and Tkinter using your distribution's package manager."
"$python_bin" -c 'import sys, struct, tkinter; assert sys.version_info >= (3, 10), "Python 3.10+ is required"; assert struct.calcsize("P") == 8, "64-bit Python is required"' ||
  fail "Python 3.10+, 64-bit support, and Tkinter are required. On Debian/Ubuntu install python3, python3-venv, and python3-tk."

if [[ ! -d .venv ]]; then
  "$python_bin" -m venv .venv ||
    fail "Could not create .venv. Install your distribution's Python venv package and rerun."
fi
[[ -x .venv/bin/python ]] || fail "The existing .venv is not a Linux environment. Move it aside and rerun setup."
py="$PWD/.venv/bin/python"
"$py" -c 'import sys, struct, tkinter; assert sys.version_info >= (3, 10); assert struct.calcsize("P") == 8'
printf 'Aevum AI Linux setup: CPU backend\n'
"$py" -m pip install --upgrade pip setuptools wheel
if ! "$py" -m pip install --upgrade --force-reinstall --only-binary=:all: \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
  llama-cpp-python==0.3.35; then
  printf 'No compatible wheel found; building the runtime locally.\n'
  command -v cc >/dev/null 2>&1 && command -v c++ >/dev/null 2>&1 ||
    fail "Install a C/C++ toolchain (build-essential on Debian/Ubuntu) and rerun setup."
  "$py" -m pip install --upgrade cmake ninja
  CMAKE_ARGS="-DGGML_METAL=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF" FORCE_CMAKE=1 \
    "$py" -m pip install --upgrade --force-reinstall --no-cache-dir \
    --no-binary=llama-cpp-python llama-cpp-python==0.3.35
fi
"$py" -m pip install -r requirements.txt
"$py" tools/setup_probe.py --require-runtime-deps
printf '\nSetup complete. Run from a desktop session: bash Linux-Run.sh\nFollow Quick start, select Granite, and click Save & load models. Wait for Ready. See QUICKSTART.md.\n'
