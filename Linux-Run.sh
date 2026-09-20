#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ "$(uname -s)" != "Linux" ]]; then
  printf 'This launcher is for Linux.\n' >&2
  exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
  printf 'Run bash Linux-Setup.sh first.\n' >&2
  exit 1
fi
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  printf 'Aevum requires a graphical desktop session (Tkinter).\n' >&2
  exit 1
fi
exec .venv/bin/python app.py "$@"
