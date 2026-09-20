#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ "$(uname -s)" != "Darwin" ]]; then
  printf 'This launcher is for macOS.\n' >&2
  exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
  printf 'Run bash macOS-Setup.command first.\n' >&2
  exit 1
fi
exec .venv/bin/python app.py "$@"
