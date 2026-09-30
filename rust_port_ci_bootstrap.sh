#!/usr/bin/env bash
set -euo pipefail

rm -rf rust-ci-work
mkdir rust-ci-work
cat rust_ci_chunks/chunk*.b64 | base64 -d | tar -xzf - -C rust-ci-work

cd rust-ci-work
python - <<'PY'
from pathlib import Path
import re
pattern = re.compile(r'(?P<prefix>^|[=(:,\[\{+*/%!<>?&|;\-])(?P<ws>\s*)\.(?=\d)')
for path in Path("crates").rglob("*.rs"):
    text = path.read_text()
    fixed = pattern.sub(lambda m: m.group("prefix") + m.group("ws") + "0.", text)
    if fixed != text:
        path.write_text(fixed)
PY
cargo fmt --all
cargo check --workspace --lib --bins
cargo test -p aevum-core
cargo clippy --workspace --lib --bins -- -D warnings
