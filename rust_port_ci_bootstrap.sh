#!/usr/bin/env bash
set -euo pipefail

rm -rf rust-ci-work
mkdir rust-ci-work
cat rust_ci_chunks/chunk*.b64 | base64 -d | tar -xzf - -C rust-ci-work

cd rust-ci-work
python - <<'PY'
from pathlib import Path
import re
pattern = re.compile(r'(?P<prefix>^|[=(:,\[\{+*/%!<>&|;\-])(?P<ws>\s*)\.(?=\d)')
for path in Path("crates").rglob("*.rs"):
    text = path.read_text()
    fixed = pattern.sub(lambda m: m.group("prefix") + m.group("ws") + "0.", text)
    if fixed != text:
        path.write_text(fixed)

# Temporary CI compatibility patches; corresponding fixes are already in the working source.
speaker = Path("crates/aevum-core/src/speaker.rs")
text = speaker.read_text()
text = text.replace("let cfg=&graph.cfg;let width=cfg.probe_channels*3;", "let cfg=&graph.cfg.clone();let width=cfg.probe_channels*3;")
text = text.replace("pub(crate) struct AdamVector{m:Vec<f32>,v:Vec<f32>}", "pub(crate) struct AdamVector{pub(crate) m:Vec<f32>,pub(crate) v:Vec<f32>}")
speaker.write_text(text)
PY
cargo fmt --all
cargo check --workspace --lib --bins
cargo test -p aevum-core
cargo clippy --workspace --lib --bins -- -D warnings
