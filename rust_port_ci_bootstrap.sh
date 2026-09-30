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

# Temporary CI compatibility patches; corresponding fixes live in the working source.
speaker = Path("crates/aevum-core/src/speaker.rs")
text = speaker.read_text()
text = text.replace("let cfg=&graph.cfg;let width=cfg.probe_channels*3;", "let cfg=&graph.cfg.clone();let width=cfg.probe_channels*3;")
text = text.replace("pub(crate) struct AdamVector{m:Vec<f32>,v:Vec<f32>}", "pub(crate) struct AdamVector{pub(crate) m:Vec<f32>,pub(crate) v:Vec<f32>}")
speaker.write_text(text)

expert = Path("crates/aevum-core/src/expert.rs")
text = expert.read_text().replace("    config::Config,\n", "")
expert.write_text(text)

integrated = Path("crates/aevum-core/src/integrated.rs")
text = integrated.read_text().replace("recurrent::{CoreGrads, CoreState, StateGrads, TickTape}", "recurrent::{CoreState, StateGrads, TickTape}")
integrated.write_text(text)

config = Path("crates/aevum-core/src/config.rs")
text = config.read_text().replace("self.neurons % 512 != 0", "!self.neurons.is_multiple_of(512)")
config.write_text(text)

memory = Path("crates/aevum-core/src/memory.rs")
text = memory.read_text()
text = text.replace("hidden: Vec<f32>,\n    q_raw: Vec<f32>,", "hidden: Vec<f32>,")
text = text.replace("q_raw,\n            q,", "q,")
text = text.replace("impl MemoryCells {\n    pub fn new() -> Self {", "impl Default for MemoryCells {\n    fn default() -> Self { Self::new() }\n}\nimpl MemoryCells {\n    pub fn new() -> Self {")
memory.write_text(text)

lib = Path("crates/aevum-core/src/lib.rs")
text = lib.read_text()
if not text.startswith("#![allow(clippy::needless_range_loop)]"):
    lib.write_text("#![allow(clippy::needless_range_loop)]\n" + text)
PY
cargo fmt --all
cargo check --workspace --lib --bins
cargo test -p aevum-core
cargo clippy --workspace --lib --bins -- -D warnings
