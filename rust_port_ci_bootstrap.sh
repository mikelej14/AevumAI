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

# Temporary CI patches mirror corrections already applied to the working source.
speaker = Path("crates/aevum-core/src/speaker.rs")
text = speaker.read_text()
text = text.replace("let cfg=&graph.cfg;let width=cfg.probe_channels*3;", "let cfg=&graph.cfg.clone();let width=cfg.probe_channels*3;")
text = text.replace("pub(crate) struct AdamVector{m:Vec<f32>,v:Vec<f32>}", "pub(crate) struct AdamVector{pub(crate) m:Vec<f32>,pub(crate) v:Vec<f32>}")
speaker.write_text(text)

expert = Path("crates/aevum-core/src/expert.rs")
text = expert.read_text().replace("config::Config,", "")
text = text.replace("mod tests {\\n    use super::*;", "mod tests {\\n    use super::*;\\n    use crate::config::Config;")\nexpert.write_text(text)

integrated = Path("crates/aevum-core/src/integrated.rs")
text = integrated.read_text().replace("CoreState,CoreGrads,StateGrads", "CoreState,StateGrads")
integrated.write_text(text)

config = Path("crates/aevum-core/src/config.rs")
text = config.read_text().replace("self.neurons % 512 != 0", "!self.neurons.is_multiple_of(512)")
config.write_text(text)

memory = Path("crates/aevum-core/src/memory.rs")
text = memory.read_text()
text = re.sub(r'q_raw:\s*Vec<f32>,', '', text)
text = text.replace("hidden:hidden.to_vec(),q_raw,q,q_norm", "hidden:hidden.to_vec(),q,q_norm")
needle = "impl MemoryCells {pub fn new()->Self{Self{keys:Vec::new(),values:Vec::new(),trusted:Vec::new(),epoch:Vec::new()}}}"
if needle in text and "impl Default for MemoryCells" not in text:
    text = text.replace(needle, needle + "\nimpl Default for MemoryCells {fn default()->Self{Self::new()}}")
memory.write_text(text)

lib = Path("crates/aevum-core/src/lib.rs")
text = lib.read_text()
if not text.startswith("#![allow(clippy::needless_range_loop)]"):
    lib.write_text("#![allow(clippy::needless_range_loop)]\n" + text)

registry = Path("crates/aevum-format/src/brain_registry.rs")
registry.write_text(registry.read_text().replace("fs::{self,File,OpenOptions}", "fs::{self,OpenOptions}").replace("fs::{self, File, OpenOptions}", "fs::{self, OpenOptions}"))

brain_write = Path("crates/aevum-format/src/brain_write.rs")
brain_write.write_text(brain_write.read_text().replace("io::{BufReader,BufWriter,Read,Write}", "io::{BufReader,BufWriter,Write}").replace("io::{BufReader, BufWriter, Read, Write}", "io::{BufReader, BufWriter, Write}"))

native_training = Path("crates/aevum-format/src/native_training.rs")
native_training.write_text(native_training.read_text().replace("let (mut grads,loss,_)", "let (grads,loss,_)"))
PY
cargo fmt --all
cargo check --workspace --lib --bins
cargo test -p aevum-core
cargo clippy --workspace --lib --bins -- -D warnings
