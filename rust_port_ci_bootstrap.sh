#!/usr/bin/env bash
set -euo pipefail

rm -rf rust-ci-work
mkdir rust-ci-work
cat rust_ci_chunks/chunk*.b64 | base64 -d | tar -xzf - -C rust-ci-work

cd rust-ci-work
cargo fmt --all -- --check
cargo check --workspace --lib --bins
cargo test -p aevum-core
cargo clippy --workspace --lib --bins -- -D warnings
