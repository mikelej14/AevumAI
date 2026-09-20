# Aevum AI 0.1.41 — Full Response / Length-Cutoff Repair

Built directly from 0.1.40.

- Preserves llama.cpp finish_reason for streamed and non-streamed Executive calls.
- Detects length/context cutoffs instead of accepting a mid-sentence response as complete.
- Automatically continues a truncated final answer from a compact continuation prompt, streaming live, while respecting the configured per-turn output ceiling.
- Executive max output tokens now apply to reasoning-only recovery and final-answer repair paths instead of hidden 700/1024 token caps.
- Executive thinking_max_chars is no longer silently replaced with 1400/2200 on fast/assisted non-native turns.
- No UI/branding/startup changes.
- Regression suite: 99 tests.
