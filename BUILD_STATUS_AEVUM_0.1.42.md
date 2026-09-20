# Aevum AI 0.1.42 — Reference-First RMEM Handoff

## Purpose
Prevent long-term RMEM transcripts from being injected into the Executive as if they were current conversation.

## Executive memory contract
- Current chat history remains normal conversation history.
- Long-term RMEM retrieval is surfaced to the Executive only as validated `record_id`, `record_type`, timestamp, and a short deterministic descriptor.
- Descriptors are derived from the stored record itself and are navigation hints, not generated summaries or current-user messages.
- Exact historical content is available only after `memory_get(record_id)`.
- `memory_get` is authorized only for IDs actually surfaced to the Executive in the current turn or returned by a current-turn `memory_search`.
- `memory_search` itself returns references/descriptors only; it no longer dumps transcript bodies.
- Silent Qwen memory integration and post-turn consolidation may still inspect full RMEM internally; that content does not cross the Executive prompt boundary unless explicitly fetched.

## Verification
- Deterministic descriptor/ID provenance regression.
- Native memory search -> reference -> exact memory_get regression.
- Unauthorized unsurfaced memory_get regression.
- Full automated suite: 103 tests passing.
