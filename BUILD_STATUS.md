# Aevum AI 0.2.4 Build Status


## 0.2.4 pretrained vocabulary

- Frequency-ranked active words: **10,000**.
- Additional Aevum conversational/technical supplement words: **224**.
- Punctuation/digit tokens: **28**.
- Shipped active labels: **10,252**.
- Distinct exact neural prototypes: **10,241**.
- Neural AVNP pack size: **196,787,281 bytes**.
- Pack SHA-256: `ab07f3ba5be4a8c9b9ea09b5742b59607dea9eeb63b8cb731472634f5e4a6d9c`.
- Deferred/quarantined multi-island words: **5** (`ENGAGE`, `ENGAGED`, `ENGAGEMENT`, `ENGAGING`, `ENCAPSULATED`). Each has the same 39 ms internal silence at 88→128 ms.
- Quarantined candidates are excluded and frequency-backfilled; normal segmentation/runtime behavior is unchanged.
- Pack is read-only/lazy; built-in MBE2 blobs stay on disk until needed.
- Known-token sentence ingest can compile entirely from exact neural prototypes with zero simulations.
- Unseen tokens still run through the real 128-neuron encoder once and then join the writable personal prototype bank.
- Existing V2/V3 user memories remain compatible.
- No decoder timing-scale change; 0.5/1/2/4 ms remains active.

## Scope

Aevum 0.1.42 UI/product shell + MicroBrain 0.9.55–0.9.57 compact neural-memory/runtime branch.

## Preserved

- Aevum branding/assets/splash screen.
- Dark obsidian + neon-green theme.
- Left multi-chat rail and persistent visible transcripts.
- Chat answer, Thinking and Tools presentation.
- Cognitive State, Memory, Activity and Settings navigation.
- GGUF browse/start/stop workflow.
- Granite native/local tool loop and normal web tools.

## Replaced

- Legacy RMEM is not the active long-term memory engine.
- Mandatory observer + memory Qwen workers are not in the active runtime.
- Neural memory owns persistent lived experience.
- Optional single Qwen semantic annotator enriches provenance/state asynchronously.

## New UI instrument

Top-right 128-neuron minimap. It consumes real compact engram frames/routes and renders relative activity strength rather than binary on/off colors or synthetic animation.

## Verification performed on build tree

- Whole-tree Python compilation: PASS.
- Neural minimap real-evidence/relative-strength tests: PASS.
- Local context/hint/tool-memory contract tests: PASS.
- Timestamped neural message store/search/open/reload: PASS.
- Multi-chunk full-document neural recall: PASS.
- Chat store + semantic state tests: PASS.

Live Granite/Qwen generation requires the user's GGUF files and was not performed in the packaging container. Live outbound browsing depends on the runtime machine's network access.
## Windows setup hotfix

- Removed shell-sensitive `>` comparison syntax from inline `python -c` verification in both setup scripts.
- `requirements.txt` intentionally excludes `llama-cpp-python`, so installing support packages cannot replace the selected Vulkan/CPU backend.
- Clean setup probe after patch: PASS.


## 0.2.1 UI smoothing verification

- Neural minimap unit tests, including large-engram no-route-walk guard: PASS (3/3).
- Setup/local-context/OpenAI-contract/semantic-state tests: PASS (8/8).
- Chat store + real neural activity adapter tests: PASS (2/2).
- Timestamped neural message roundtrip: PASS.
- Multi-chunk complete-document recall: PASS.
- Real Tk GUI startup/navigation with Smooth HUD profile: PASS.
- Synthetic 220,000-route HUD preparation: ~0.011 s first pass on packaging host versus ~0.23–0.25 s before the 0.2.1 renderer change.

## 0.2.3 Granite/memory correction

- Granite tool schemas are supplied unconditionally to the local Executive; `model_info.native_tools` is diagnostic only and no longer controls access.
- Exact stored `RED FISH / COLD WATER` V2 memories were independently queried before the runtime change: neural hints and complete decoded documents were correct. The failure was therefore above the decoder/search layer.
- Recent context is bounded before prompt construction.
- Turns with relevant neural hints receive normal Granite reasoning budget; wording alone does not force recall or a tool call.
- New neural stores use lossless V3 prototype/layout coding with exact compiled reuse.
- Existing V2 memories remain readable/searchable and can seed exact V3 prototypes.
- V3 direct-vs-compiled neural event equality tests: PASS (2/2).
- Fast runtime/context/minimap/setup contracts after changes: PASS (12/12).
- Product neural roundtrip tests: PASS (2/2).
- Neural activity adapter test: PASS (1/1).
- Measured old-data migration: first repeated red-fish store ~2.45 s, subsequent identical store ~0.003 s, zero new prototypes simulated.

Live Granite generation still requires the user's actual GGUF and is intentionally left to normal use; no extra model self-test was added.
