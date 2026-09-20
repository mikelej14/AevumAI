# Aevum AI Changelog

## Unreleased — Guided setup and platform installers

- Added a Quick start page that opens when the local Executive model is missing, with model download links, shared Settings selections, and a Save & load models action.
- Added model-file validation, visible loading/ready/error states, and protection against overlapping model loads.
- Added macOS Metal/CPU and Linux CPU setup and launch scripts; renamed Windows installers to identify the platform.
- Reworked the README around the product, intended Granite/Qwen models, hardware planning, and platform setup; added QUICKSTART.md.
- Added installer flow and Tk UI regression checks. Full macOS/Linux GUI and model-runtime validation is pending.

## 0.2.4 — Pretrained Neural Vocabulary

- Expanded the shipped read-only neural vocabulary to **10,000 frequency-ranked English words**, **224 additional Aevum conversational/technical supplement words**, and **28 punctuation/digit tokens**: 10,252 active labels total.
- The active pack contains **10,241 distinct exact MBE2 neural prototypes** in one lazy disk-backed AVNP pack.
- Added an exact optimized offline encoder whose neural-event output is regression-checked against the canonical Brain.
- Added resumable/sharded vocabulary build + merge tools and frequency-ranked backfill.
- Added explicit quarantine handling for single-token traces containing >=32 ms internal silence. Those words do not block the main build and are not inserted into the active shipped pack.
- Current deferred exceptions: `ENGAGE`, `ENGAGED`, `ENGAGEMENT`, `ENGAGING`, and `ENCAPSULATED`; all exhibit the same 39 ms internal silence at the same early trace position. No global ingest/segmentation behavior was changed to accommodate them.
- Known-token first-run ingest reuses exact shipped neural prototypes with zero neural simulation.
- Unknown words still undergo one genuine 128-neuron simulation and are then cached in the user's writable vocabulary.
- Pretrained blobs remain lazy/disk-backed and are not copied into each neural-memory project.
- No changes to Granite memory autonomy, V3 episodic semantics, or 0.5/1/2/4 ms decoder timing scales.

# AevumAI 0.2.3

- Neural-memory hints are advisory only. Current wording/query drives the automatic hint search, but never forces Granite to open or search memory.
- Removed the explicit remember/recall wording detector from memory decision logic.
- Granite receives relevant neural IDs/provenance hints and independently decides whether to call `neural_memory_open`, `neural_memory_search`, or ignore them.
- Hint presence may grant the Executive normal reasoning budget, but does not mandate any tool call.
- No synthetic tool self-test or status badge was added.

# Changelog

## 0.2.3 — Granite Recall + Compiled Neural Memory

- Removed the runtime gate that withheld tools when a template-string heuristic failed. Local Granite now always receives Aevum's neural-memory/web schemas.
- Preserved Granite 4.2's documented OpenAI-style tool loop and existing XML/structured call parsing. No synthetic model self-test or UI tool badge was added.
- Recall/hint turns use normal Granite reasoning; routine chat may continue using low-effort mode.
- Restored bounded recent-context handling so long chats do not crowd tools/recalled evidence out of an 8K context.
- Promoted the already-validated MicroBrain 0.9.56 lossless prototype/layout V3 storage path into Aevum.
- Repeated learned tokens reuse exact deterministic neural prototypes; genuinely new tokens are still simulated by the 128-neuron system once.
- Added lazy V2→V3 exact-prototype promotion for existing 0.2.0/0.2.1 neural memories.
- V2 memory/search/open compatibility is preserved; new memories are V3.
- Removed duplicate neural-file persistence work during each store.
- Default bulk chunk size raised from 12 to 128 tokens for compiled ingest; strict neural physics/decoder behavior is unchanged.


## 0.2.1 — Neural HUD Smoothing / UI Polish

- Kept the 0.2.0 neural memory, decoder, GGUF runtime and persistence formats unchanged.
- Reworked the top-right neural minimap so large engrams use bounded real firing-frame strength instead of scanning every recurrent transmission.
- Small engrams still blend real recurrent-route density into the glow for fine-grained relative strength.
- Added Smooth / Balanced / Detailed Neural HUD quality profiles; Smooth is the new default.
- Capped the HUD at 60 sampled frames and protected old configs from restoring the previous 90-frame/fast-redraw path.
- Coalesced bursty remember/recall animations so the HUD shows current activity instead of replaying a backlog.
- Reused canvas items, precomputed glow color tables, quantized glow levels and skipped unchanged node paints.
- Added an LRU cache for recently visualized neural engrams.
- Removed redundant per-animation writes to the Activity text widget; meaningful memory/tool receipts remain.
- Clean first-run package no longer ships a generated `data/config.json` or chat file.
- Windows Vulkan/CPU setup hotfix from 0.2.0 remains preserved.

## 0.2.0 — Neural Memory Fork

- Hotfix: repaired Windows `SETUP.bat` and `SETUP_CPU_ONLY.bat` Python/Tkinter verification; the previous inline version check used a `>` comparison that `cmd.exe` parsed as redirection.
- Re-based the neural-memory runtime onto the AevumAI 0.1.42 product UI/branding.
- Restored Aevum splash, assets, chat rail, cards, pages and model controls as the primary interface.
- Added compact top-right 128-neuron neural minimap driven by real MBE2 firing/route evidence.
- Relative glow strength reflects recurrent traffic plus recorded firing, with display-only visual decay.
- Preserved local GGUF Executive loading; target deployment is Granite 4.2 Q6.
- Reduced model architecture to required Executive + optional Qwen3.5 ~2B semantic annotator.
- Replaced active RMEM path with compact continuous neural memory.
- Retained recent chat as fast local context and metadata-only automatic neural memory hints.
- Explicit neural recall returns complete reconstructed documents, not matching chunks.
- Retained normal web search/fetch/weather runtime tools with compact persistent receipts.
- Made the Memory page metadata-only on passive refresh to avoid decoding a lifetime store merely by opening the page.
- Moved superseded Aevum 0.1.42 RMEM/two-worker core/tests/docs into `history/` rather than leaving them in the active import tree.
