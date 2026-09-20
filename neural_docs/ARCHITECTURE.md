# Architecture — 0.9.55

## 1. Brain

`microbrain.py` remains the 128-neuron recurrent network. Neural physics, weights, delays, calibrated input routing, and the active 0.5/1/2/4 ms decoder are unchanged. 0.9.55 changes how evidence is **recorded, stored, indexed, and loaded**.

During bulk encoding the Brain suppresses the heavyweight historical event log and sends recurrent transmissions to a streaming compact recorder. `last_fired` supplies the active-neuron frame directly, avoiding repeated scans of the growing event list.

## 2. Continuous encoding

`continuous_sentence_experience.run_compact_token_experience()` drives a bounded token chunk through one Brain instance. Token labels are consumed by the trainer but are not returned in the neural engram.

The existing baseline still uses a long quiet interval between decoder tokens. That gives a deliberately strong first boundary cue while storage/recall scaling is developed.

## 3. MBE2 engram

`compact_engram.py` records only decoder-relevant operational evidence:

- sparse active frames;
- recurrent `(source,destination,send_time)` transmissions;
- duration and counts.

The old duplicated causal/debug trace is not persisted in operational memories because the promoted decoder does not consume it.

Frame and route streams are delta/varint coded and compressed independently. Historical route evidence is therefore not retained as Python dictionaries/tuples.

## 4. Disk-backed memory

A finished engram is immediately written to a scratch `.mbe` file. The in-memory episodic record is only a reference plus small metadata.

Saved `.lrnr` V2 projects store each engram as its own ZIP member. Reloaded records point to that member and do not read the blob until required.

## 5. Neural recall index

`Brain.neural_memory_index` is rebuildable derived state:

`memory_id -> [(start_ms,end_ms,route_digest), ...]`

It contains no English labels or source strings. `route_digest` is SHA-256 over the translation-invariant recurrent route stream of that recovered neural region.

`route_postings` is an in-process inverse index from route digest to memory/region positions. This makes exact neural search O(matches) instead of scanning and expanding every historical episode.

## 6. Decoder dictionary

English translation remains separate from episodic memory.

- `decoder_lexicon`: label -> up to three fuzzy exemplar references.
- `decoder_route_labels`: exact neural route digest -> learned label/labels.
- new ingest exemplars can be **region references** into already-stored engrams, so decoder learning does not duplicate route evidence.

During bulk supervised ingest, discovered neural regions are aligned with the source tokens once. This teaches the neural-language decoder; it does not insert labels into the episodic engram.

## 7. Recall/open

For a compact memory, normal recall uses its route digest index. If a digest maps unambiguously to one decoder label, no engram body is expanded.

Only an unknown or ambiguous region loads the relevant `.mbe`, extracts that one word-sized region, and applies the full 0.5/1/2/4 ms multiscale scorer.

## 8. Bulk document layer

Large text is split into bounded token chunks (12 by default). A document record stores only ordered chunk IDs, token count, and optional filename metadata. The document body is not retained.

`Open Full Document` decodes chunk route digests through the neural dictionary and assembles the translated token stream.

## 9. Compatibility

- 0.9.54 `continuous_v1` memories migrate to compact scratch engrams.
- 0.9.53 segmented records remain readable.
- V1 `.lrnr` projects load and migrate; the next save writes compact V2.
