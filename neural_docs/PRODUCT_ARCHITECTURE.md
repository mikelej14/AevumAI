# Product Architecture — Aevum AI 0.2.3

## Design objective

Aevum AI 0.2.3 is a local conversational application with persistent neural lived memory. It deliberately separates **reasoning**, **storage**, **working context**, and **semantic interpretation** so no single optional component can silently become the memory database.

## Components

### 1. Executive — local GGUF

The only required model is the Executive. The target setup is Granite Q6 loaded through `llama-cpp-python`, with GPU offload configured in Settings. The loader is architecture-agnostic for the Executive and prefers the model's embedded chat template when available.

Executive responsibilities: normal conversation, reasoning, choosing when a historical memory or web source is needed, issuing native tool calls, and writing the final answer.

The Executive does **not** own long-term persistence.

### 2. Deterministic runtime

Python owns everything that does not require generative interpretation: recent-turn selection, timestamps, speaker/role provenance, automatic neural cue lookup, full-document memory opening, browser execution, compact tool receipts, post-turn message archival, crash reconciliation, model lifecycle, and UI events.

This replaces the old design where multiple small LLM workers sat in the critical path before an Executive response.

### 3. 128-neuron memory substrate

The 0.9.55 compact neural core is preserved. Continuous messages/documents are encoded into the recurrent 128-neuron system and persisted as compact MBE2 engrams. Historical engrams remain disk-backed; only small indexes/provenance stay hot.

The decoder's active comparison scales are 0.5, 1.0, 2.0 and 4.0 ms. Internal lexical labels are a decoder codec and are not exposed as user memory.

### 4. Optional semantic Qwen

One optional Qwen3.5 ~2B GGUF is a narrow asynchronous sensor. It receives one completed message at a time and emits a bounded JSON annotation. It never decides the assistant's answer, never decides whether raw memory is stored, never blocks Granite, and never becomes the authoritative content record.

Annotations are attached to provenance and can help automatic hints expose useful topic/context metadata. User annotations also feed the deterministic `SemanticState` accumulator.

## Context ladder

The Executive receives context in a deliberate hierarchy.

### Fast recent context

The most recent configured chat turns are supplied verbatim from the visible transcript. This is the fast RAM-like conversational window.

### Automatic persistent hints

Current user text is used to perform an exact/learned neural cue lookup. Returned hints contain metadata only:

```text
document_id
timestamp_utc
speaker
role
matched neural cue labels
short Qwen context/topics (if available)
```

They never contain decoded memory body text. The system prompt explicitly prohibits inventing memory content from the hint.

### Explicit persistent recall

When Granite needs actual historical content it calls a memory tool. `neural_memory_search` resolves candidate neural documents and **opens every returned document completely**. If one logical document spans several MBE2 chunks, all chunks are decoded and reassembled before the tool result reaches Granite.

`neural_memory_open(document_id)` is primarily useful for a document surfaced by an automatic hint.

## Tool evidence versus tool receipt

Full tool evidence is ephemeral private context for the model. Persistent tool history is intentionally small.

For neural search, the model receives complete decoded documents. The transcript/tool card and stored receipt receive only query/purpose/result IDs plus provenance.

For web search, the model receives the search result data; the stored receipt contains only query/purpose and a short title/URL list. `web_fetch` page text is private evidence and is not independently archived.

Normal user and assistant messages are different: their full text is always encoded as lived neural memory.

## Turn ordering

```text
1. User text is written to visible transcript.
2. Recent transcript + automatic neural hints are assembled.
3. Executive answers, optionally calling neural/web tools.
4. Final assistant answer is written to visible transcript.
5. Full user message is queued for neural encoding.
6. Full assistant message is queued for neural encoding.
7. Compact search receipts are queued for neural encoding.
8. Optional Qwen annotation runs after each raw message has been stored.
```

Archiving at step 5 rather than step 1 prevents the current question from being returned by historical recall during its own answer.

## Persistence

`data/chats.json` is a visible-session store and fast-recent-context source. It is not queried by neural memory tools. Deleting a chat does not erase already-lived neural memory.

`data/neural_memory/` contains compact neural persistence and indexes. `data/semantic_state.json` contains bounded functional state derived from optional Qwen observations.

On startup the runtime compares transcript message IDs to neural document provenance and queues any missing messages, providing crash-safe eventual archival.
