# MicroBrain 0.9.57 — Granite Neural Context

0.9.57 keeps the **0.9.55 Compact Bulk Neural Memory** core frozen and turns the chat/runtime around it into the intended local product architecture: one local GGUF executive (target: Granite Q6), deterministic neural-memory/runtime services, and one **optional** small Qwen semantic sidecar.

The old two-worker cognitive pipeline is gone from the critical path. The runtime itself owns transcript capture, recent context, automatic memory hints, neural recall, web tools, crash recovery, and persistence. The optional Qwen model annotates meaning/state after raw memory is already safe; it never has to finish before Granite can answer.

## Runtime architecture

```text
                         ┌──────────────────────────────┐
                         │       Granite Q6 GGUF        │
                         │ conversation / reasoning /   │
                         │ native tool use / final text │
                         └──────────────┬───────────────┘
                                        │
             recent chat + hint metadata│       tool calls
                                        ▼
┌────────────────────────────────────────────────────────────────────┐
│                    Deterministic Python Runtime                    │
│ recent-chat window • neural hint preload • tool loop • browser    │
│ message capture • timestamps • speakers • crash reconciliation    │
└───────────────┬──────────────────────────────┬─────────────────────┘
                │                              │
                │ full chat messages           │ private evidence
                ▼                              ▼
┌─────────────────────────────┐       ┌─────────────────────────────┐
│ 128-neuron MicroBrain       │       │ Neural recall / Web        │
│ compact MBE2 lived memory   │       │ full docs/pages to Granite │
└──────────────┬──────────────┘       └─────────────────────────────┘
               │
               │ after raw storage (optional, asynchronous)
               ▼
┌─────────────────────────────┐
│ Qwen3.5 ~2B semantic sensor │
│ tone • salience • context   │
│ positive/negative • state   │
└─────────────────────────────┘
```

## The three levels of memory context

**1. Recent chat — fast local working memory.** The configured recent turns are passed directly to Granite on every request. This is ordinary short-term conversational continuity and does not require neural search.

**2. Automatic neural hints — silent metadata only.** Before Granite answers, the current user message is tokenized into decoder cues. Matching older neural memories may surface automatically as tiny hints containing document ID, timestamp, speaker/role, matched cue words, and (when available) short semantic tags from the optional Qwen sidecar. **No decoded body text is preloaded.** Granite is explicitly instructed not to infer remembered facts from a hint.

**3. Explicit neural recall — complete documents.** If an older hint looks relevant, or Granite independently decides prior context is required, it calls `neural_memory_search` or `neural_memory_open`. Explicit search returns the **entire decoded document/message**, even when that document spans multiple compact MBE2 chunks. The model never receives a misleading isolated matching chunk as if it were the memory.

This gives normal chat the speed of a recent-context window without throwing a lifetime transcript into every prompt.

## What is stored

Normal user and assistant messages are neurally encoded in full after the assistant finishes the turn. Provenance keeps the speaker name, role, UTC timestamp, chat ID, and message ID. The question is deliberately archived **after** recall/tool use so it cannot retrieve itself as historical memory.

Tool evidence follows a different rule. A neural recall document or fetched webpage may be large and is private working evidence for the current answer. That evidence is **not copied into chat history or re-encoded as a second memory**. Search actions instead create a compact neural receipt such as:

```text
WEB SEARCH FOR: current Granite release
PURPOSE: verify a current model detail
RESULTS: title — URL; title — URL; ...
```

or:

```text
NEURAL MEMORY SEARCH FOR: car
PURPOSE: recover what was discussed last week
RESULTS: doc 421 2026-09-12T18:04:11+00:00 Mike; doc 438 ...
```

Opening an individual result does **not** create another memory receipt. This prevents tool evidence from recursively bloating lived memory while still preserving that a search happened, why it happened, and what it found.

## Optional Qwen semantic sidecar

The old mandatory Observer + Memory worker arrangement is removed. There is now one optional **Semantic Annotator** slot intended for a small Qwen3.5 model around 2B parameters, CPU-only by default.

It runs asynchronously only after the raw message has already been encoded. It can attach bounded metadata such as sentiment, positivity, negativity, hostility, praise, urgency, salience, correction, uncertainty, novelty, intent, context, topics, and entities. User-message observations also feed a deterministic persistent functional state with inertia/decay for valence, arousal, frustration, warmth, confidence, and curiosity.

The Qwen result never replaces the original memory. If Qwen is disabled, unloaded, slow, or crashes, **chat, web tools, neural storage, and recall continue normally**.

## Hidden decoder codec

There is no user-facing dictionary. The neural pattern→word label map remains internal because the decoder needs it to translate a recalled neural trace back into English. New vocabulary is learned automatically from the same continuous ingest experience. It is a codec, not a separate content database.

The active temporal decoder remains:

```text
0.5 ms / 1.0 ms / 2.0 ms / 4.0 ms
```

No part of 0.9.57 returns to the accidental single-1-ms representation.

## Web access

The local Granite runtime exposes normal `web_search`, `web_fetch`, and `current_weather` tools through the retained Aevum browser layer. Search/fetch evidence is provided privately to Granite. Only compact search receipts are archived as described above.

A compatible GGUF/chat template with native tool support is strongly recommended so Granite can autonomously call neural-memory and web tools. If a model does not expose native tools, recent chat and automatic neural hints still work, but autonomous explicit recall/web calls are unavailable for that model.

## GUI

The product GUI remains **memory-first**. Memory opens by default, Chat is secondary, and Settings is small. The right-side 128-neuron field remains visible and is driven by the **actual compact neural engram** during storage and recall: active frame neuron IDs and sampled recurrent source→destination routes illuminate the matching graphical neurons and edges. The afterglow is visual persistence of real activity, not fabricated traffic.

Settings expose only the model/runtime controls that matter:

- local chat provider (default) or optional OpenAI provider;
- Granite/chat GGUF path, context size, GPU-layer offload, auto-load;
- optional Qwen3.5 semantic sidecar path/toggle;
- recent-chat turn count and automatic neural-hint count;
- user/assistant display names and system prompt;
- local browser toggle;
- optional OpenAI API key/model.

API keys are session-only or read from `OPENAI_API_KEY`; they are never written to `config.json`.

## Chat flow

```text
USER MESSAGE
   │
   ├─ visible transcript (recent-turn window)
   │
   ├─ silent automatic neural hint lookup → metadata only
   │
   ▼
GRANITE GGUF
   │
   ├─ optionally neural_memory_search/open → complete neural document
   ├─ optionally web_search/fetch → private page/search evidence
   ▼
FINAL ASSISTANT MESSAGE
   │
   ├─ visible transcript
   ├─ full user message → neural encoder
   ├─ full assistant message → neural encoder
   ├─ compact search receipts → neural encoder
   └─ optional async Qwen annotations → provenance/state metadata
```

On startup, `chats.json` is reconciled against neurally archived message IDs. A message that reached the visible transcript but was missed because of a crash/forced exit is queued for neural encoding automatically.

## Running

Windows:

```text
SETUP.bat
RUN.bat
```

or:

```bash
python -m pip install -r requirements.txt
python app.py
```

In **Settings**, select the Granite Q6 (or other chat-capable GGUF), choose the desired GPU-layer offload, Save, then **Load models**. The Executive loader is not hard-locked to Granite architecture; Granite Q6 is simply the intended configuration.

The Qwen sidecar is optional. Leave it disabled until you want semantic/state enrichment.

## Scope and current limitations

0.9.57 is a local-memory/chat product milestone, not a claim that 128 neurons provide general intelligence. The recurrent network is being used as the persistent temporal storage/recall substrate; Granite remains the conversational reasoning model.

Neural recall is currently strongest for learned lexical cues/exact route variants, with the preserved multiscale temporal decoder as fallback. It is not embedding-vector semantic search. Qwen semantic tags improve context and hint quality without replacing neural evidence.

Large manual/conversation ingestion is storage-efficient because old engrams are disk-backed, but encoding speed is still bounded by actually simulating the 128-neuron recurrent system. The next scaling work should target encoding throughput/prototype-delta reuse without changing the neural memory contract.

## Verification targets for this release

The package includes automated contracts for metadata-only automatic hints, full-document explicit recall, compact tool telemetry, Qwen-state bounds/persistence, transcript behavior, real-neural graph sampling, OpenAI fallback contracts, and the real neural store/search/open/reload path. See `docs/CURRENT_STATE.md` and `CHANGELOG.md` for the final pass matrix shipped with the ZIP.
