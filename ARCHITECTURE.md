# Aevum AI 0.2.4 Architecture


## Shipped neural vocabulary pack

0.2.4 adds a read-only `pretrained_vocabulary/` layer beneath the writable personal prototype bank. Startup reads only the JSON index and creates lightweight pack references:

```text
TOKEN -> prototype SHA-256 -> (pack path, byte offset, byte length)
```

The MBE2 prototype itself is read lazily only when compilation, recall, visualization, or validation needs it. Shipped prototypes are decoder/encoder codec material, not episodic memories and not chat text. User experiences continue to be represented by V3 neural layouts/postings plus provenance.

Resolution order for deterministic token ingest is:

```text
personal learned prototype
        ↓ if absent
shipped pretrained prototype
        ↓ if absent
existing exact V2 region promotion
        ↓ if absent
real 128-neuron simulation once
```

The first two are physically the same kind of exact neural prototype; the distinction is only read-only product data versus writable lived vocabulary.

## Runtime flow

```text
User message
   │
   ├── recent visible chat window ─────────────────────────────┐
   │                                                           │
   ├── silent neural cue search -> metadata-only memory hints  │
   │                                                           ▼
   └────────────────────────────────────────────────────── Granite GGUF
                                                               │
                         ┌─────────────────────────────────────┼────────────────────┐
                         │                                     │                    │
                  neural search/open                      web search/fetch       answer
                         │                                     │                    │
                  whole decoded document                 private evidence          │
                         └─────────────────────────────────────┴────────────────────┘
                                                               │
                                         visible assistant reply committed
                                                               │
                                   user + assistant messages neurally encoded
                                                               │
                                      compact MBE2 engrams on disk
                                                               │
                         optional async Qwen 2B semantic annotation/state update
```

The neural minimap subscribes to **actual engram bytes** emitted by remember/recall operations. It has no independent activity generator.

## Memory layers

### Working context
A bounded recent message window is passed directly to Granite. This avoids wasting neural recall/tool calls on the current conversation.

### Neural hint preload
`NeuralMemoryRuntime.hint_search()` matches current cues against the neural store and returns metadata only. The system prompt may expose IDs, timestamps, speakers, roles, matched cue tokens and bounded semantic tags, but not decoded old message bodies.

### Full recall
`neural_memory_search` and `neural_memory_open` decode complete logical documents. A logical document may span multiple MBE2 engrams; document reconstruction reassembles all of them before the private tool result reaches Granite.

### Provenance
Timestamp, speaker, role, chat ID, message ID, source name and semantic annotations are conventional metadata around the neural evidence. They make time/speaker filtering possible without pretending that date arithmetic itself must be encoded in spike geometry.

## Compact engrams

The MBE2 memory format uses compact frame masks and packed recurrent-route streams rather than keeping every route event as a Python tuple. Historical engrams remain disk-backed and are expanded only for decoding/inspection.

The active decoder timing scales remain:

`S = {0.5, 1.0, 2.0, 4.0} ms`

Do not collapse the active decoder to a single 1 ms representation. Native simulation timestamps may use a 1 ms clock while the decoder compares multiple temporal resolutions.

## Neural minimap strength

For each bounded sampled neural frame:

- `F_i(t)` is 1 when neuron `i` is recorded active in the stored firing frame, otherwise 0.
- `C_i(t)` is the number of recurrent route endpoints involving neuron `i` assigned to that sampled window.
- `C_max(t) = max_i C_i(t)`.

The displayed base activity is:

```text
R_i(t) = 0.90 * log(1 + C_i(t)) / log(1 + C_max(t))
A_i(t) = max(R_i(t), 0.42) if F_i(t)=1
         R_i(t)             otherwise
```

`A_i` controls the glow interpolation. Display persistence uses a short decaying visual heat value solely to make rapid firing legible; persistence is not written back to the Brain and is not part of retrieval scoring.

## Models

### Executive
One local GGUF through llama-cpp-python. Intended deployment is Granite 4.2 Q6, but the slot is not hard-coded to Granite architecture.

### Semantic annotator
Optional Qwen3.5-class small GGUF, CPU-oriented and non-thinking. It may produce compact bounded semantic annotations. If absent, disabled, or failed, Granite/chat/neural memory continue normally.

## Web evidence

Web tools remain an Executive capability. Fetched page/search contents are private evidence during the turn. Visible/persistent tool receipts are deliberately compact and exclude fetched page bodies. Normal assistant answers produced from that evidence are stored fully as lived chat.

## UI lineage

Aevum 0.1.42 is the UI/product lineage. 0.2.3 intentionally keeps that shell rather than replacing it with MicroBrain's research GUI. The only persistent high-visibility addition is the top-right neural minimap.

## Granite tool transport and V3 neural ingest (0.2.3)

The Executive always receives Aevum's OpenAI-style function schemas. Tool availability is no longer inferred from brittle string matching against the embedded Jinja template. Granite may emit llama-cpp structured `tool_calls` or its documented XML `<tool_call>` form; both feed the same assistant-tool/tool-response transcript loop.

Persistent neural memory now accepts both formats:

- `compact_continuous_v2` — existing 0.2.0/0.2.1 MBE2 engrams;
- `prototype_continuous_v3` — shared exact MBE2 neural-region prototypes plus MBL1 timing layouts and packed prototype postings.

V3 is lossless for the current fixed deterministic/non-plastic encoder. Reuse is a neural-event cache, not a text cache: episodic layouts store prototype hashes/timing, while English remains solely in the decoder codec. If neural plasticity/state changes are introduced later, compiled reuse must be disabled until equivalence is revalidated.
