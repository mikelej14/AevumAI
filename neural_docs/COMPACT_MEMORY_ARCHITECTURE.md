# Compact Memory Architecture — 0.9.55

## 1. Separation of concerns

0.9.55 treats three things separately:

1. **Neural engram** — compressed evidence produced by the brain.
2. **Neural recall index** — rebuildable route digests and time offsets used to locate likely memories without expanding them.
3. **Decoder dictionary** — the learned mapping from neural patterns/digests to English tokens.

The episodic engram never contains English labels.

## 2. MBE2 coding

For route event `i` with native integer send time `t_i` and recurrent edge `(s_i,d_i)`:

`Δt_i = t_i - t_(i-1)`

`edge_i = 128*s_i + d_i`

Both are variable-length integers (signed time differences use zig-zag coding). Thus a route that formerly required Python tuple/list/dict objects usually needs only a few raw bytes before compression.

Frame activity is a 128-bit vector:

`B_t = Σ 2^n  for active neuron n at t`

Only active frames are retained for continuous memories. Silent millisecond frames are reconstructed from `duration_ms` when a decoder region is expanded.

The two streams are compressed separately because their statistics differ strongly.

## 3. RAM behavior

An engram is spooled to a `.mbe` file as soon as a chunk finishes. The Brain retains only a tiny reference plus its derived route index. A saved project points directly to the engram member inside the `.lrnr` ZIP.

Exact open/search does not need to read the engram body at all. It uses the neural route digest index and decoder route dictionary.

Only a novel/ambiguous pattern triggers region extraction and the full 0.5/1/2/4 ms fuzzy scorer. At that point only one word-sized neural region is expanded.

## 4. Bulk chunking

A large source is tokenized and split into bounded chunks (`DEFAULT_CHUNK_TOKENS = 12`). Each chunk is a continuous Brain run with the same quiet-gap baseline used by 0.9.54. Chunks are storage/compute boundaries; they are not plaintext storage.

The document layer stores chunk IDs in order so a complete document can be decoded and reassembled later.

## 5. Supervised decoder growth without a second simulation

During ingest the source tokens are known to the trainer. After the continuous run, boundaries are discovered from neural inactivity. If discovered region count equals token count, each neural region can supervise the decoder label for that token. This is dictionary learning, not episodic payload storage.

A lightweight digest->label table records exact neural-language variants. Up to three region references per label are retained as fuzzy exemplars; the exemplar refers back into the existing neural memory rather than duplicating its route data.

## 6. Compatibility

- `continuous_v1` episodes from 0.9.54 are migrated to MBE2 scratch blobs when opened.
- 0.9.53 segmented records remain readable.
- `.lrnr` V1 projects are accepted and converted in memory; the next save writes V2 compact format.

## 7. Next compression tier: prototype + delta

The current MBE2 layer is lossless at the decoder-evidence level and deliberately conservative. An exploratory corpus of the four baseline facts contains 16 neural word regions but only 8 unique recurrent route patterns. Storing one compact prototype for each unique region plus neural references would reduce that small corpus from about 139.6 KB to about 84.6 KB before any residual/delta coding.

A future codec can therefore represent a continuous engram as neural prototypes plus residual deltas while still storing no English. This should be validated as a lossless reconstruction layer before replacing MBE2 as the primary format.
