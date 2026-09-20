# Experiment / Decision History

## Retained

### High-resolution route timing (0.9.33)
10 ms route bins were too coarse.  Raw recurrent send times at the native 1 ms simulation clock were retained.

### Multi-scale timing (0.9.34)
A single Gaussian width was also insufficient.  Four simultaneous comparison widths—0.5, 1, 2, 4 ms—plus ISI rhythm were introduced.  This is the active timing architecture in this handoff.

### Causal lineage discovery (0.9.37)
`origin` and `born` were found to be useful native route fields that older serializers discarded.  They remain captured in experiences even though the current active decoder does not reduce everything to them.

### 30-word causal-birth development gate (0.9.38)
The causal-birth family decoder achieved 30/30 on a fresh 30-word development cohort.  This is retained as an important historical benchmark.

## Rejected / superseded as active paths

### 10 ms-only route bins
Rejected for decoder identity because meaningful within-bin timing differences disappear.

### Single-width high-resolution timing
Superseded by multi-scale comparison; one width forces a tradeoff between discrimination and jitter tolerance.

### Causal-birth-only as the handoff's 'current decoder'
Useful historical branch, but too reductive for the intended temporal architecture.  The first 0.9.52 handoff accidentally promoted this as `current_best_decoder.py`; corrected here.

### Simultaneous four-input glyph injection (0.9.40–0.9.47 branch)
Useful as a collision/mechanism probe, but it changes the input geometry.  It must not silently replace calibrated `stimulus(ch)` input in the normal language decoder.

### Brittle CAT/CAR fixed-gap or word-specific tie-breaks
Rejected.  A rule that names a collision or moves a threshold after seeing the validation cohort is not evidence of generalization.

### Structural hybrid tie-breaker from the later session
Rejected after it introduced a CHAIR→WHITE error on a fresh cohort.  Do not stack it on the active decoder merely because it repaired a development collision.

## Validation discipline

A development cohort can guide changes.  Once inspected for tuning, it is no longer an untouched validation cohort.  Promotion requires new seeds/experiences that were not used to select decoder rules.

## 0.9.54 continuous-memory decision

### Retained: one continuous episode per new sentence
0.9.53 proved neural Store/Search/Open but persisted an ordered list of word-pattern IDs. 0.9.54 keeps the decoder dictionary but removes the token sequence from newly stored episodic memories.

### Retained: strong quiet-gap boundary baseline
A 96 ms no-input interval is used between text words so the first autonomous-from-trace boundary test is unambiguous. The boundary detector itself uses only observed frame inactivity and a 32 ms split criterion. This is a deliberate baseline, not a claim that natural/no-gap segmentation is solved.

### Retained: exact recurrent-route identity as an accelerator
In deterministic baseline runs, recovered word regions reproduce isolated recurrent route timing exactly after onset normalization. Using this exact neural identity prevents unnecessary O(route-event fuzzy matching). It is not a semantic database lookup; non-identical experiences still require the multiscale decoder.

### Still open
Shorter/no gap segmentation, noisy continuous episodes, large-vocabulary streaming, and long-term interference/plastic-memory tests remain future gates.

## 0.9.55 compact/bulk-memory decision

**Problem:** 0.9.54 proved continuous neural Store/Search/Open but persisted full route/frame structures. That representation was scientifically transparent but operationally unsuitable for large corpora because Python object overhead dominated RAM.

**Promoted changes:**

- delta/varint-coded MBE2 neural engrams;
- immediate disk spooling and lazy ZIP-backed reload;
- sparse frame masks instead of dense Python frame objects;
- recurrent-route digest recall index with no plaintext payload;
- direct decoder learning from the already-generated ingest regions;
- bounded chunked text ingestion with punctuation tokens and GUI progress;
- `.lrnr` V2 separates binary engrams from JSON metadata.

**Rejected as the primary 0.9.55 representation:** simply gzip/zlib-compressing the old JSON record. It lowers disk size but does not solve Python-object inflation during normal runtime and does not provide lazy region access/indexed recall.

**Exploratory but not promoted:** cross-memory prototype/reference compression. In the four-fact corpus, 16 regions reduce to 8 unique exact recurrent patterns and a prototype/reference estimate falls from about 139.6 KB to 84.6 KB. This is promising, but it reintroduces a segmentation-aware compression layer and therefore must be validated as lossless neural reconstruction before replacing MBE2.
