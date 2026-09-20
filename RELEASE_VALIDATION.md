# Aevum AI 0.2.4 Release Validation


## 0.2.4 vocabulary validation

- Final manifest labels: **5,480**: PASS.
- Distinct prototypes: **5,474**: PASS.
- Lazy read-only pack references: PASS.
- Canonical neural-event equality at frequency ranks 1/1000/2000/3000/4000/5000: PASS.
- Canonical equality for modern additions including HELLO/CAR/CAT/CHAT/USER/AI/GRANITE/NEURAL/GGUF/VULKAN/PTFE/MANUAL/DON'T/I'M: PASS.
- All-known sentence produces zero new simulations: PASS.
- Unknown token is simulated once and then reused: PASS.
- Search/open on pretrained-token memories: PASS.
- Fresh-process startup: ~1.06 s / ~101.7 MB peak RSS on packaging host.

Coverage sanity check on two prior real chat exports (not training text): the final pack covered ~**88.6% of alphabetic token occurrences**. Remaining frequent misses were dominated by proper nouns, place names, historical terms and other domain-specific vocabulary that runtime learning is intended to absorb.

## Setup / first run

- Original Aevum Vulkan-wheel-first installer retained: **PASS**
- Short Windows fallback build path (`C:\AEVUM_BUILD`) retained: **PASS**
- `llama-cpp-python` removed from generic `requirements.txt` to prevent backend replacement: **PASS**
- Tkinter/runtime asset/config setup probe added: **PASS**
- Clean first launch with no `data/`, no config, and no model configured: **PASS**
- Clean launch automatically creates config/data/model/log/neural-memory directories: **PASS**
- Granite-only architecture: optional annotator no longer blocks Send: **PASS**
- OpenAI provider no longer requires a local GGUF merely to Send: **PASS**
- Setup/self-test is sharded to avoid one long neural validation command: **PASS**

The Windows `.bat` files are preserved from the prior working Aevum setup pattern and updated conservatively. The packaging container is Linux, so the batch files cannot be executed natively here; their command/path/control-flow contract was statically reviewed. The actual Python first-run path was exercised under a clean virtual display.

## UI / neural minimap

- Aevum 0.1.42 visual shell and branding retained: **PASS**
- Neural minimap embedded in existing top-right header: **PASS**
- 128 lights use actual 8×16 neuron layout: **PASS**
- Glow strength comes from real firing frames + recurrent route traffic: **PASS**
- No synthetic activity generator in minimap: **PASS**
- Remember/automatic hint/search/open operations emit real neural blobs: **PASS**
- Bulk text/Markdown import restored to Memory page and feeds real REMEMBER traces to minimap: **PASS**

## Runtime / memory

- Decoder timing scales: `0.5 / 1 / 2 / 4 ms`: **PASS**
- Recent chat remains direct fast context: **PASS**
- Automatic neural preload is metadata-only: **PASS**
- Explicit AI neural search reconstructs complete logical documents: **PASS**
- Compact web/memory receipts exclude full fetched/opened evidence: **PASS**
- Raw user/assistant messages are neurally archived in full: **PASS**
- Optional Qwen semantic annotator is asynchronous and non-blocking: **PASS**

## Environment limitations

No Granite/Qwen GGUF files are present in the packaging container, so live model generation was not claimed as executed here. Outbound network availability in the packaging environment is also not representative of the target Windows machine.

## 0.2.3 focused validation

- Exact prior red-fish/cold-water neural hints: PASS.
- Complete decoded red-fish/cold-water document recall: PASS.
- Runtime supplies five tools even when legacy `native_tools` metadata is false: PASS.
- Full private neural document reaches the second Granite-style tool pass in the contract harness: PASS.
- Lossless compiled V3 event-stream equivalence: PASS.
- V2 state load + lazy prototype promotion + V3 save/reload: PASS.
- Bounded recent-context path: PASS.

No model self-test was added. Actual Granite/Q6 behavior is validated by normal user operation with the selected GGUF.

## 0.2.4 10k vocabulary validation

- Active frequency words: 10,000; supplement words: 224; punctuation/digits: 28; total active labels: 10,252.
- Unique exact neural prototypes: 10,241.
- AVNP SHA-256 matches manifest: PASS.
- Quarantine scan across merged candidates: 5 deferred words, all absent from the active manifest.
- Common eight-token ingest from the shipped bank: zero neural simulations, exact decoded reconstruction, PASS.
- Final-pack runtime initialization on packaging host: ~0.07 s for lazy manifest/reference registration; prototype bodies remain disk-backed.
- No segmentation/runtime workaround was introduced for the quarantined words.
