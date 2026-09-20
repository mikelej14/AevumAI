# Current State — Aevum AI 0.2.3

## Promoted product baseline

Neural storage core: **0.9.55 Compact Bulk Neural Memory**.

Product/runtime layer: **Aevum AI 0.2.3 Neural Memory Fork**.

The MicroBrain 0.9.55–0.9.57 development branch supplied the promoted neural core/runtime behavior. Aevum AI 0.2.3 is the branded product fork built around that subsystem.

## Active architecture

- Executive: one local architecture-agnostic GGUF slot, intended for Granite Q6.
- Persistent storage: 128-neuron compact MBE2 neural memory.
- Decoder: hidden codec, 0.5 / 1 / 2 / 4 ms multiscale timing.
- Recent memory: direct transcript window.
- Automatic long-term recall: metadata-only neural hints.
- Explicit long-term recall: complete decoded documents.
- Web: deterministic local search/fetch/weather tools callable by Granite.
- Semantic enrichment: optional asynchronous Qwen3.5 ~2B sidecar.
- Cloud: optional OpenAI provider remains available but is not required.

## What was deliberately removed from the critical path

The former dual small-worker choreography is not used for transcript parsing, memory relevance, or response gating. Deterministic runtime code can capture full chat messages accurately without asking an LLM to summarize them first.

The remaining Qwen slot is a sensor only. Its failure cannot prevent storage, recall, web use, or Granite chat.

## Memory rules

Normal user/assistant messages are encoded in full.

Automatic hint preload never contains decoded historical body text.

Explicit neural search always returns complete logical documents, not individual matching chunks.

Tool evidence is not recursively archived. Search receipts remember only what was searched, a brief purpose, and a quick result list. Individual opens/fetches do not create duplicate full-content memories.

## Verification status

The final package is expected to pass:

```text
Python compileall                                      PASS
Fast context/tool/semantic/chat/graph contracts       PASS
Real neural store -> search -> open -> reload          PASS
Multi-chunk explicit full-document recall              PASS
GUI Memory -> Chat -> Settings smoke                    PASS
Fresh-unzip scale assertion 0.5/1/2/4 ms               PASS
```

The exact final counts/hash are recorded in the release response after fresh-ZIP verification.
