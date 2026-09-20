# Local GGUF Runtime

## Executive loading

`core/model_worker.py` is inherited from the mature Aevum/CognitiveBrain local runtime and retained as the GGUF process boundary. The Executive slot is intentionally architecture-agnostic: the selected GGUF must be loadable/chat-capable in the installed `llama-cpp-python`, but Aevum does not maintain a Granite-only architecture whitelist.

The intended target is the user's Granite Q6 GGUF. Settings expose model path, context size, and GPU-layer offload. `n_gpu_layers=-1` requests maximal available offload; actual behavior depends on the local llama-cpp build/backend/hardware.

## Native tools

When the selected GGUF's chat template exposes native tools, Granite receives neural-memory and browser tool schemas directly. The runtime also retains Granite-style XML tool-call parsing as a compatibility path.

The tools are:

```text
neural_memory_search
neural_memory_open
web_search
web_fetch
current_weather
```

The model sees full private tool evidence. The user-facing transcript and persistent tool receipt see only compact telemetry.

Aevum always supplies its OpenAI-style memory/web schemas to the local Executive. Granite 4.2 is the target and natively supports this tool format through its chat template; the runtime no longer withholds tools based on template substring heuristics.

## Optional annotator

The Qwen sidecar is loaded into a separate CPU worker only when enabled. It uses JSON-constrained output and thinking disabled. Its entire contract is one message -> one annotation object.

No Qwen completion is awaited by the Executive path.
