# Aevum AI

**A local desktop AI assistant with persistent neural memory.**

Aevum AI brings conversation, long-term memory, and web tools into a desktop app. It is built for ongoing conversations: recent messages provide immediate context, while a persistent memory system lets the assistant retrieve older conversations and imported material when they are relevant.

The intended model pairing is **IBM Granite 4.2 3B** for conversation and reasoning, with **Qwen3.5-2B** as an optional semantic annotator. Aevum's MicroBrain memory engine handles storage and recall independently of those language models.

## Features

- **Local AI conversation.** Run GGUF models on your own computer, with streaming responses, multiple chats, and model start/stop controls.
- **Persistent neural memory.** Store conversations and imported text as compact neural engrams, retaining speaker, time, and source information.
- **Recall when it matters.** Relevant memory hints help the assistant decide when to retrieve complete stored documents.
- **Text and Markdown import.** Add material through the Memory page, then search and recall it alongside conversations.
- **Web search, page reading, and weather.** Let the assistant look up current information and use it in its answers.
- **Optional semantic annotation.** Add topic, entity, sentiment, urgency, salience, and correction signals to stored messages.
- **Visible neural activity.** A 128-neuron minimap displays activity derived from memory events, with Smooth, Balanced, and Detailed display profiles.
- **Desktop controls.** An obsidian and neon-green interface brings together Chat, Memory, Cognitive State, Activity, and Settings.
- **Customizable identity.** Configure assistant and user names, system instructions, personality, generation settings, and model paths.

## Models and downloads

Language-model weights are downloaded separately. Aevum's local runtime loads **GGUF files** through `llama-cpp-python`; select the downloaded files in Settings.

### Executive: IBM Granite 4.2 3B

**Granite 4.2 3B is the intended main model**, with the **Q6_K** quantization as the project's intended configuration. It handles conversation, reasoning, tool selection, web use, and decisions about when to retrieve memories.

- [IBM Granite 4.2 3B — official Hugging Face model card](https://huggingface.co/ibm-granite/granite-4.2-3b)
- [IBM Granite 4.2 3B — official GGUF downloads](https://huggingface.co/ibm-granite/granite-4.2-3b-GGUF/tree/main)
- Intended file: `granite-4.2-3b-Q6_K.gguf`

The Executive loader also accepts other compatible chat-capable GGUF models. Their behavior and tool support depend on the model, its embedded chat template, and runtime compatibility.

### Semantic Annotator: Qwen3.5-2B

**Qwen3.5-2B is the intended optional supporting model.** It analyzes stored messages for bounded semantic metadata such as topics, entities, sentiment, importance, urgency, and corrections. That metadata supports memory hints and the assistant's functional state.

- [Qwen3.5-2B — official Hugging Face model card](https://huggingface.co/Qwen/Qwen3.5-2B)
- [Qwen3.5-2B — community GGUF downloads by Unsloth](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF)

Select a compatible Qwen3.5-2B GGUF in **Semantic Annotator** and enable it to use this feature. The default configuration is CPU-oriented with thinking disabled for structured annotation. The current application has one optional annotator slot; ordinary conversation and neural memory continue to work without it.

## How the system works

Aevum separates the language model's conversation context from its persistent memory.

1. **Read the current conversation.** A bounded set of recent messages gives the Executive immediate context.
2. **Find relevant memory hints.** The current message is used to search the memory index. Candidate memories appear as metadata hints containing source information and matching cues.
3. **Choose whether to recall or browse.** The Executive can open a hinted memory, search stored material, use web tools, or answer from the context it already has.
4. **Retrieve complete documents.** Explicit memory retrieval reconstructs the full stored document, including documents split across multiple engrams.
5. **Store the exchange.** Completed user and assistant messages are archived into neural memory. If enabled, Qwen adds semantic annotations after storage.

The memory subsystem uses MicroBrain's deterministic **128-neuron encoder** and compact prototype/layout representation. Repeated tokens reuse previously encoded neural prototypes, while new tokens are encoded and added to the user's personal bank. This learning extends the memory store; it does not fine-tune the Granite or Qwen model weights.

### Pretrained neural vocabulary

Aevum includes a starter bank of **10,252 active token labels** backed by **10,241 distinct neural prototypes**, covering common English words, conversational and technical terms, punctuation, and digits.

The approximately **196.8 MB** vocabulary pack is loaded on demand and shared as a read-only bank. Familiar tokens can reuse its prototypes immediately; unfamiliar words extend the writable personal bank. The starter vocabulary accelerates memory encoding and is separate from the language models downloaded above.

### Memory you can inspect

The Memory page opens with metadata rather than decoding the entire collection into RAM. Search and document retrieval decode stored material on demand.

Visible chat history and neural memory have separate lifecycles: **deleting a chat removes its visible transcript/session but does not erase its stored neural memories.** At startup, Aevum also checks for transcript messages that were not archived before an interrupted session and attempts to recover them into memory.

## System requirements and memory planning

**Plan for roughly 6 GB of GPU memory just to run the intended Granite 4.2 3B Q6_K configuration. An 8 GB or larger GPU is recommended for headroom.** A GPU with 6 GB total VRAM may have too little free memory once the desktop and other applications are using it.

The Granite Q6_K file is approximately 3 GB on disk, but **download size is not runtime memory usage**. Model weights, the context/KV cache, compute buffers, and backend overhead all contribute. Longer context settings can increase memory use. The 6 GB figure is a planning estimate for the intended setup, not a fixed allocation or a guarantee for every device.

- **System RAM:** 16 GB is a practical starting recommendation; 32 GB is preferable when running both models, using CPU inference, keeping large histories, or multitasking. These are planning recommendations rather than benchmarked minimums.
- **Windows GPU inference:** use a Vulkan-capable GPU with current drivers. Budget approximately 6 GB of free VRAM for Granite, with 8 GB or more total VRAM recommended. The default optional Qwen annotator runs on the CPU and needs additional system RAM.
- **Apple Silicon:** Metal uses unified memory shared by the GPU, CPU, and macOS. The Granite memory budget comes out of that shared pool. Prefer at least 16 GB unified memory; 24–32 GB provides more room for Qwen and other applications. An 8 GB Mac is not a recommended target for the intended two-model setup.
- **CPU-only use:** a dedicated GPU is optional. Model weights and runtime buffers use system RAM instead, and response speed depends heavily on the CPU and memory bandwidth. The Linux installer and Intel Mac setup use this mode by default.
- **Disk space:** allow around 20 GB free as a starting budget for model downloads, the Python environment, the vocabulary pack, and working space. Additional quantizations, source builds, and growing personal memory need more storage.
- **Runtime:** 64-bit Python 3.10+ with Tkinter; Python 3.11 or 3.12 is recommended by the setup workflow. Linux requires a graphical desktop session. Internet access is needed for installation, model downloads, and web tools.

If a model fails to load because memory is exhausted, close other GPU-heavy applications, reduce the configured context size or GPU offload, use a smaller quantization, or leave the optional Qwen annotator disabled. Those choices trade capacity or speed for a smaller memory footprint.

## Getting started on Windows

Setup files are named for each platform: `Windows-Setup.bat`, `macOS-Setup.command`, and `Linux-Setup.sh`. Follow the section for your operating system below.

### 1. Get the project and vocabulary

Install Git with Git LFS, then clone the repository:

```powershell
git lfs install
git clone https://github.com/mikelej14/AevumAI.git
cd AevumAI
git lfs pull
```

The pretrained vocabulary is stored through Git LFS. Use an LFS-enabled clone to obtain the actual `pretrained_vocabulary/vocabulary.avnp` file.

### 2. Install the runtime

Install **64-bit Python 3.10 or newer**, including Tkinter, and make Python available on PATH. The setup script recommends Python 3.11 or 3.12 for prebuilt-wheel compatibility.

Run `Windows-Setup.bat` for the Vulkan-oriented Windows setup. It creates a virtual environment, installs the runtime and support packages, and checks the installation. If a compatible Vulkan wheel is unavailable, the source-build fallback requires Microsoft C++ Build Tools and the Vulkan SDK.

For a CPU-only installation, use `Windows-Setup-CPU-Only.bat`. Actual speed and memory requirements depend on the selected models, quantization, context size, and hardware.

### 3. Select your models

Download the Granite GGUF and, optionally, a Qwen GGUF using the links above. Model files can live anywhere on your computer.

Run `RUN.bat`, open **Settings**, and:

1. Select `granite-4.2-3b-Q6_K.gguf` under **Executive**.
2. Optionally select a Qwen3.5-2B GGUF under **Semantic Annotator** and enable it.
3. Save your settings and click **Start model**.

Start a conversation, or use **Memory → Import text / Markdown** to add material for later recall.

## Getting started on macOS

Use the same Git LFS clone steps and model downloads above. Install a **64-bit Python 3.11 or 3.12 from [python.org](https://www.python.org/downloads/macos/)**; its macOS installers include [Tkinter](https://www.python.org/download/mac/tcltk/).

In Terminal, from the project directory:

```bash
bash macOS-Setup.command
bash macOS-Run.command
```

The installer creates a local virtual environment and selects **Metal on Apple Silicon** or **CPU on Intel Macs**. It first tries a prebuilt runtime wheel, then a source build if needed. For the source-build fallback, install Apple's Command Line Tools with `xcode-select --install` and rerun setup. The backend installation follows the [llama-cpp-python installation guide](https://github.com/abetlen/llama-cpp-python#installation).

On Apple Silicon, use native arm64 Python rather than running under Rosetta. To request CPU-only installation, run `bash macOS-Setup.command --cpu`. To choose a particular Python installation, use `AEVUM_PYTHON=/path/to/python3 bash macOS-Setup.command`.

After launching, select the Executive and optional Semantic Annotator models in Settings, save, and click **Start model**. Keep the project in a writable folder so Aevum can save settings and memory.

## Getting started on Linux

Use the same Git LFS clone steps and model downloads above. Aevum needs a **graphical desktop session**, **64-bit Python 3.10+**, Tkinter, and Python's venv support. Install those prerequisites through your distribution's package manager. On Debian/Ubuntu, the packages are `python3`, `python3-venv`, and `python3-tk`; `build-essential` is needed if the runtime must be built from source.

From the project directory:

```bash
bash Linux-Setup.sh
bash Linux-Run.sh
```

The Linux installer creates a local virtual environment and installs the **CPU backend**, trying a prebuilt wheel before falling back to a source build. It does not install system packages or require sudo itself. To choose a particular Python installation, use `AEVUM_PYTHON=/path/to/python3 bash Linux-Setup.sh`.

In Settings, select the Granite GGUF and optional Qwen GGUF, save, and click **Start model**. This installer provides CPU inference; Linux GPU backend installation is not automated.

The macOS and Linux scripts are newly added. Script checks cover installation branches and failure handling; full GUI and model-runtime validation on those operating systems is still pending.

## Local data and connectivity

Local model inference and memory storage run on your computer. Web tools require internet access and send requests to external services. An optional hosted fallback is also available; the default model configuration uses local GGUF inference.

- `data/chats.json` stores visible chat history.
- `data/neural_memory/` stores persistent neural memory and provenance.
- `data/config.json` stores application settings and selected model paths.
- `logs/` contains runtime logs.

Back up the `data/` directory to preserve your chats, settings, and personal memory. Local data, downloaded models, and logs are excluded from Git tracking.

## Project status and documentation

Aevum AI is under active development. The current release is **0.2.4**.

- [Architecture](ARCHITECTURE.md)
- [Changelog](CHANGELOG.md)
- [Build status](BUILD_STATUS.md)
- [Release validation](RELEASE_VALIDATION.md)
- [Neural memory documentation](neural_docs/)

MicroBrain supplies the neural memory subsystem; Aevum provides the desktop assistant, model orchestration, and user interface. Earlier implementations are retained in `history/` for reference.

## License

Aevum AI is released under the [MIT License](LICENSE). Downloaded models are distributed under their respective licenses; see their Hugging Face model cards.
