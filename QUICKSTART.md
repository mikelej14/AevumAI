# Aevum AI quick start

**Install the app → choose Granite → save and load → wait for Ready → chat.**

For your first run, start with Granite only. You can add the optional Qwen annotator after the main model is working.

## 1. Install for your computer

Clone the repository with Git LFS so the pretrained vocabulary is included:

```text
git lfs install
git clone https://github.com/mikelej14/AevumAI.git
cd AevumAI
git lfs pull
```

Then use the setup and launcher for your operating system:

- **Windows:** run `Windows-Setup.bat`, then `RUN.bat`. Use `Windows-Setup-CPU-Only.bat` for CPU-only installation.
- **macOS:** in Terminal, run `bash macOS-Setup.command`, then `bash macOS-Run.command`.
- **Linux:** in a desktop session, run `bash Linux-Setup.sh`, then `bash Linux-Run.sh`.

See the [installation prerequisites and hardware guidance](README.md#system-requirements-and-memory-planning) before installing. The intended Granite Q6_K configuration needs a budget of roughly **6 GB VRAM**, with **8 GB or more recommended**. CPU inference uses system RAM; Apple Silicon uses shared unified memory.

## 2. Download the main model

Open [IBM's Granite 4.2 3B GGUF files on Hugging Face](https://huggingface.co/ibm-granite/granite-4.2-3b-GGUF/tree/main).

Download **`granite-4.2-3b-Q6_K.gguf`** and wait for the download to finish. Keep it in a folder where it can stay; Aevum remembers the file's location. You do not need to move it into the application folder.

## 3. Follow the in-app Quick start page

When Aevum has no usable Executive model configured, it opens **Quick start** automatically. You can also open **Quick start** from the sidebar at any time.

1. Under **Choose your main model**, click **Browse** and select your downloaded Granite `.gguf` file.
2. Leave **Enable optional Qwen annotator** unchecked for now.
3. Click **Save & load models** in the bar at the bottom of the Quick start page.
4. Wait while the status says **Loading models…**. The app saves your selection and loads the model; you do not need to press another Start button.
5. When it says **Ready**, click **Open chat**.
6. Send a simple first message, such as “Hello, introduce yourself.”

A selected path alone does not mean the model is running. Wait for **Ready** before chatting.

## Using Settings instead

The same model selections appear in **Settings**:

1. Under **Models**, click **Browse** beside **Executive · Granite/chat GGUF**.
2. Select `granite-4.2-3b-Q6_K.gguf`.
3. Scroll to the **bottom of Settings**.
4. Click **Save + reload model** to save your choices and load them in one step.
5. Wait for the header to say **Ready**, then open **Chat**.

**Save settings** only saves the configuration. If you choose that button instead, click **Start model** in the header afterward when the model is stopped. Use **Save + reload model** when changing a model that is already running.

## 4. Add Qwen when you are ready (optional)

[Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) can add topic, sentiment, urgency, and other semantic annotations to memory. Conversation and neural memory work without it.

Download a compatible model from [Unsloth's Qwen3.5-2B GGUF collection](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF). Choose the language-model GGUF; an `mmproj` vision-projector file is not the chat model.

In **Quick start**, Browse to the Qwen file under step 2, check **Enable optional Qwen annotator**, and click **Save & load models**. In Settings, use the **Semantic annotator** field and enable checkbox, then **Save + reload model** at the bottom.

The annotator is CPU-oriented by default and uses additional system RAM. If it cannot load while Granite succeeds, you can still chat.

## 5. Start building memory

Normal conversations are archived automatically. To add your own text, open **Memory → Import text / Markdown**. You can search the Memory page or ask about material you previously discussed.

Deleting a visible chat does **not** erase its stored neural memories. Back up the `data/` folder to preserve chats, settings, and personal memory.

## Next time you open Aevum

Your saved model paths remain in Settings. Click **Start model** in the header, wait for **Ready**, and continue chatting. If you move or delete the selected model file, Quick start opens again so you can choose its new location.

## If you get stuck

- **“Choose your Executive .gguf file”:** use Browse to select the downloaded file, then Save & load models.
- **“Not a GGUF model”:** wait for the download to finish and select the actual model, not a webpage, archive, or pointer file.
- **“Executive not loaded” or “Model stopped”:** save and load from Quick start, or use Save + reload model at the bottom of Settings.
- **Out of memory:** close other GPU-heavy apps, leave Qwen disabled, lower **Executive context** in Settings, or use a smaller Granite quantization. To use CPU inference, set **Executive GPU layers** to `0`, then Save + reload model.
- **Load failure:** open **Activity** for details. Check the selected file, available memory, and your platform's runtime installation.
- **Linux cannot open a window:** start from a graphical desktop session with Tkinter installed.
