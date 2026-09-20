# Optional OpenAI Provider — Aevum AI 0.2.3

OpenAI is retained as an **optional alternate Executive provider**. The default product path is local GGUF/Granite. Switching to OpenAI must not change the memory architecture.

The OpenAI runtime therefore receives the same recent chat window and metadata-only automatic neural hints. Its neural function tools return complete decoded documents privately to the model, but the persisted/displayed tool event is compact and excludes recalled body text. Neural search receipts are eligible for compact neural archival; opening a document is not duplicated as a new memory.

The API key is session-only or read from `OPENAI_API_KEY` and is never written to `data/config.json`.

The OpenAI provider currently focuses on chat + neural-memory function tools. The **local Granite provider is the feature-complete path for the retained local web_search/web_fetch/current_weather toolset** in this release.
