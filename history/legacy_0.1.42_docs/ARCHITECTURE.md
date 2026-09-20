# Cognitive Brain 0.1.33 Architecture


## 0.1.33 native answer streaming boundary

The native Executive path streams two independent channels: reasoning goes to Thinking, while ordinary assistant content goes to Chat. The scheduler forwards normal answer chunks as they arrive rather than waiting for the native pass to complete. Because Granite may represent native tool calls as XML content, a minimal stream gate suppresses content whose opening resolves to `<tool_call>` and releases ordinary text once it is no longer ambiguous. Final committed text remains authoritative and may reconcile citation/source formatting at completion.


## 0.1.32 tool-budget boundary

The configured native tool-round limit is an execution budget, not a conversation terminator. If Granite requests another tool after the budget is spent, Python does not execute the call and does not end the turn. It returns a `role="tool"` budget-exhausted result inside the same transcript and asks Granite to finish from existing evidence. A single bounded final-answer-only Granite retry exists only if the model ignores that notice and requests yet another tool. Runtime code never converts private Thinking into answer text.

## 0.1.31 native Executive contract

Native Granite is an agent loop, not a planner followed by a runtime-authored answer stage:

```text
user message
    ↓
Granite reasoning  ─────────────→ Thinking UI
    ↓
optional native tool_call
    ↓
Python validates + executes
    ↓
role=tool JSON result
    ↓
Granite reasoning continues as needed
    ↓
Granite normal assistant content ─→ Chat UI
```

The scheduler may validate side effects, enforce tool-round limits, route reasoning to Thinking, and route assistant content to Chat. It must not derive a user-facing answer from private reasoning. A native tool-call assistant message preserves that turn's reasoning content with its `tool_calls`, matching Granite 4.2's expected multi-turn tool transcript. Thinking remains enabled after tool results; tool exhaustion does not force a non-thinking answer stage.

Native turns use the configured Executive output/guard settings rather than hidden 0.1.30-specific 900-token/1,400-character limits. Low/high reasoning effort remains a model-template choice based on scheduling mode.


## 0.1.30 native-tool streaming boundary

Native-tool capability no longer implies buffered cognition. The Executive receives its real tool definitions exactly as in 0.1.29, but the native tool-selection/finalization pass runs with streaming enabled. Reasoning is forwarded immediately to the public Thinking channel while ordinary content remains buffered until the scheduler knows whether it is a final answer or tool protocol. This preserves Granite's native tool authority without sacrificing live cognitive observability.

The worker also accumulates structured `delta.tool_calls` fragments during streaming and returns them in the same normalized completion object used by the scheduler. Granite's XML tool wire format remains supported as a content fallback. The runtime never streams tool-call markup into the answer card. Completed reasoning is persisted with the assistant turn, but is not replayed to the UI after streaming.


## 0.1.25 tool presentation boundary

Private tool payloads still remain outside ordinary user/model dialogue, but every executed action now produces a bounded public receipt. A start event reaches the active assistant card before network or memory work begins. Completion records request metadata, success/failure, elapsed time, freshness/provider fields, and source links without exposing fetched page bodies or private memory contents.

Completed receipts are stored beside the assistant message in `chats.json`; the scheduler's JSONL trace also retains the same structured `tool_activity` list. Therefore chat presentation, persisted history, and audit logs agree about whether a tool actually ran.

## 0.1.24 protocol boundary

The Executive's ordinary system prompt no longer invents textual tool syntax. Native tool-capable templates receive real function definitions through their embedded chat-template interface; compatibility planners retain their separate structured decision prompt.

For a native-tool Executive, tool availability follows runtime capability rather than Python intent classification. Memory tools are supplied as native functions on ordinary turns; web tools are supplied whenever the browser is enabled. Granite decides whether to call them. Python validates and executes the call, then returns the result through a real `tool` role. Current weather remains a structured Open-Meteo capability, but Granite must select `current_weather` itself rather than Python pre-inserting the lookup.

## Native Executive tool loop

When the Executive GGUF advertises tool support in its embedded chat template, the scheduler uses that protocol directly on every turn. The same conversation contains the user request, Granite's structured tool call when it chooses one, a `tool`-role result, any subsequent call, and the final answer. Granite XML calls and OpenAI-style decoded calls normalize to the same internal action packet. Python never decides that a native model is ineligible to see a tool merely because a keyword heuristic missed the request; Python still owns validation and every side effect.

GGUFs without native tool-template support continue through the bounded structured-planner compatibility path. That compatibility path may still use deterministic intent gating/fallback because those models cannot make the native function decision themselves. Web evidence and recent history have explicit payload budgets so tool turns leave space for a user-facing answer in the configured context window.

## 0.1.17 presentation layer (preserved in 0.1.21)

The UI remains a first-party in-process Tk shell over the same cognitive/runtime objects. 0.1.17 changes presentation only: `ui/theme.py` owns the visual tokens and raised controls; `ui/chat_widgets.py` owns message/Thinking presentation; `ui/main_window.py` wires those visuals to the existing handlers. The cognitive engine, RMEM and chat-store architecture remain unchanged.

Assistant avatars derive their initials from each displayed assistant name, so visual identity follows the configured/stored name instead of a hardcoded letter.

## First principle

Conversation, long-term experience, and model cognition are related but not the same storage problem.

```text
Visible session continuity          Lived / associative memory
        chats.json                         experience.rmem
            |                                   |
            +-------- current chat --------------+
                         |
                 Cognitive Scheduler
                  /               \
             CPU Qwens         Executive GGUF
```

## Session layer

`core/chats.py` owns visible multi-chat persistence. It is intentionally independent from RMEM so losing/replacing RMEM does not erase chat transcripts and deleting a chat does not require rewriting an append-only memory file.

Each chat stores:

- stable chat ID;
- title / created / updated time;
- ordered user/assistant messages;
- Thinking text for assistant turns;
- completed tool receipts for assistant turns;
- assistant display name;
- turn ID and optional RMEM record provenance.

The scheduler's immediate conversation history comes from the active chat only. This prevents one chat's recent dialogue from leaking into another chat's working context.

RMEM remains global by design: a relevant lived experience from another chat may still be recalled associatively.

## Deletion semantics

Deleting a chat means deleting the transcript/session. It does not erase or rewrite historical RMEM records. The UI tells the user this before deletion.

If every chat is deleted, a fresh `New chat` is created automatically.

## Legacy migration

On the first 0.1.16 launch only, if no session transcript exists and RMEM is available, legacy `turn` records are copied into a `Previous conversation` chat. `legacy_import_done` is then persisted so later chat deletions are final from the session UI and cannot be repopulated from RMEM.

## RMEM resilience

`ResilientMemory` wraps the binary RMEM engine. If the pack disappears or becomes invalid during runtime, it moves to an offline state. Retrieval returns no evidence and memory writes return record ID 0, allowing conversation to continue. The Memory page can reopen/create a valid pack.

This prevents a deleted pack from being silently recreated by append mode without a valid header.

## UI shell

The Tk interface is now a single dark application shell:

- fixed left rail: brand, New Chat, scrollable chats, page navigation, RMEM health;
- top application header: page/chat title, model status, Start/Stop controls;
- central page stack: Chat, Cognitive State, Memory, Activity, Settings.

The Chat page retains the critical streaming contract:

- answer text streams to the assistant response;
- reasoning streams only to the expandable Thinking panel;
- each Thinking panel owns its own wheel events;
- the transcript scroll owns normal chat wheel events;
- streaming only auto-follows a viewport that was already at its bottom.

## Cognitive/model architecture preserved from 0.1.15

- model-agnostic Executive Browse slot with family presets;
- Qwen3.5 0.8B observer and memory workers on CPU;
- asynchronous cognition / FAST path;
- deterministic affect/self-state;
- hidden memory and web tools;
- native/fallback Thinking transport;
- proprietary binary RMEM.


## 0.1.21 memory-facade boundary fix
The scheduler no longer reaches into CognitiveRMEM private tokenization internals. Both CognitiveRMEM and ResilientMemory expose the same public `tokenize()` contract, and scheduler fast-recall uses that contract.


## 0.1.22 final-answer state invariant

For native-tool Granite, current/public-data retrieval is model-selected: the real tools are present whenever enabled, and the private native-tool policy tells Granite to use an applicable web tool rather than claim it has no real-time access. Python executes valid calls and never fabricates a tool result. For non-native compatibility models, the legacy explicit-web fallback remains. Any Executive generation containing private reasoning but no visible answer still receives a bounded final-answer recovery pass.


### Executive reasoning effort
Granite 4.2 has a native low-effort reasoning mode. The scheduler uses low effort for FAST/ASSISTED turns and high effort only for DEEP turns. This keeps visible reasoning useful without allowing ordinary answers to become recursive internal debate. Per-turn reasoning guards provide a deterministic escape into the existing non-thinking final-answer recovery.

## Authoritative answer delivery

Streaming answer tokens are an incremental UI preview, not the completion contract. The scheduler owns the authoritative `assistant_text`. After persistence it emits `visible_complete` with that exact text; the UI reconciles its card to the committed answer and `turn_done` repeats the reconciliation before releasing the turn. This guarantees that buffered/recovered answers cannot remain trapped behind the Thinking view when live answer-token delivery is absent or partial.
