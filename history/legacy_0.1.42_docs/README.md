# Cognitive Brain 0.1.33

A local multi-model cognitive chat runtime with persistent lived memory, asynchronous subconscious workers, a model-agnostic Executive GGUF slot, private web/memory tools, and a new dark multi-chat interface.


## 0.1.33 live chat-answer streaming fix

0.1.33 fixes the visible-answer buffering regression in the native Granite agent loop. 0.1.32 already streamed the dedicated Thinking channel, but `_native_tool_phase()` did not forward Granite's normal content callback. The final chat answer therefore appeared only after the entire native pass completed.

Native Granite answer chunks now flow through the existing chat `on_token` callback as they are generated. A small protocol gate holds only the ambiguous opening of a pass and suppresses Granite's `<tool_call>...</tool_call>` XML wire format, so native tool syntax does not flash into the answer card. The native tool policy also explicitly requires tool planning to stay in the reasoning channel and forbids normal assistant preamble before a tool call.

Tool-assisted answers no longer get appended a second time after they have already streamed. The existing `visible_complete` reconciliation remains authoritative so citation-link/source-footer cleanup can still synchronize the final rendered answer without sacrificing live generation. Direct self-state questions retain their existing validation/repair buffer so a rejected generic AI disclaimer is not flashed before repair.

Regression coverage now streams fragmented normal answer chunks and fragmented XML tool-call chunks through the same native path, verifying that answer chunks arrive individually while tool protocol remains hidden. The full suite passes **97/97 tests**.


## 0.1.32 native tool-budget finalization fix

0.1.32 fixes a native Granite agent-loop failure where a turn could stop after exactly the configured number of verified tools and never produce an answer. In 0.1.31, once `executive_tool_rounds` was exhausted, the scheduler removed the tool definitions. If Granite then attempted one more native tool call, the protocol parser correctly stripped that tool-call text from the user-facing answer, leaving an empty response and ending the turn.

The budget is now a graceful stop rather than a dead end: an over-budget call is rejected back into the same assistant/tool transcript with a normal `role="tool"` error telling Granite that no more tools can execute and that it must answer from the evidence already gathered. Granite then gets a final model-owned answer pass. If it ignores that notice and asks for another tool again, one bounded non-thinking Granite retry requests only the final answer. Thinking is never copied or promoted into chat text, and no over-budget tool action is executed.

Two regressions cover both the normal budget-exhaustion path and the bounded retry path. The full suite passes 97 tests.

## 0.1.31 unified native Granite agent loop

0.1.31 corrects the architecture rather than adding another answer-recovery heuristic. For a native-tool-capable Granite Executive, one model-owned conversation now performs the complete turn: Granite reasons, chooses whether to call a supplied native tool, Python validates/executes that call, the JSON tool result is returned as a real `role="tool"` message, Granite may reason again, and Granite itself emits the final assistant content. Python does not promote, extract, or rewrite text from Thinking into the chat answer.

The native tool transcript now matches Granite 4.2's documented multi-turn contract more closely: an assistant tool-call message keeps that round's `<think>...</think>` content alongside `tool_calls`, and the following tool message contains the ordinary JSON result rather than the runtime's private evidence wrapper. Tools remain available for subsequent native decisions until the configured tool-round limit, while thinking remains enabled independently of tool availability.

0.1.30 also imposed hidden native-pass limits of 1,400 reasoning characters and 900 generation tokens. Those overrides are removed. Native Granite now uses the Executive's configured `thinking_max_chars` and `max_tokens`; FAST/ASSISTED still request Granite's native low-effort mode and DEEP requests high effort. This gives Granite room to close its reasoning block and produce its own normal answer instead of being cut off while the answer is still inside Thinking.

The system contract now states explicitly that private reasoning must end with user-facing assistant content outside the reasoning block. Native final content is sent through the normal chat answer callback; `visible_complete` reconciliation remains only a delivery safety net. If a deliberately malformed native model turn produces Thinking but zero assistant content, the runtime refuses to convert that Thinking into an answer and does not silently launch a second tool-less Executive turn.

## 0.1.30 live native-tool Thinking repair

0.1.29 fixed native tool authority but introduced an observability regression: because every native-tool-capable Executive turn entered `_native_tool_phase()` and that phase called Granite with `stream=False`, Granite could reason normally yet the Thinking panel received the reasoning only after the entire native pass returned. This affected even turns where Granite ultimately used no tool.

0.1.30 keeps Granite in charge of native tool selection while changing that phase to a streaming transport. Reasoning chunks are forwarded to the existing Thinking callback immediately. Answer text and tool protocol text remain buffered until the pass is classified, so an XML/native tool call can never leak into the answer card. Structured streaming `tool_calls` are accumulated across deltas, while Granite XML tool calls remain supported through the existing parser. If a third-party chat handler explicitly rejects streaming automatic tool choice, the runtime falls back to the older buffered transport for that handler rather than breaking the turn.

The saved assistant Thinking transcript now retains the native-tool reasoning rounds that were actually shown live. The scheduler no longer replays the completed reasoning aggregate after the pass, preventing duplicate text at turn completion. No reasoning budget, tool-authority policy, RMEM format, affect/state math, browser implementation, or answer-delivery reconciliation was changed.

Regression coverage includes a native-tool-capable Granite turn that uses no tool: the worker must receive `stream=True`, three separate reasoning chunks must reach the thought callback before completion, and the final saved Thinking text must match without duplication. A worker-level test also verifies fragmented structured tool-call deltas survive the streaming transport. The Tk/Xvfb live-thinking probe confirms multiple thought chunks are visibly present before the final answer is committed.

## 0.1.29 native tool authority repair

0.1.29 removes the Python intent gate from Granite/native-tool turns. If the Executive GGUF advertises native tool support, the runtime supplies the enabled native memory tools on every turn and supplies the native web tools whenever the browser capability is enabled. Granite decides whether a tool is needed; Python only validates arguments, executes the side effect, records the receipt, and returns the result as a real `tool`-role message.

This fixes questions such as **“Who won the last Fisher Cats game?”** that are plainly searchable but did not match the old `_web_tools_needed()` keyword rules. In 0.1.28 that wording could hide `web_search` from Granite entirely, forcing a stale-weights disclaimer. In 0.1.29 Granite sees `web_search`, `web_fetch`, and `current_weather` regardless of that heuristic and is explicitly told that supplied tools are real runtime capabilities: for current/recent/externally verifiable information it should use them instead of claiming it has no real-time access. Stable common-knowledge questions can still be answered directly without a tool call.

The old deterministic web-intent gate remains only for Executive models whose chat template does **not** support native tool calling. Native Granite no longer receives forced weather/search calls from Python; its own native function call is the authority for tool selection. Python remains the security boundary for URL validation, search-query validation, surfaced memory IDs, maximum rounds, and browser enable/disable state.

Regression coverage now includes a Fisher Cats-style sports question that deliberately fails the legacy web heuristic but succeeds through Granite's native `web_search`, plus a static common-knowledge question that receives the same tool definitions and correctly makes zero browser calls.
The clean extracted release passes **92/92 automated tests** and compiles all **38 Python files**.

## 0.1.28 answer-delivery runtime repair

0.1.28 fixes the actual failure where the Executive had a valid final answer internally but the chat card could remain blank. Live `on_token` callbacks are now treated only as a streaming preview. The scheduler's committed `assistant_text` is sent with `visible_complete`, the Tk chat card reconciles to that authoritative text immediately, and `turn_done` performs the same reconciliation again before the turn is released. Recovery, buffered self-state/tool answers, citation cleanup, or a missed token callback can therefore no longer leave a successful turn with Thinking visible but no answer.

The Executive reasoning watchdog is also no longer gated by detection of one specific GGUF `enable_thinking` template variable. If an Executive call requested thinking and emits reasoning, the guard can stop a reasoning-only runaway even when template capability metadata is incomplete. A reasoning-only turn still receives the bounded final-answer fallback.

Regression coverage includes: a completely unstreamed committed answer becoming visible, partial-stream reconciliation without duplication, repaired-draft replacement, an authoritative answer attached to `visible_complete`, and a Granite-style `reasoning_content` stream whose template does not advertise `enable_thinking`. The full suite passes 91 automated tests, plus a real Tk/Xvfb probe that starts with a blank answer card and verifies the committed final answer becomes visible.

## 0.1.27 reasoning control repair

0.1.27 added Granite 4.2 low-effort reasoning for FAST/ASSISTED turns, high effort for DEEP turns, and per-turn reasoning ceilings. It also strengthened the self-state response contract and retained a bounded non-thinking recovery pass. 0.1.28 keeps those controls but fixes the separate answer-delivery/UI synchronization defect they did not address.

## 0.1.25 visible tool receipts

Tool execution is now visible inside each assistant message in its own live, expandable **Tools** panel, separate from both the answer and Thinking. The panel appears as soon as Python starts a tool and records the exact request, success/failure, elapsed time, and user-verifiable result metadata.

Current-weather receipts include the resolved location, coordinates, local observation time and timezone, condition, temperature, apparent temperature, humidity, wind, provider, and a clickable source URL. Search receipts include the exact query, provider, result count, and clickable results; fetched pages report their title, URL, and retrieved character count. Tool receipts persist with the chat and return after restarting the app.

Successful online answers also receive clickable source attribution when a small Executive omits it. The chat status changes immediately when retrieval begins, so fast tool turns no longer look frozen or indistinguishable from an unsupported model guess.

## 0.1.24 identity/tool-protocol correction

0.1.24 removes the conflicting bracketed pseudo-tool reference left in the default Executive prompt. Real tool definitions now come only from the runtime/template protocol. Existing 0.1.23 default prompts migrate automatically; deliberately customized prompts remain untouched.

The configured local identity is now explicit: the Executive is not instructed or permitted to identify itself as ChatGPT, Codex, OpenAI, or another hosted assistant. Meta drafts such as “The user says…,” “As ChatGPT…,” invented textual tool calls, and tool-planning narration are rejected before display and receive one bounded final-answer repair. Native tool-planning thoughts remain private instead of being copied into the visible Thinking panel.

Current-weather questions now use a dedicated live Open-Meteo lookup with location resolution and observation time instead of trusting a search-result snippet. The measurement is returned through the same native tool-role transcript before the Executive answers.

## 0.1.23 native tool-conversation repair

0.1.23 fixes the deeper integration error left in 0.1.22. Granite 4.2's embedded chat template already defines a native tool protocol, but earlier builds ignored it: a separate constrained-JSON planner ran, its conversation was discarded, and retrieved evidence was buried in a new system prompt. Native-tool GGUFs now receive real tool definitions and keep assistant tool calls, tool-role responses, and the final answer in one correctly formatted conversation. Granite XML and OpenAI-style tool-call responses are both supported.

Explicit current/web requests still cannot silently skip retrieval. If the model does not issue the required call, the runtime performs a bounded deterministic search (or directly fetches a user-supplied URL), returns that result through the same native tool transcript, and requires a final answer. Search and fetched-page payloads plus recent history are bounded to leave generation room in the default 8K context.

Models whose embedded templates do not advertise native tools keep the 0.1.22 structured-planner fallback.

## 0.1.22 tool/answer state-machine hotfix

0.1.22 fixes the explicit-web/Granite reasoning-only failure seen in 0.1.21. Current public-data requests such as current weather now enter the browser path, explicit web requests get a deterministic real-search fallback if the Executive planner returns `none`, and a turn that produces private reasoning but no visible answer is automatically finalized before the turn can complete.

- The default System Prompt is now a short quick-reference rather than a numbered rule list.
- 0.1.22 named tools directly in the prompt; 0.1.24 removes that superseded pseudo-syntax because it conflicted with native tool templates.
- Identity, Personality, Self-State, and Background wrappers are shorter while remaining separate and private.
- Simple/casual replies are explicitly concise by default; detailed tasks may still expand freely.
- Known shipped 0.1.18 default prompts migrate automatically to the compact default. Deliberately custom prompts are preserved.
- The hidden tool-decision prompt remains separate and only appears when a tool-planning pass actually runs.

All model/runtime, RMEM, browser, asynchronous cognition, multi-chat, Thinking, and UI behavior from 0.1.18 is preserved.

## 0.1.16 highlights

- Full modern dark UI redesign.
- Left multi-chat rail with new/switch/rename/delete.
- Chat transcripts persist independently in `data/chats.json`.
- Per-chat recent context; global RMEM associative recall remains available.
- Deleting a chat does not rewrite RMEM.
- One-time import of legacy RMEM transcript into `Previous conversation`.
- App continues if RMEM is missing/corrupt, with a visible offline state and reconnect control.
- Expanded Thinking remains separately streamed and independently scrollable.
- Existing 0.1.15 model presets, browser tools, async cognition, prompts, and RMEM format are preserved.

## Session vs memory

`chats.json` is the canonical visible transcript store. `experience.rmem` is long-term associative/lived experience. This separation is deliberate: session deletion and RMEM loss no longer destroy each other.

---

A local persistent cognitive chat runtime built around:

- **any chat-capable GGUF supported by llama-cpp-python** as the GPU Executive/conversational model;
- **1Qwen 0.8B** on CPU for narrow perception/attention signals;
- **2Qwen 0.8B** on CPU for memory relevance and episodic consolidation;
- deterministic Python appraisal/affect/state;
- binary RMEM persistent experience storage.

Example current setup:

```text
Executive: granite-4.2-3b-Q5_K_S.gguf GPU
1Qwen:    Qwen3.5-0.8B-Q4_0.gguf      CPU / thinking OFF
2Qwen:    Qwen3.5-0.8B-Q4_0.gguf      CPU / thinking OFF
```

The Executive Browse field is deliberately model-agnostic. It does not whitelist architectures. The two subconscious slots remain intentionally Qwen3.5 because their prompts/schemas are designed for those workers.

Both CPU roles may point to the same physical 0.8B GGUF. No `mmproj` is needed for text-only operation.

## 0.1.11+ asynchronous cognition

The CPU models are no longer a mandatory gate before every Executive response.

Adaptive scheduling has three paths:

```text
FAST       ordinary turn
           RMEM + current state -> Executive immediately
           1Qwen/2Qwen work in the background

ASSISTED   explicit continuity cue
           wait only for 1Qwen perception/expanded retrieval
           -> Executive
           2Qwen continues in background

DEEP       explicit autobiographical/deep-recall request
           1Qwen -> expanded RMEM -> 2Qwen -> optional memory tools
           -> Executive
```

`adaptive` is the default. Settings can force `fast`, `assisted`, or `deep` for testing.

On FAST turns, background cognition is queued **before** GPU inference starts. 1Qwen perceives the interaction, 2Qwen integrates relevant experience, Python updates appraisal/state, and 2Qwen consolidates the completed episode after the Executive response. The UI becomes ready for the next user turn without waiting for that consolidation to finish.

Background turns are serialized in conversation order and carry immutable turn IDs. RMEM/state access and individual model-worker requests are thread-safe, preventing late subconscious results from contaminating another turn.

The Activity tab reports scheduling mode, pre-Executive latency, first-output timing, background stages, and background completion.

## Conversation contract

The Executive receives a conventional chat layout:

```text
SYSTEM
  operating rules
  assistant identity
  personality
  optional quiet relevant memory/background

recent user/assistant history

USER
  exactly what the user typed
```

No runtime block, XML tag, memory packet, or cognitive metadata is appended to the current user message.

The configured **Assistant name** is explicitly the AI's own name. Assistant identity labels are stripped from recalled memory before memory is shown to the Executive.

Numeric affect/state stays inside Python. The Executive always receives a compact qualitative **first-person self-state** (for example: `generally okay; calm; mildly curious; reasonably confident`) so it has something genuine to consult for ordinary self-report. Raw scores and variable names never cross the boundary. Pronounced states may additionally yield a small behavioral response tendency.


## 0.1.12 first-person self-state

The persistent affect engine is no longer invisible to consciousness when state is neutral. Before every visible Executive response, Python projects the private numeric state into a short qualitative self-state and places it in the single system message.

The Executive is told to treat this as its own current functional/affective condition, not as user data or runtime telemetry. Casual questions such as `how are you?` should therefore be answered naturally in first person from that state rather than with a canned `I am an AI and do not have feelings` response.

The app still does not invent a physical body or biological sensations. If asked directly about biological emotion, sentience, or embodiment, the Executive should distinguish the software's functional state from those claims.

## Roles

**1Qwen — perception**
- intent/focus;
- entities;
- explicit goals/corrections;
- tone/hostility/praise/frustration/urgency;
- salience and memory-search cues.

**2Qwen — experience**
- integrates RMEM candidates;
- reconstructs prior experience;
- reports contradictions/open questions;
- consolidates completed turns after the Executive responds.

**Python**
- async scheduler and IPC;
- deterministic appraisal;
- persistent affect/state;
- RMEM writes/retrieval;
- memory + web tool execution, validation, and source handling;
- turn ordering and concurrency protection.

**Executive — user-selected GGUF**
- the only conversational voice;
- receives normal conversation plus sparse private background;
- uses the GGUF's embedded native thinking template;
- reasoning streams to the existing expandable Thinking pane.

## Identity and prompts

Settings keeps three separate values:

- **Assistant name** — the AI's own conversational name;
- **System prompt** — operating rules and tool/memory policy;
- **Personality prompt** — voice and temperament only.

Qwen's template requires one system-role message at the beginning, so these are compiled into distinct sections of that one system message while remaining independently editable and persisted.

## Private tools

The Executive has four real private operations executed by Python:

- `memory_search(query, limit)` — search autobiographical/project RMEM;
- `memory_get(record_id)` — fetch an already surfaced RMEM record;
- `web_search(query, limit)` — search the public web through DuckDuckGo HTML with Bing fallback;
- `web_fetch(url)` — fetch readable HTML/text from a public page.

Tool calls are never simulated by prompt text. Web search queries are length/shape checked so a model cannot dump an entire conversational prompt into a search engine. `web_fetch` rejects localhost/private-network targets and is restricted to URLs the user supplied or a prior `web_search` surfaced. Search snippets are treated as discovery evidence; fetched-page text is passed to the Executive as stronger source evidence with its URL.

Latency remains selective: ordinary offline chat skips the hidden tool planner. Explicit web/search/URL/freshness requests invoke it even on a FAST cognitive turn, while DEEP autobiographical turns can invoke the memory tools. Browser settings are available in Settings and require no API key or localhost service.


## 0.1.15 browser/tool restoration

0.1.15 restores a real in-process browser layer after 0.1.14 only exposed RMEM tools. The browser is part of the Python runtime, not another local server. Explicit online/current requests can run `web_search` and then `web_fetch` before the visible Executive answer, while ordinary turns keep the low-latency asynchronous path.

## Executive model runtime

The Executive slot does **not** restrict `general.architecture`. Browse to any GGUF that the installed llama-cpp-python build can load for chat.

Runtime behavior:

- embedded `tokenizer.chat_template` is preferred when present;
- if the template exposes `enable_thinking`, native thinking is enabled for visible Executive generation and routed to the expandable Thinking pane;
- Granite 4.2 ordinary/assisted turns use its native low-effort reasoning mode; DEEP turns retain high-effort reasoning, preventing simple answers from spiraling into repeated self-reconsideration;
- reasoning budgets are adaptive by scheduling mode (FAST 1400 chars, ASSISTED 2200, DEEP 4000) before the existing non-thinking answer recovery path takes over;
- models that expose a dedicated reasoning field or emit `<think>...</think>` are also separated into Thinking;
- models without native thinking simply answer normally;
- if a chat template rejects the `system` role, Cognitive Brain moves private operating context into an earlier internal primer pair while keeping the **current user message unchanged**;
- if no embedded template exists, llama-cpp-python's own detected/configured chat handler is allowed to try instead of rejecting the GGUF by architecture.

Executive sampler mode defaults to **Auto**. The app applies a model-family preset using **GGUF metadata first and filename second**. Browse immediately estimates the preset from the filename; after load, `general.architecture`, `general.name`, and the embedded chat template refine/override that estimate. Presets currently cover Qwen 2/3/3.5, Granite (including 4.2 thinking), Gemma 2/3, Mistral/Mixtral, Llama, Phi, DeepSeek/R1, Command-R/Cohere, Yi, Falcon, and a generic fallback.

A preset controls sampler defaults, system-role adaptation, and thinking transport only. It is never a whitelist. The selected GGUF's embedded chat template remains authoritative for conversation formatting. Unknown/future chat-capable GGUFs still load through the generic preset. Switch to **Custom** in Settings only when you intentionally want to override the detected sampler defaults.

The two CPU workers remain fixed Qwen3.5 non-thinking runtimes and still validate `general.architecture=qwen35`.

## Windows

Run:

```text
SETUP.bat
RUN.bat
```

Then choose all three GGUF paths in Settings and save/reload models.
