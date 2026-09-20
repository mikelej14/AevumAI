# Cognitive Brain 0.1.33 — Live Chat-Answer Streaming Fix

## 0.1.33 focused verification

- Reproduced the UI behavior: native Granite Thinking streamed live, but normal answer content was not connected to the scheduler's live answer callback and therefore appeared as one completed block.
- Added live native answer forwarding with a protocol gate that suppresses fragmented `<tool_call>` XML while releasing ordinary assistant text immediately.
- Prevented the already-streamed native answer from being appended again during final tool/citation handling.
- Preserved the authoritative `visible_complete` reconciliation path and the existing direct self-state validation buffer.
- Added regression behavior for fragmented live answer tokens and fragmented XML tool-call tokens.
- Full automated suite: **97 tests passed**.

## 0.1.32 focused verification

- Reproduced the 0.1.31 failure with `executive_tool_rounds = 3`: Granite executed three real searches, the runtime passed an empty tool set on the next pass, Granite emitted a fourth native tool call, and the scheduler raised `Executive Model returned an empty response`.
- Fixed only the native tool-budget boundary in `core/scheduler.py`; no browser, memory, affect, appraisal, model loading, RMEM, prompt, or chat-storage behavior was redesigned.
- Added regressions for graceful tool-budget exhaustion and for Granite ignoring the first budget notice.
- Full automated suite: **97 tests passed**.


Focused correction of the native Executive architecture after 0.1.30 could stream Thinking but still allow a turn to end with the answer trapped inside reasoning.

## Root cause

0.1.30 still treated Granite native tooling as a special planning phase. It imposed a hidden 1,400-character reasoning ceiling and 900-token native generation budget, disabled thinking after certain tool results, omitted Granite's reasoning content from assistant tool-call history, and relied on later runtime answer/recovery layers. That could terminate a legitimate Granite reasoning block before `</think>` and leave no normal assistant content.

## Fixed

- Native Granite now owns one continuous turn: reasoning → optional tool call → real tool result → further reasoning if useful → Granite's own final assistant content.
- Python remains only the validator/executor/security boundary for tools.
- Assistant tool-call transcript entries preserve the round's `<think>...</think>` content alongside `tool_calls`.
- Tool-role messages contain plain JSON tool results as Granite's template expects.
- Thinking stays enabled after tool use and when the tool budget is exhausted; tool availability no longer controls reasoning mode.
- Hidden native limits of 1,400 reasoning characters / 900 output tokens are removed. Native turns honor the Executive `thinking_max_chars` and `max_tokens` settings.
- FAST/ASSISTED native turns use low-effort reasoning; DEEP uses high effort.
- Native Granite's final content is explicitly delivered to the normal chat answer callback.
- The scheduler no longer performs the separate reasoning-only final-answer recovery for native Granite and no longer runs native final content through the legacy tool-narration repair.
- A native Thinking-only response is never promoted/extracted into chat text and never triggers a silent second tool-less Executive generation.

## Preserved

RMEM storage/format, affect/appraisal math, Observer/Memory worker behavior, browser implementation and validation, tool receipts, chat persistence, 0.1.28 answer reconciliation safety net, settings UI, setup scripts, and non-native Executive compatibility paths remain intact.

## Verification

- Full automated suite: **95/95 tests pass**.
- Native multi-tool regression verifies reasoning is retained with assistant tool calls, tool messages are plain JSON, tools remain available across rounds, thinking remains enabled, and configured Executive budgets are honored.
- Native no-tool regression verifies live Thinking plus Granite's own final content through the normal answer callback.
- No-extraction regression deliberately returns a Thinking-only native turn and verifies zero answer tokens, one Granite call only, and a hard failure instead of converting reasoning into chat text.
- Tk/Xvfb probe verifies multiple live Thinking updates and the final response becoming visible through `append_answer`, not reconciliation.
- Full Python compilation passes for all **39 Python files**.
- All six Tk/Xvfb UI probes pass.
- Clean-package extraction, test-suite, and ZIP integrity verification are performed on the release archive before handoff.
