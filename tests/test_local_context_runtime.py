from core.local_agent_runtime import LocalAgentRuntime


class FakeMemory:
    def hint_search(self, query, top_k=6, **kwargs):
        return [{
            "document_id": 4, "timestamp_utc": "2026-09-12T12:00:00+00:00",
            "speaker": "Mike", "role": "user", "match_tokens": ["CAR"],
            "semantic": {"topics": ["vehicle"], "context": "discussing a car repair"},
        }]

    def search_full_documents(self, query, **kwargs):
        return [{
            "document_id": 4, "timestamp_utc": "2026-09-12T12:00:00+00:00",
            "speaker": "Mike", "role": "user", "match_tokens": ["CAR"],
            "text": "THE CAR NEEDS A NEW BELT AND WE TALKED ABOUT IT FOR A WHILE",
        }]

    def open_document(self, document_id):
        return {"document_id": document_id, "sentence": "THE CAR NEEDS A NEW BELT"}


CFG = {
    "chat": {"system_prompt": "Be useful."},
    "identity": {"user_name": "Mike", "assistant_name": "Ava"},
    "runtime": {"auto_memory_hints": 6},
    "browser": {"enabled": False},
    "models": {"executive": {}, "annotator": {"enabled": False}},
}


def test_auto_hints_are_metadata_only(tmp_path):
    rt = LocalAgentRuntime(FakeMemory(), CFG, data_dir=tmp_path)
    hints = rt._automatic_hints("what did we say about the car")
    system = rt._system_prompt("what did we say about the car", hints)
    assert "doc=4" in system
    assert "topics=vehicle" in system
    assert "THE CAR NEEDS A NEW BELT" not in system
    assert "metadata only" in system.lower()


def test_explicit_neural_search_returns_full_docs_but_receipt_does_not(tmp_path):
    rt = LocalAgentRuntime(FakeMemory(), CFG, data_dir=tmp_path)
    private, receipt, note = rt._execute_tool("neural_memory_search", {"query": "car", "top_k": 4}, user_text="car")
    assert private[0]["text"].startswith("THE CAR NEEDS")
    assert "text" not in receipt
    assert "THE CAR NEEDS" not in str(receipt)
    assert receipt["documents"][0]["document_id"] == 4
    assert "doc 4" in note


def test_web_search_receipt_never_contains_snippet_or_page_body(tmp_path):
    rt = LocalAgentRuntime(FakeMemory(), CFG, data_dir=tmp_path)
    rt.browser.search = lambda query, limit=8: {
        "ok": True, "query": query, "provider": "fake",
        "results": [{"title": "Result One", "url": "https://example.com/a", "snippet": "SECRET BODY"}],
    }
    private, receipt, note = rt._execute_tool("web_search", {"query": "test", "limit": 8}, user_text="why")
    assert private["results"][0]["snippet"] == "SECRET BODY"
    assert "SECRET BODY" not in str(receipt)
    assert "SECRET BODY" not in note
    assert receipt["results"][0] == {"title": "Result One", "url": "https://example.com/a"}

class _Status:
    loaded = True


class FakeExecutive:
    def __init__(self):
        self.model_info = {"native_tools": True}
        self.calls = []
        self.n = 0
    def status(self):
        return _Status()
    def chat(self, messages, **kwargs):
        import copy
        self.calls.append(copy.deepcopy(messages))
        self.n += 1
        if self.n == 1:
            return {"result": "", "thinking": "", "tool_calls": [{"id": "c1", "function": {"name": "neural_memory_search", "arguments": '{"query":"car","top_k":2}'}}]}
        return {"result": "I found the old car discussion.", "thinking": "", "tool_calls": []}


def test_native_granite_loop_gets_recent_chat_hints_and_private_full_memory(tmp_path):
    rt = LocalAgentRuntime(FakeMemory(), CFG, data_dir=tmp_path)
    rt.executive = FakeExecutive()
    events = []
    out = rt.respond([
        {"role": "user", "content": "we talked yesterday"},
        {"role": "assistant", "content": "yes"},
        {"role": "user", "content": "what did we say about the car?"},
    ], on_tool=events.append)
    assert out["text"] == "I found the old car discussion."
    first_system = rt.executive.calls[0][0]["content"]
    assert "AUTOMATIC NEURAL MEMORY HINTS" in first_system
    assert "THE CAR NEEDS A NEW BELT" not in first_system
    assert rt.executive.calls[0][-1]["content"] == "what did we say about the car?"
    second = rt.executive.calls[1]
    tool_msg = next(m for m in second if m.get("role") == "tool")
    assert "THE CAR NEEDS A NEW BELT" in tool_msg["content"]
    assert "THE CAR NEEDS A NEW BELT" not in str(events[0])
    assert out["memory_notes"] and "doc 4" in out["memory_notes"][0]["text"]


def test_memory_wording_does_not_force_recall_instruction(tmp_path):
    rt = LocalAgentRuntime(FakeMemory(), CFG, data_dir=tmp_path)
    hints = rt._automatic_hints("remember the car")
    system = rt._system_prompt("remember the car", hints)
    low = system.lower()
    assert "automatic hints are suggestions only" in low
    assert "decide yourself whether any memory needs to be opened" in low
    assert "use a neural memory tool before answering" not in low
