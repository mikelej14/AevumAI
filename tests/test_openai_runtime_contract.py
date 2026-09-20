from core.openai_runtime import OpenAIChatRuntime


class FakeMemory:
    def search_full_documents(self, *a, **k): return [{"document_id": 7, "text": "HELLO"}]
    def open_document(self, document_id): return {"document_id": document_id, "sentence": "HELLO"}
    def recent(self, **k): return []
    def stats(self): return {"documents": 1}


def test_strict_tool_schemas_and_dispatch():
    tools = OpenAIChatRuntime.tool_definitions()
    assert {x["name"] for x in tools} == {"neural_memory_search", "neural_memory_open", "neural_memory_recent", "neural_memory_stats"}
    for tool in tools:
        assert tool["strict"] is True
        p = tool["parameters"]
        assert p["additionalProperties"] is False
        assert set(p.get("required", [])) == set(p.get("properties", {}))
    rt = OpenAIChatRuntime(FakeMemory())
    assert rt._execute_tool("neural_memory_open", {"document_id": 7})["sentence"] == "HELLO"
    assert rt._execute_tool("neural_memory_stats", {})["documents"] == 1


def test_openai_compact_receipt_does_not_duplicate_recalled_body():
    rt = OpenAIChatRuntime(FakeMemory())
    result = rt._execute_tool("neural_memory_search", {
        "query": "hello", "purpose": "recover an older statement", "start_utc": None,
        "end_utc": None, "speaker": None, "role": None, "top_k": 4,
    })
    event = rt._compact_tool_event("neural_memory_search", {"query": "hello"}, result)
    note = rt._compact_memory_note("neural_memory_search", {"query": "hello", "purpose": "recover an older statement"}, result)
    assert result[0]["text"] == "HELLO"
    assert "HELLO" not in str(event)
    assert "HELLO" not in str(note)
    assert "PURPOSE: recover an older statement" in note["text"]
