from core.neural_memory_runtime import NeuralMemoryRuntime


def test_timestamped_chat_memory_roundtrip(tmp_path):
    m = NeuralMemoryRuntime(tmp_path / "nm")
    info = m.store_message(
        "THE CAR IS RED", role="user", speaker="Mike", chat_id="chat-1", message_id="msg-1",
        timestamp_utc="2026-09-12T18:00:00+00:00",
    )
    rows = m.search("car", start_utc="2026-09-08T00:00:00+00:00", end_utc="2026-09-13T23:59:59+00:00")
    assert [x["document_id"] for x in rows] == [info["document_id"]]
    assert rows[0]["preview"] == "THE CAR IS RED"
    assert rows[0]["metadata"]["speaker"] == "Mike"
    assert rows[0]["metadata"]["message_id"] == "msg-1"
    doc = next(x for x in m.brain.neural_documents if x["id"] == info["document_id"])
    assert "text" not in doc and "content" not in doc and "sentence" not in doc
    m2 = NeuralMemoryRuntime(tmp_path / "nm")
    assert m2.open_document(info["document_id"])["sentence"] == "THE CAR IS RED"


def test_explicit_recall_reassembles_complete_multichunk_document(tmp_path):
    m = NeuralMemoryRuntime(tmp_path / "nm2", chunk_tokens=12)
    text = "THE DOG IS BLACK THE CAT IS WHITE THE DOG IS FURRY THE CAT IS FAST"
    info = m.store_message(text, role="user", speaker="Mike", chat_id="chat-2", message_id="msg-2",
                           timestamp_utc="2026-09-13T18:00:00+00:00")
    assert info["chunks"] >= 2
    rows = m.search_full_documents("dog", top_k=4)
    hit = next(x for x in rows if x["document_id"] == info["document_id"])
    assert hit["text"] == text
