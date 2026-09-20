from core.chats import ChatStore


def test_chat_transcript_is_session_layer(tmp_path):
    s = ChatStore(tmp_path / "chats.json")
    cid = s.active_chat_id()
    mid = s.add_message(cid, "user", "hello", name="Mike")
    s.add_message(cid, "assistant", "hi", name="Ava")
    assert [x["content"] for x in s.messages(cid)] == ["hello", "hi"]
    assert s.messages(cid)[0]["id"] == mid
    s.delete_chat(cid)
    assert s.active_chat_id()
