import copy
from pathlib import Path

from core.config import DEFAULT_CONFIG
from core.chats import InMemoryChatStore
from core.local_agent_runtime import LocalAgentRuntime
from core.neural_memory_runtime import NeuralMemoryRuntime
from core.runtime_adapters import AevumNeuralScheduler
from ui.neural_minimap import sample_activity_strength


def test_remember_and_recall_emit_real_neural_blob(tmp_path):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["models"]["annotator"]["enabled"] = False
    mem = NeuralMemoryRuntime(tmp_path / "memory")
    chats = InMemoryChatStore()
    local = LocalAgentRuntime(mem, cfg, data_dir=tmp_path)
    sched = AevumNeuralScheduler(cfg, local, mem, chats)
    events = []
    info = sched._archive_with_activity(
        "THE DOG IS BLACK", role="user", speaker="Mike", chat_id="c1", message_id="m1",
        timestamp_utc="2026-09-19T20:00:00+00:00", on_event=lambda n, d: events.append((n, d)),
    )
    remember = [d for n, d in events if n == "neural_activity" and d.get("mode") == "REMEMBER"]
    assert remember and isinstance(remember[0].get("blob"), (bytes, bytearray))
    assert sample_activity_strength(remember[0]["blob"], max_frames=24)

    events.clear()
    sched._blob_activity(lambda n, d: events.append((n, d)), mode="RECALL", document_id=info["document_id"], label="dog")
    recall = [d for n, d in events if n == "neural_activity" and d.get("mode") == "RECALL"]
    assert recall and isinstance(recall[0].get("blob"), (bytes, bytearray))
    assert sample_activity_strength(recall[0]["blob"], max_frames=24)
