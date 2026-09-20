from microbrain import Brain
from neural_sentence_memory import NeuralSentenceMemory
from continuous_sentence_experience import run_compact_token_experience
from prototype_engram import neural_event_digest


def test_compiled_reuse_matches_direct_brain_event_stream():
    b = Brain(); m = NeuralSentenceMemory(b)
    info = m.store_sentence("THE DOG IS BLACK")
    assert info["compiled_reuse"] is True
    compiled = m._memory_blob(b.neural_memories[0])
    direct = run_compact_token_experience(["THE", "DOG", "IS", "BLACK"])
    assert neural_event_digest(compiled) == neural_event_digest(direct)


def test_pretrained_repeat_is_layout_only_from_first_store():
    b = Brain(); m = NeuralSentenceMemory(b)
    first = m.store_sentence("THE DOG IS BLACK")
    nproto = len(b.neural_prototype_bank)
    second = m.store_sentence("THE DOG IS BLACK")
    assert len(b.neural_prototype_bank) == nproto
    assert first["new_prototype_bytes"] == 0
    assert first["new_prototypes_simulated"] == 0
    assert first["engram_bytes"] == first["layout_bytes"]
    assert second["new_prototype_bytes"] == 0
    assert second["new_prototypes_simulated"] == 0
    assert second["engram_bytes"] == second["layout_bytes"]


def test_unseen_token_is_learned_once_then_layout_only():
    b = Brain(); m = NeuralSentenceMemory(b)
    assert "ZXQJ" not in b.neural_token_prototypes
    first = m.store_sentence("ZXQJ")
    nproto = len(b.neural_prototype_bank)
    second = m.store_sentence("ZXQJ")
    assert first["new_prototypes_simulated"] == 1
    assert first["new_prototype_bytes"] > 0
    assert len(b.neural_prototype_bank) == nproto
    assert second["new_prototypes_simulated"] == 0
    assert second["new_prototype_bytes"] == 0
    assert second["engram_bytes"] == second["layout_bytes"]
    assert second["engram_bytes"] < first["engram_bytes"]
