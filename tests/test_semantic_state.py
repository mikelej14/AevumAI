from core.semantic_state import SemanticState


def test_semantic_state_is_bounded_and_persistent(tmp_path):
    p = tmp_path / "state.json"
    s = SemanticState(p)
    out = s.apply({
        "positivity": 0.1, "negativity": 0.9, "hostility": 0.8, "praise": 0.0,
        "urgency": 0.7, "salience": 0.9, "correction": 0.8, "uncertainty": 0.2,
        "novelty": 0.4, "context": "user is correcting a memory error", "topics": ["memory"],
    })
    assert -1 <= out["valence"] <= 1
    assert 0 <= out["frustration"] <= 1
    assert out["last_topics"] == ["memory"]
    s2 = SemanticState(p)
    assert s2.snapshot()["last_context"] == "user is correcting a memory error"
