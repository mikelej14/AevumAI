from compact_engram import pack_episode
from ui.neural_minimap import sample_activity_strength


def test_minimap_strength_comes_from_real_firing_and_route_density():
    # Neuron 3 fires and carries much more recurrent traffic than neuron 9.
    frames = [(0, (3, 9)), (1, (3,)), (2, (3,)), (3, (9,))]
    routes = []
    for t in range(4):
        for _ in range(12):
            routes.append((3, 4, float(t)))
        routes.append((9, 10, float(t)))
    blob = pack_episode({"frames": frames, "route_events": routes}, sparse_frames=False)
    rows = sample_activity_strength(blob, max_frames=24)
    assert rows
    strongest3 = max(row[1][3] for row in rows)
    strongest9 = max(row[1][9] for row in rows)
    assert 0.0 < strongest9 < strongest3 <= 1.0
    assert all(len(row[1]) == 128 for row in rows)


def test_minimap_is_not_binary_color_data():
    frames = [(0, (1, 2, 3))]
    routes = [(1, 4, 0.0)] * 1 + [(2, 5, 0.0)] * 4 + [(3, 6, 0.0)] * 12
    blob = pack_episode({"frames": frames, "route_events": routes}, sparse_frames=False)
    vals = sample_activity_strength(blob, max_frames=24)[0][1]
    active_levels = sorted({round(vals[i], 3) for i in (1, 2, 3)})
    assert len(active_levels) == 3
    assert active_levels[0] < active_levels[1] < active_levels[2]


def test_large_engram_hud_does_not_walk_route_stream(monkeypatch):
    # The HUD must stay bounded even when a stored memory has huge recurrent traffic.
    frames = [(t, ((t * 7) % 128,)) for t in range(64)]
    routes = []
    for i in range(12001):
        t = float(i // 200)
        routes.append((i % 128, (i * 3 + 1) % 128, t))
    blob = pack_episode({"frames": frames, "route_events": routes}, sparse_frames=True)

    import ui.neural_minimap as nm
    def forbidden(_blob):
        raise AssertionError("large HUD sample should not iterate recurrent route stream")
    monkeypatch.setattr(nm, "iter_routes", forbidden)
    rows = nm.sample_activity_strength(blob, max_frames=36)
    assert rows
    assert len(rows) <= 36
    assert any(max(row[1]) > 0 for row in rows)
