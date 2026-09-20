"""Deterministic calibrated single-input experience generator.

Important invariants:
- Uses stimulus(ch)'s calibrated input neuron.  It does NOT inject every glyph
  through simultaneous inputs 0/32/64/96.
- The neural simulator remains on its native 1 ms event clock.
- The returned episode preserves frames, route_events, and causal_events so the
  active 0.5/1/2/4 ms multiscale decoder has the information it needs.
"""
from collections import Counter
import random
from microbrain import Brain, Config
from stimulus import stimulus


def run_calibrated_experience(text, seed=None, sigma=3.0, jitter=2, tail_ms=45):
    rng = random.Random(seed)
    brain = Brain(Config(plasticity=False))
    brain.trace_transmissions = True
    frames = []
    anchors = []

    for ch in str(text):
        start = brain.t + 1
        anchors.append(start)
        spec = stimulus(ch)
        for pulse in spec["pulses"]:
            if seed is None:
                amp = pulse["amp"]
                dt = pulse["dt"]
            else:
                amp = max(0.0, pulse["amp"] + rng.gauss(0.0, sigma))
                dt = pulse["dt"] + rng.randint(-jitter, jitter)
            brain.inject(
                pulse["input_neuron"],
                amp,
                max(start, start + dt),
                origin=pulse["input_neuron"],
            )
        for _ in range(32):
            brain.step()
            active = tuple(sorted(
                e["neuron"] for e in brain.events
                if e["type"] == "spike" and e["t"] == brain.t - 1
            ))
            frames.append((brain.t - start, active))

    if not anchors:
        return {
            "text": str(text), "frames": [], "routes": Counter(),
            "transmissions": 0, "route_events": tuple(), "causal_events": tuple(),
        }

    for _ in range(int(tail_ms)):
        brain.step()
        active = tuple(sorted(
            e["neuron"] for e in brain.events
            if e["type"] == "spike" and e["t"] == brain.t - 1
        ))
        frames.append((brain.t - anchors[-1], active))

    a = anchors[0]
    route_events = tuple(
        (x["source"], x["destination"], x["send_t"] - a)
        for x in brain.transmissions
    )
    causal_events = tuple(
        (x["origin"], x["source"], x["destination"], x["send_t"] - a, x["born"] - a)
        for x in brain.transmissions
    )
    routes = Counter((s, d, int(t) // 10) for s, d, t in route_events)
    return {
        "text": str(text),
        "frames": frames,
        "routes": routes,
        "transmissions": len(brain.transmissions),
        "route_events": route_events,
        "causal_events": causal_events,
    }
