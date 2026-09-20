"""Continuous sentence experience for MicroBrain 0.9.54.

A sentence is written through one Brain instance. Characters inside a word use the
normal calibrated glyph stimulus. Whitespace is represented by a neural quiet gap;
no token list, source sentence, or word-boundary table is returned or persisted.

The decoder later rediscovers word regions from sustained inactivity in the stored
neural trace. This is intentionally a baseline for continuous episodic memory: the
word gap is generous so boundary discovery itself can be validated before reducing
that cue in later work.
"""
from __future__ import annotations

from microbrain import Brain, Config
from stimulus import stimulus

CHAR_SLOT_MS = 32
WORD_GAP_MS = 96
TAIL_MS = 64


def _active_at(brain, t):
    return tuple(sorted(
        e["neuron"] for e in brain.events
        if e.get("type") == "spike" and e.get("t") == t
    ))


def run_continuous_sentence_experience(text, seed=None, sigma=3.0, jitter=2,
                                       char_slot_ms=CHAR_SLOT_MS,
                                       word_gap_ms=WORD_GAP_MS,
                                       tail_ms=TAIL_MS):
    """Encode text into one continuous neural episode.

    `seed=None` is deterministic/calibrated. With a seed, the same amplitude/timing
    perturbation model used by calibrated_experience is applied to each pulse.

    The returned mapping contains neural evidence only. It deliberately does not
    contain the input text, tokens, token count, or word boundaries.
    """
    import random
    import re

    # Text is consumed by the encoder, but is never copied into the returned engram.
    words = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", str(text))
    if not words:
        raise ValueError("Nothing to encode")

    rng = random.Random(seed)
    brain = Brain(Config(plasticity=False))
    brain.trace_transmissions = True
    frames = []
    origin_t = brain.t

    def step_record(n):
        for _ in range(int(n)):
            brain.step()
            frames.append((brain.t - 1 - origin_t, _active_at(brain, brain.t - 1)))

    for wi, word in enumerate(words):
        if wi:
            # A real interval in the neural stream, not persisted boundary metadata.
            step_record(word_gap_ms)
        for ch in word.upper():
            start = brain.t + 1
            spec = stimulus(ch)
            for pulse in spec["pulses"]:
                if seed is None:
                    amp = pulse["amp"]
                    dt = pulse["dt"]
                else:
                    amp = max(0.0, pulse["amp"] + rng.gauss(0.0, sigma))
                    dt = pulse["dt"] + rng.randint(-jitter, jitter)
                brain.inject(
                    pulse["input_neuron"], amp,
                    max(start, start + dt), origin=pulse["input_neuron"]
                )
            step_record(char_slot_ms)

    step_record(tail_ms)

    route_events = tuple(
        (int(x["source"]), int(x["destination"]), float(x["send_t"] - origin_t))
        for x in brain.transmissions
    )
    causal_events = tuple(
        (int(x["origin"]), int(x["source"]), int(x["destination"]),
         float(x["send_t"] - origin_t), float(x["born"] - origin_t))
        for x in brain.transmissions
    )
    return {
        "format": "LRNR_CONTINUOUS_SENTENCE_V1",
        "frames": frames,
        "route_events": route_events,
        "causal_events": causal_events,
        "transmissions": len(route_events),
        "duration_ms": int(brain.t - origin_t),
        "encoding": {
            "char_slot_ms": int(char_slot_ms),
            "word_gap_ms": int(word_gap_ms),
            "tail_ms": int(tail_ms),
        },
    }


def run_compact_token_experience(tokens, seed=None, sigma=3.0, jitter=2,
                                  char_slot_ms=CHAR_SLOT_MS,
                                  word_gap_ms=WORD_GAP_MS,
                                  tail_ms=TAIL_MS):
    """Encode already-tokenized text directly into an MBE2 compact engram.

    Unlike the 0.9.54 proof path this does not accumulate Brain.events,
    Brain.transmissions, per-route dictionaries, or dense frame lists.  Recurrent
    transmissions are delta-coded as they happen and only active millisecond
    frames are retained.  `tokens` are consumed by the encoder but are never
    written into the returned neural engram.
    """
    import random
    from compact_engram import CompactEngramRecorder

    toks = [str(x) for x in tokens if str(x)]
    if not toks:
        raise ValueError("Nothing to encode")
    rng = random.Random(seed)
    brain = Brain(Config(plasticity=False))
    brain.capture_events = False
    origin_t = brain.t
    recorder = CompactEngramRecorder(origin_t=origin_t)
    brain.transmission_recorder = recorder

    def step_record(n):
        for _ in range(int(n)):
            brain.step()
            recorder.add_frame(brain.t - 1, brain.last_fired)

    for ti, token in enumerate(toks):
        if ti:
            step_record(word_gap_ms)
        for ch in token.upper():
            start = brain.t + 1
            spec = stimulus(ch)
            for pulse in spec["pulses"]:
                if seed is None:
                    amp = pulse["amp"]
                    dt = pulse["dt"]
                else:
                    amp = max(0.0, pulse["amp"] + rng.gauss(0.0, sigma))
                    dt = pulse["dt"] + rng.randint(-jitter, jitter)
                brain.inject(
                    pulse["input_neuron"], amp,
                    max(start, start + dt), origin=pulse["input_neuron"]
                )
            step_record(char_slot_ms)

    step_record(tail_ms)
    blob = recorder.finish(brain.t - origin_t)
    brain.transmission_recorder = None
    return blob
