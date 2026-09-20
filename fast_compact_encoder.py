"""Exact optimized compact encoder for deterministic/non-plastic MicroBrain.

This is an execution optimization only. It preserves the same neuron integration,
firing order, recurrent route multiplicity, cycle prevention, timing, frame stream,
and transmission stream as ``run_compact_token_experience``.

The ordinary Brain Route object carries route IDs, origins, full path tuples, and
independence bookkeeping needed by the research UI. MBE2 compact memory stores only
firing frames and recurrent source/destination/time transmissions. In this encoder,
a route's full path tuple is therefore represented by the exactly equivalent
(path_length, 128-bit visited-neuron mask). Analysis-only route bookkeeping is not
computed. The resulting MBE2 neural-event stream is regression-tested against the
canonical Brain implementation.
"""
from __future__ import annotations

from collections import defaultdict

from compact_engram import CompactEngramRecorder
from microbrain import Brain, Config
from stimulus import stimulus
from continuous_sentence_experience import CHAR_SLOT_MS, WORD_GAP_MS, TAIL_MS

_TEMPLATE = Brain(Config(plasticity=False))
_EDGES = tuple(tuple(x) for x in _TEMPLATE.edges)
_WEIGHTS = dict(_TEMPLATE.w)
_DELAYS = dict(_TEMPLATE.delay)
_N = int(_TEMPLATE.c.n)
_REST = float(_TEMPLATE.c.rest)
_RESET = float(_TEMPLATE.c.reset)
_THRESHOLD = float(_TEMPLATE.c.threshold)
_TAU = float(_TEMPLATE.c.tau)
_REFRACTORY = int(_TEMPLATE.c.refractory)
_ROUTE_TTL = int(_TEMPLATE.c.route_ttl)


def run_fast_compact_token_experience(tokens, *, char_slot_ms=CHAR_SLOT_MS,
                                      word_gap_ms=WORD_GAP_MS,
                                      tail_ms=TAIL_MS):
    """Return an MBE2 engram exactly equivalent to deterministic canonical ingest.

    This function intentionally supports only the fixed, deterministic,
    non-plastic path (the path used for the shipped vocabulary and normal compiled
    prototype learning). No jitter/seed/plasticity mode is accepted here.
    """
    toks = [str(x) for x in tokens if str(x)]
    if not toks:
        raise ValueError("Nothing to encode")

    v = [_REST] * _N
    refractory = [0] * _N
    pending = defaultdict(list)
    now = 0
    recorder = CompactEngramRecorder(origin_t=0)

    def inject(neuron, amp, at):
        i = int(neuron)
        # Route state needed for propagation is exactly path length + membership.
        pending[int(at)].append((i, float(amp), 1, 1 << i))

    def step():
        nonlocal now
        current = [0.0] * _N
        routes = [None] * _N
        for i, amp, path_len, visited in pending.pop(now, ()):
            current[i] += amp
            bucket = routes[i]
            if bucket is None:
                routes[i] = [(path_len, visited)]
            else:
                bucket.append((path_len, visited))

        fired = []
        for i in range(_N):
            if refractory[i]:
                refractory[i] -= 1
                v[i] = _RESET
                continue
            v[i] += (_REST - v[i]) / _TAU + current[i]
            if v[i] >= _THRESHOLD:
                fired.append(i)

        recorder.add_frame(now, fired)
        for i in fired:
            v[i] = _RESET
            refractory[i] = _REFRACTORY
            incoming = routes[i]
            if not incoming:
                # Canonical fallback. Under the current dynamics this should not be
                # reached during compact stimulus encoding, but preserving it keeps
                # the transition semantics complete.
                incoming = [(1, 1 << i)]
            for j in _EDGES[i]:
                bit = 1 << j
                weight = _WEIGHTS[i, j]
                delay = _DELAYS[i, j]
                for path_len, visited in incoming:
                    if path_len >= _ROUTE_TTL or (visited & bit):
                        continue
                    pending[now + delay].append(
                        (j, 10.0 * weight, path_len + 1, visited | bit)
                    )
                    recorder.add_transmission(now, i, j)
        now += 1

    for ti, token in enumerate(toks):
        if ti:
            for _ in range(int(word_gap_ms)):
                step()
        for ch in token.upper():
            start = now + 1
            spec = stimulus(ch)
            for pulse in spec["pulses"]:
                inject(
                    pulse["input_neuron"], pulse["amp"],
                    max(start, start + int(pulse["dt"])),
                )
            for _ in range(int(char_slot_ms)):
                step()

    for _ in range(int(tail_ms)):
        step()
    return recorder.finish(now)
