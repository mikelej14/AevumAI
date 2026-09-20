"""Compact binary neural engrams for MicroBrain 0.9.55.

The 0.9.54 proof stored continuous episodes as large Python/JSON lists.  This
module keeps the same decoder-facing evidence in a compact binary container:

* frame activity: delta-time + 128-neuron bitset
* recurrent transmissions: delta-time + packed source/destination edge id
* each section independently DEFLATE-compressed

No English text, token IDs, or decoder labels are stored in an engram.
The binary form is intended to remain resident in RAM.  Small word regions are
expanded into ordinary episode dictionaries only while they are actively being
decoded.
"""
from __future__ import annotations

import hashlib
import struct
import zlib

MAGIC = b"MBE2"
VERSION = 2
N_NEURONS = 128
FRAME_MASK_BYTES = N_NEURONS // 8
FLAG_SPARSE_FRAMES = 1


def _uvarint(n: int) -> bytes:
    n = int(n)
    if n < 0:
        raise ValueError("uvarint requires non-negative integer")
    out = bytearray()
    while n >= 0x80:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    out.append(n)
    return bytes(out)


def _read_uvarint(buf: bytes, pos: int):
    shift = 0
    value = 0
    while True:
        if pos >= len(buf):
            raise ValueError("truncated varint")
        b = buf[pos]
        pos += 1
        value |= (b & 0x7F) << shift
        if not (b & 0x80):
            return value, pos
        shift += 7
        if shift > 63:
            raise ValueError("varint too large")


def _zigzag(n: int) -> int:
    n = int(n)
    return (n << 1) if n >= 0 else ((-n << 1) - 1)


def _unzigzag(n: int) -> int:
    return -(n // 2) - 1 if (n & 1) else n // 2


def _mask(active) -> bytes:
    bits = 0
    for n in active:
        n = int(n)
        if not (0 <= n < N_NEURONS):
            raise ValueError(f"neuron id out of range: {n}")
        bits |= 1 << n
    return bits.to_bytes(FRAME_MASK_BYTES, "little")


def _active(mask: bytes):
    bits = int.from_bytes(mask, "little")
    return tuple(i for i in range(N_NEURONS) if (bits >> i) & 1)


def _int_time(t):
    v = float(t)
    i = int(round(v))
    if abs(v - i) > 1e-9:
        raise ValueError(f"native route time is not integral: {t!r}")
    return i


def pack_episode(ep, *, sparse_frames=False, compression=6) -> bytes:
    """Pack an ordinary decoder episode into an MBE2 blob.

    `sparse_frames=True` is optimal for continuous memories because duration is
    known and silent millisecond frames can be reconstructed on demand.  Lexicon
    exemplars use full-frame mode so their historical frame sequence is preserved
    exactly, including local-time resets between characters.
    """
    frames = ep.get("frames", ())
    routes = ep.get("route_events", ())
    duration = int(ep.get("duration_ms", 0))
    if not duration and frames:
        try:
            duration = max(int(t) for t, _ in frames) + 1
        except Exception:
            duration = len(frames)

    fb = bytearray()
    prev_t = 0
    frame_count = 0
    for t, active in frames:
        if sparse_frames and not active:
            continue
        ti = int(t)
        fb += _uvarint(_zigzag(ti - prev_t))
        fb += _mask(active)
        prev_t = ti
        frame_count += 1

    rb = bytearray()
    prev_t = 0
    route_count = 0
    for s, d, t in routes:
        ti = _int_time(t)
        rb += _uvarint(_zigzag(ti - prev_t))
        rb += _uvarint(int(s) * N_NEURONS + int(d))
        prev_t = ti
        route_count += 1

    fc = zlib.compress(bytes(fb), int(compression))
    rc = zlib.compress(bytes(rb), int(compression))
    flags = FLAG_SPARSE_FRAMES if sparse_frames else 0

    out = bytearray(MAGIC)
    out += bytes((VERSION, flags))
    for n in (duration, frame_count, route_count, len(fc), len(rc)):
        out += _uvarint(n)
    out += fc
    out += rc
    return bytes(out)


def header(blob: bytes):
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise TypeError("compact engram must be bytes-like")
    blob = bytes(blob)
    if blob[:4] != MAGIC:
        raise ValueError("not an MBE2 compact engram")
    if len(blob) < 6 or blob[4] != VERSION:
        raise ValueError(f"unsupported compact engram version: {blob[4] if len(blob)>4 else None}")
    flags = blob[5]
    pos = 6
    vals = []
    for _ in range(5):
        x, pos = _read_uvarint(blob, pos)
        vals.append(x)
    duration, frame_count, route_count, frame_len, route_len = vals
    frame_off = pos
    route_off = frame_off + frame_len
    if route_off + route_len != len(blob):
        raise ValueError("compact engram length mismatch")
    return {
        "version": VERSION,
        "flags": flags,
        "sparse_frames": bool(flags & FLAG_SPARSE_FRAMES),
        "duration_ms": duration,
        "frame_count": frame_count,
        "route_count": route_count,
        "frame_offset": frame_off,
        "frame_length": frame_len,
        "route_offset": route_off,
        "route_length": route_len,
        "bytes": len(blob),
    }


def _sections(blob: bytes):
    h = header(blob)
    fb = zlib.decompress(blob[h["frame_offset"]:h["frame_offset"] + h["frame_length"]])
    rb = zlib.decompress(blob[h["route_offset"]:h["route_offset"] + h["route_length"]])
    return h, fb, rb


def iter_stored_frames(blob: bytes):
    h = header(blob)
    raw = zlib.decompress(blob[h["frame_offset"]:h["frame_offset"] + h["frame_length"]])
    pos = 0
    t = 0
    for _ in range(h["frame_count"]):
        z, pos = _read_uvarint(raw, pos)
        t += _unzigzag(z)
        if pos + FRAME_MASK_BYTES > len(raw):
            raise ValueError("truncated frame stream")
        m = raw[pos:pos + FRAME_MASK_BYTES]
        pos += FRAME_MASK_BYTES
        yield t, _active(m)
    if pos != len(raw):
        raise ValueError("trailing bytes in frame stream")


def iter_routes(blob: bytes):
    h = header(blob)
    raw = zlib.decompress(blob[h["route_offset"]:h["route_offset"] + h["route_length"]])
    pos = 0
    t = 0
    for _ in range(h["route_count"]):
        z, pos = _read_uvarint(raw, pos)
        t += _unzigzag(z)
        edge, pos = _read_uvarint(raw, pos)
        yield edge // N_NEURONS, edge % N_NEURONS, t
    if pos != len(raw):
        raise ValueError("trailing bytes in route stream")


def active_times(blob: bytes):
    return [int(t) for t, active in iter_stored_frames(blob) if active]


def discover_regions(blob: bytes, min_silence_ms=32):
    times = active_times(blob)
    if not times:
        return []
    groups = [[times[0], times[0]]]
    for t in times[1:]:
        if t - groups[-1][1] > int(min_silence_ms):
            groups.append([t, t])
        else:
            groups[-1][1] = t
    return [(int(a), int(b)) for a, b in groups]


def route_fingerprint_region(blob: bytes, start_t=None, end_t=None):
    """Exact translation-invariant route fingerprint without expanding tuples."""
    first = None
    h = hashlib.sha256()
    found = False
    for s, d, t in iter_routes(blob):
        if start_t is not None and t < int(start_t):
            continue
        if end_t is not None and t > int(end_t):
            # continuous episode route times are monotone
            break
        if first is None:
            first = float(t)
        h.update(struct.pack("<IId", int(s), int(d), float(t) - first))
        found = True
    return h.hexdigest() if found else None


def route_signature_region(blob: bytes, start_t=None, end_t=None, nbytes=16):
    fp = route_fingerprint_region(blob, start_t, end_t)
    if not fp:
        return None
    return bytes.fromhex(fp)[:int(nbytes)]


def extract_region(blob: bytes, start_t: int, end_t: int):
    """Expand only one active region into the decoder's ordinary episode shape."""
    s = int(start_t)
    e = int(end_t)
    if e < s:
        raise ValueError("end before start")
    h = header(blob)
    sparse = h["sparse_frames"]

    if sparse:
        amap = {int(t): tuple(active) for t, active in iter_stored_frames(blob) if s <= int(t) <= e}
        frames = [(t - s, amap.get(t, ())) for t in range(s, e + 1)]
    else:
        frames = [(int(t) - s, tuple(active)) for t, active in iter_stored_frames(blob) if s <= int(t) <= e]

    routes = []
    for a, b, t in iter_routes(blob):
        if t < s:
            continue
        if t > e:
            break
        routes.append((int(a), int(b), float(t - s)))
    return {"frames": frames, "route_events": tuple(routes), "transmissions": len(routes)}


def unpack_episode(blob: bytes):
    """Full expansion for compatibility/debugging. Avoid for large memories."""
    h = header(blob)
    if h["sparse_frames"]:
        amap = {int(t): tuple(active) for t, active in iter_stored_frames(blob)}
        frames = [(t, amap.get(t, ())) for t in range(h["duration_ms"])]
    else:
        frames = list(iter_stored_frames(blob))
    routes = tuple((s, d, float(t)) for s, d, t in iter_routes(blob))
    return {
        "frames": frames,
        "route_events": routes,
        "transmissions": len(routes),
        "duration_ms": int(h["duration_ms"]),
        "format": "LRNR_COMPACT_EPISODE_V2",
    }


class CompactEngramRecorder:
    """Streaming recorder used by the simulator to avoid giant trace dictionaries."""
    def __init__(self, origin_t=0):
        self.origin_t = int(origin_t)
        self._routes = bytearray()
        self._frames = bytearray()
        self._prev_route_t = 0
        self._prev_frame_t = 0
        self.route_count = 0
        self.frame_count = 0

    def add_transmission(self, send_t, source, destination, origin=None, born=None):
        t = int(send_t) - self.origin_t
        self._routes += _uvarint(_zigzag(t - self._prev_route_t))
        self._routes += _uvarint(int(source) * N_NEURONS + int(destination))
        self._prev_route_t = t
        self.route_count += 1

    def add_frame(self, t, active):
        if not active:
            return
        t = int(t) - self.origin_t
        self._frames += _uvarint(_zigzag(t - self._prev_frame_t))
        self._frames += _mask(active)
        self._prev_frame_t = t
        self.frame_count += 1

    def finish(self, duration_ms, compression=6):
        fc = zlib.compress(bytes(self._frames), int(compression))
        rc = zlib.compress(bytes(self._routes), int(compression))
        out = bytearray(MAGIC)
        out += bytes((VERSION, FLAG_SPARSE_FRAMES))
        for n in (int(duration_ms), self.frame_count, self.route_count, len(fc), len(rc)):
            out += _uvarint(n)
        out += fc
        out += rc
        return bytes(out)


def route_signatures_for_regions(blob: bytes, regions, nbytes=16):
    """Compute exact route signatures for sorted non-overlapping regions in one pass."""
    regs = [(int(a), int(b)) for a, b in regions]
    out = [None] * len(regs)
    if not regs:
        return out
    idx = 0
    hasher = None
    first = None
    for s, d, t in iter_routes(blob):
        while idx < len(regs) and t > regs[idx][1]:
            if hasher is not None:
                out[idx] = hasher.digest()[:int(nbytes)]
            idx += 1
            hasher = None
            first = None
        if idx >= len(regs):
            break
        a, b = regs[idx]
        if t < a:
            continue
        if hasher is None:
            hasher = hashlib.sha256()
            first = float(t)
        hasher.update(struct.pack("<IId", int(s), int(d), float(t) - first))
    if idx < len(regs) and hasher is not None:
        out[idx] = hasher.digest()[:int(nbytes)]
        idx += 1
    return out


def slice_region_blob(blob: bytes, start_t: int, end_t: int, compression=6):
    """Create a compact standalone neural pattern for one region."""
    ep = extract_region(blob, start_t, end_t)
    ep["duration_ms"] = int(end_t) - int(start_t) + 1
    return pack_episode(ep, sparse_frames=True, compression=compression)
