"""Lossless prototype/delta layout compression for MicroBrain 0.9.56.

A continuous MBE2 engram can contain many exact repetitions of neural regions.
This module factors those regions into a content-addressed prototype bank plus a
very small temporal layout.  The layout stores only:

* total neural duration,
* silent-gap deltas between active neural regions,
* 256-bit content IDs of the exact neural region prototypes.

No English labels, token IDs, source text, or source token boundaries are stored.
The segmentation used for compression is rediscovered from neural inactivity.
Reconstruction restores the original active-frame and recurrent-route event stream
exactly; therefore this is a storage codec, not a semantic substitute for recall.
"""
from __future__ import annotations

import hashlib

from compact_engram import (
    CompactEngramRecorder,
    _read_uvarint,
    _uvarint,
    discover_regions,
    header,
    iter_routes,
    iter_stored_frames,
    slice_region_blob,
)

MAGIC = b"MBL1"
VERSION = 1
PID_BYTES = 32


def prototype_id(blob: bytes) -> str:
    return hashlib.sha256(bytes(blob)).hexdigest()


def pack_layout(duration_ms: int, placements) -> bytes:
    """Pack placements as (gap_before_ms, prototype_hex_id)."""
    out = bytearray(MAGIC)
    out.append(VERSION)
    out += _uvarint(int(duration_ms))
    vals = [(int(g), str(pid)) for g, pid in placements]
    out += _uvarint(len(vals))
    for gap, pid in vals:
        if gap < 0:
            raise ValueError("negative neural gap")
        raw = bytes.fromhex(pid)
        if len(raw) != PID_BYTES:
            raise ValueError("prototype id must be SHA-256")
        out += _uvarint(gap)
        out += raw
    return bytes(out)


def parse_layout(blob: bytes):
    data = bytes(blob)
    if data[:4] != MAGIC or len(data) < 5 or data[4] != VERSION:
        raise ValueError("not an MBL1 prototype layout")
    pos = 5
    duration, pos = _read_uvarint(data, pos)
    count, pos = _read_uvarint(data, pos)
    placements = []
    for _ in range(count):
        gap, pos = _read_uvarint(data, pos)
        if pos + PID_BYTES > len(data):
            raise ValueError("truncated prototype id")
        pid = data[pos:pos + PID_BYTES].hex()
        pos += PID_BYTES
        placements.append((int(gap), pid))
    if pos != len(data):
        raise ValueError("trailing bytes in MBL1 layout")
    return {"version": VERSION, "duration_ms": int(duration), "placements": placements}


def factorize_engram(blob: bytes):
    """Losslessly factor one continuous MBE2 engram into layout + prototypes.

    Returns (layout_blob, prototypes, regions), where prototypes maps SHA-256 hex
    IDs to standalone compact MBE2 region blobs and regions contains
    (start_ms, end_ms, prototype_id) for validation/indexing.
    """
    h = header(blob)
    bounds = discover_regions(blob)
    protos = {}
    rows = []
    placements = []
    cursor = 0
    for start, end in bounds:
        pblob = slice_region_blob(blob, start, end)
        pid = prototype_id(pblob)
        protos.setdefault(pid, pblob)
        gap = int(start) - int(cursor)
        if gap < 0:
            raise ValueError("overlapping neural regions")
        placements.append((gap, pid))
        rows.append((int(start), int(end), pid))
        cursor = int(end) + 1
    layout = pack_layout(h["duration_ms"], placements)
    return layout, protos, rows


def layout_regions(layout: bytes, resolver):
    """Return (start,end,pid) placements without expanding prototype events."""
    parsed = parse_layout(layout)
    cursor = 0
    rows = []
    for gap, pid in parsed["placements"]:
        start = cursor + int(gap)
        pblob = resolver(pid)
        dur = int(header(pblob)["duration_ms"])
        if dur <= 0:
            raise ValueError("prototype has zero duration")
        end = start + dur - 1
        rows.append((start, end, pid))
        cursor = end + 1
    if cursor > parsed["duration_ms"]:
        raise ValueError("prototype layout exceeds stored duration")
    return rows


def reconstruct_engram(layout: bytes, resolver, compression=6) -> bytes:
    """Reconstruct a continuous MBE2 event stream from a prototype layout."""
    parsed = parse_layout(layout)
    rows = layout_regions(layout, resolver)
    rec = CompactEngramRecorder(origin_t=0)
    for start, _end, pid in rows:
        pblob = resolver(pid)
        for t, active in iter_stored_frames(pblob):
            rec.add_frame(int(start) + int(t), active)
        for s, d, t in iter_routes(pblob):
            rec.add_transmission(int(start) + int(t), int(s), int(d))
    return rec.finish(parsed["duration_ms"], compression=compression)


def neural_event_digest(blob: bytes) -> str:
    """Compression-independent digest of duration, active frames, and routes."""
    import struct
    h = hashlib.sha256()
    hh = header(blob)
    h.update(b"D" + struct.pack("<Q", int(hh["duration_ms"])))
    for t, active in iter_stored_frames(blob):
        h.update(b"F" + struct.pack("<qI", int(t), len(active)))
        for n in active:
            h.update(struct.pack("<I", int(n)))
    for s, d, t in iter_routes(blob):
        h.update(b"R" + struct.pack("<IIq", int(s), int(d), int(t)))
    return h.hexdigest()


def validate_lossless(source: bytes, layout: bytes, resolver):
    rebuilt = reconstruct_engram(layout, resolver)
    return {
        "source_digest": neural_event_digest(source),
        "rebuilt_digest": neural_event_digest(rebuilt),
        "equal": neural_event_digest(source) == neural_event_digest(rebuilt),
        "source_bytes": len(source),
        "layout_bytes": len(layout),
        "rebuilt_bytes": len(rebuilt),
    }
