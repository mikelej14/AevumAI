"""Build one shard of the shipped frequency-ranked neural prototype vocabulary.

Developer/offline utility. The output contains exact MBE2 neural prototypes, not
source-text memories. Frequency data chooses labels only.

Words whose *single-token* neural trace contains a >=32 ms internal silence are
quarantined rather than inserted into the active vocabulary. They are preserved
in quarantine.json for later decoder-format analysis and never block the shard.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compact_engram import active_times, header, route_fingerprint_region, slice_region_blob
from fast_compact_encoder import run_fast_compact_token_experience
from prototype_engram import prototype_id

PACK_MAGIC = b"AVNP1"
PACK_VERSION = 1
INTERNAL_SILENCE_MS = 32


def load_frequency_words(path: Path, limit: int):
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        try:
            word, count = line.rsplit(None, 1)
            count = int(count)
        except Exception:
            continue
        if re.fullmatch(r"[A-Za-z]+(?:'[A-Za-z]+)*", word):
            rows.append((count, word.upper()))
    rows.sort(key=lambda x: (-x[0], x[1]))
    seen = set()
    out = []
    for count, word in rows:
        if word in seen:
            continue
        seen.add(word)
        out.append((count, word))
        if len(out) >= int(limit):
            break
    return out


def internal_silences(times, threshold=INTERNAL_SILENCE_MS):
    ts = [int(x) for x in times]
    out = []
    for a, b in zip(ts, ts[1:]):
        quiet = int(b) - int(a) - 1
        if quiet >= int(threshold):
            out.append({"last_active_ms": int(a), "next_active_ms": int(b), "silence_ms": quiet})
    return out


def build_one(item):
    rank, count, token = item
    blob = run_fast_compact_token_experience([token])
    times = active_times(blob)
    if not times:
        return {"rank": rank, "count": count, "token": token, "error": "no_activity"}
    start, end = int(times[0]), int(times[-1])
    pblob = slice_region_blob(blob, start, end)
    gaps = internal_silences(active_times(pblob))
    if gaps:
        return {
            "rank": int(rank), "count": int(count), "token": token,
            "quarantine": "internal_silence", "internal_silences": gaps,
            "duration_ms": int(header(pblob)["duration_ms"]),
            "prototype_id": prototype_id(pblob),
        }
    route = route_fingerprint_region(pblob)
    return {
        "rank": int(rank), "count": int(count), "token": token,
        "pid": prototype_id(pblob), "lead_ms": int(start),
        "route": route or "", "blob": pblob,
        "duration_ms": int(header(pblob)["duration_ms"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--start", type=int, default=0, help="zero-based rank offset for sharded builds")
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    args = ap.parse_args()

    source = Path(args.source)
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    all_words = load_frequency_words(source, int(args.start) + int(args.limit))
    selected = all_words[int(args.start):int(args.start) + int(args.limit)]
    extras = [str(i) for i in range(10)] + list(".,!?;:-()[]{}%/\\\"'")
    known = {w for _, w in selected}
    work = [(int(args.start) + i + 1, c, w) for i, (c, w) in enumerate(selected)]
    if int(args.start) == 0:
        for tok in extras:
            if tok not in known:
                work.append((0, 0, tok))

    started = time.time()
    results = []
    quarantine = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        for i, row in enumerate(pool.map(build_one, work, chunksize=1), 1):
            if row.get("quarantine"):
                quarantine.append(row)
                print(f"QUARANTINE {row['token']}: {row['quarantine']} {row.get('internal_silences')}", flush=True)
            elif "error" in row:
                quarantine.append(row)
                print(f"SKIP {row['token']}: {row['error']}", flush=True)
            else:
                results.append(row)
            if i % 100 == 0 or i == len(work):
                print(f"built {i}/{len(work)} valid={len(results)} quarantine={len(quarantine)} elapsed={time.time()-started:.1f}s", flush=True)

    unique = {}
    for row in results:
        unique.setdefault(row["pid"], row["blob"])

    pack_path = outdir / "vocabulary.avnp"
    offsets = {}
    with pack_path.open("wb") as f:
        f.write(PACK_MAGIC)
        f.write(struct.pack("<I", PACK_VERSION))
        for pid in sorted(unique):
            blob = unique[pid]
            off = f.tell()
            f.write(blob)
            offsets[pid] = (off, len(blob))

    entries = {}
    for row in results:
        off, n = offsets[row["pid"]]
        entries[row["token"]] = {
            "prototype_id": row["pid"], "lead_ms": row["lead_ms"],
            "route": row["route"], "offset": off, "length": n,
            "duration_ms": row["duration_ms"], "rank": row["rank"], "count": row["count"],
        }

    manifest = {
        "format": "AEVUM_NEURAL_VOCAB_V1",
        "version": 1,
        "word_target": int(args.limit),
        "rank_start": int(args.start),
        "entry_count": len(entries),
        "unique_prototypes": len(unique),
        "quarantine_count": len(quarantine),
        "pack_file": pack_path.name,
        "pack_sha256": hashlib.sha256(pack_path.read_bytes()).hexdigest(),
        "entries": entries,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
    (outdir / "tokens.txt").write_text("\n".join(entries) + "\n", encoding="utf-8")
    (outdir / "quarantine.json").write_text(json.dumps(quarantine, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("entry_count", "unique_prototypes", "quarantine_count", "pack_sha256")}, indent=2))
    print(f"pack_bytes={pack_path.stat().st_size} manifest_bytes={(outdir/'manifest.json').stat().st_size}")

if __name__ == "__main__":
    main()
