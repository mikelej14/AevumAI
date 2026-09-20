from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compact_engram import active_times
from build_pretrained_vocabulary import load_frequency_words, internal_silences

PACK_MAGIC = b"AVNP1"
PACK_VERSION = 1
EXTRAS = [str(i) for i in range(10)] + list(".,!?;:-()[]{}%/\\\"'")


def read_blob(pack: Path, meta):
    with pack.open("rb") as f:
        f.seek(int(meta["offset"]))
        b = f.read(int(meta["length"]))
    if len(b) != int(meta["length"]):
        raise ValueError("truncated shard")
    return b


def load_pack_dir(sd: Path):
    m = json.loads((sd / "manifest.json").read_text())
    pack = sd / m["pack_file"]
    rows = {token: (dict(meta), read_blob(pack, meta)) for token, meta in m["entries"].items()}
    quarantine = []
    qpath = sd / "quarantine.json"
    if qpath.exists():
        try:
            quarantine = json.loads(qpath.read_text())
        except Exception:
            quarantine = []
    return rows, quarantine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--source", required=True, help="frequency source used to rank/backfill active words")
    ap.add_argument("--target-words", type=int, default=10000)
    ap.add_argument("--supplement", action="append", default=[], help="optional curated vocabulary pack dir; may repeat")
    ap.add_argument("shards", nargs="+")
    a = ap.parse_args()

    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    candidates = {}
    shard_quarantine = []
    for sd in map(Path, a.shards):
        rows, q = load_pack_dir(sd)
        shard_quarantine.extend(q)
        for token, pair in rows.items():
            candidates.setdefault(token, pair)

    supplements = {}
    for sd in map(Path, a.supplement):
        rows, q = load_pack_dir(sd)
        shard_quarantine.extend(q)
        for token, pair in rows.items():
            supplements.setdefault(token, pair)

    freq = load_frequency_words(Path(a.source), max(int(a.target_words) + 5000, 30000))
    rank = {word: (i + 1, count) for i, (count, word) in enumerate(freq)}

    quarantine = {}
    valid = {}
    for token, (meta, blob) in {**candidates, **supplements}.items():
        gaps = internal_silences(active_times(blob))
        if gaps:
            quarantine[token] = {
                "token": token,
                "rank": rank.get(token, (meta.get("rank", 0), 0))[0],
                "count": rank.get(token, (0, meta.get("count", 0)))[1],
                "reason": "internal_silence",
                "internal_silences": gaps,
                "duration_ms": int(meta.get("duration_ms", 0)),
                "prototype_id": str(meta["prototype_id"]),
            }
        else:
            valid[token] = (meta, blob)

    for q in shard_quarantine:
        tok = str(q.get("token") or "")
        if tok and tok not in quarantine:
            x = dict(q)
            x["reason"] = str(x.get("reason") or x.get("quarantine") or x.get("error") or "deferred")
            x.pop("quarantine", None)
            quarantine[tok] = x

    selected_words = []
    for count, token in freq:
        if token in valid:
            selected_words.append(token)
            if len(selected_words) >= int(a.target_words):
                break
    if len(selected_words) < int(a.target_words):
        missing = int(a.target_words) - len(selected_words)
        raise SystemExit(f"need {missing} more valid frequency-ranked prototypes; build additional tail shards")

    selected = list(selected_words)
    for tok in EXTRAS:
        if tok in valid and tok not in selected:
            selected.append(tok)
    supplement_added = []
    for tok in supplements:
        if tok in valid and tok not in selected:
            selected.append(tok)
            supplement_added.append(tok)

    blobs = {}
    rows = {}
    for token in selected:
        meta, blob = valid[token]
        pid = str(meta["prototype_id"])
        blobs.setdefault(pid, blob)
        rows[token] = dict(meta)
        if token in rank:
            rows[token]["rank"] = int(rank[token][0])
            rows[token]["count"] = int(rank[token][1])
        if token in supplement_added:
            rows[token]["supplement"] = True

    packout = out / "vocabulary.avnp"
    offsets = {}
    with packout.open("wb") as f:
        f.write(PACK_MAGIC)
        f.write(struct.pack("<I", PACK_VERSION))
        for pid in sorted(blobs):
            off = f.tell()
            b = blobs[pid]
            f.write(b)
            offsets[pid] = (off, len(b))
    for token, meta in rows.items():
        off, n = offsets[str(meta["prototype_id"])]
        meta["offset"] = off
        meta["length"] = n

    qrows = sorted(quarantine.values(), key=lambda x: (int(x.get("rank") or 10**9), str(x.get("token") or "")))
    manifest = {
        "format": "AEVUM_NEURAL_VOCAB_V1",
        "version": 1,
        "word_target": int(a.target_words),
        "entry_count": len(rows),
        "word_entry_count": len(selected_words),
        "extra_entry_count": len([x for x in EXTRAS if x in rows]),
        "supplement_entry_count": len(supplement_added),
        "unique_prototypes": len(blobs),
        "quarantine_count": len(qrows),
        "pack_file": packout.name,
        "pack_sha256": hashlib.sha256(packout.read_bytes()).hexdigest(),
        "entries": rows,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
    (out / "tokens.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (out / "quarantine.json").write_text(json.dumps(qrows, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("word_entry_count", "supplement_entry_count", "entry_count", "unique_prototypes", "quarantine_count", "pack_sha256")}, indent=2))
    print("pack_bytes", packout.stat().st_size)
    if qrows:
        print("quarantined", ", ".join(str(x.get("token")) for x in qrows[:30]))

if __name__ == "__main__":
    main()
