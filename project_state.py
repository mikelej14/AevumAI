from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone

FORMAT = "LRNR_PROJECT_V2_COMPACT"
LEGACY_FORMAT = "LRNR_PROJECT_V1"


def _memory_index_to_json(index):
    out = {}
    for k, rows in (index or {}).items():
        vals = []
        for a, b, dig in rows:
            if isinstance(dig, str):
                hx = dig
            else:
                hx = bytes(dig).hex()
            vals.append([int(a), int(b), hx])
        out[str(k)] = vals
    return out


def _memory_index_from_json(index):
    out = {}
    for k, rows in (index or {}).items():
        vals = []
        for a, b, hx in rows:
            try:
                dig = bytes.fromhex(hx) if isinstance(hx, str) else bytes(hx)
            except Exception:
                dig = b""
            vals.append((int(a), int(b), dig))
        out[str(k)] = vals
    return out


def _pattern_bank_metadata(bank):
    meta = {}
    blobs = {}
    for pid, value in (bank or {}).items():
        if isinstance(value, (bytes, bytearray, memoryview)):
            member = f"patterns/{pid}.mbe"
            meta[pid] = {"kind": "blob", "member": member}
            blobs[member] = bytes(value)
        elif isinstance(value, dict):
            meta[pid] = value
    return meta, blobs


def _memory_metadata(memories):
    rows = []
    blobs = {}
    for rec in memories or []:
        if rec.get("format") == "compact_continuous_v2" and "engram" in rec:
            member = f"engrams/{int(rec['id']):08d}.mbe"
            x = {k: v for k, v in rec.items() if k != "engram"}
            x["engram_member"] = member
            rows.append(x)
            blobs[member] = rec["engram"]
        else:
            rows.append(rec)
    return rows, blobs


def brain_state(brain, positions, input_neuron, dictionary=None, ui=None, history=None):
    pmeta, _ = _pattern_bank_metadata(getattr(brain, "neural_pattern_bank", {}))
    mm, _ = _memory_metadata(getattr(brain, "neural_memories", []))
    return {
        "format": FORMAT,
        "saved_utc": datetime.now(timezone.utc).isoformat(),
        "brain": {
            "config": vars(brain.c),
            "time_ms": brain.t,
            "positions": {str(k): v for k, v in positions.items()},
            "input_neuron": int(input_neuron),
            "neurons": [vars(n).copy() for n in brain.n],
            "synapses": [
                {"pre": i, "post": j, "weight": w, "delay": brain.delay[i, j]}
                for (i, j), w in sorted(brain.w.items())
            ],
            "decoder_lexicon": getattr(brain, "decoder_lexicon", {}),
            "decoder_route_fingerprints": getattr(brain, "decoder_route_fingerprints", {}),
            "decoder_route_labels": getattr(brain, "decoder_route_labels", {}),
            "neural_pattern_bank": pmeta,
            "neural_memories": mm,
            "neural_memory_index": _memory_index_to_json(getattr(brain, "neural_memory_index", {})),
            "neural_documents": getattr(brain, "neural_documents", []),
            "memory_next_id": int(getattr(brain, "memory_next_id", 0)),
            "document_next_id": int(getattr(brain, "document_next_id", 0)),
        },
        "dictionary": dictionary or {},
        "ui": ui or {},
        "history": history or [],
    }


def _copy_blob_to_zip(dst_zip, member, ref):
    info = zipfile.ZipInfo(member)
    info.compress_type = zipfile.ZIP_STORED  # MBE2 is already internally compressed.
    with dst_zip.open(info, "w", force_zip64=True) as out:
        if isinstance(ref, (bytes, bytearray, memoryview)):
            out.write(bytes(ref))
            return
        if not isinstance(ref, dict):
            raise TypeError(f"unsupported blob reference {type(ref)!r}")
        kind = ref.get("kind")
        if kind == "bytes":
            out.write(bytes(ref["data"]))
        elif kind == "file":
            with open(ref["path"], "rb") as src:
                shutil.copyfileobj(src, out, length=1024 * 1024)
        elif kind == "zip":
            with zipfile.ZipFile(ref["path"], "r") as srcz:
                with srcz.open(ref["member"], "r") as src:
                    shutil.copyfileobj(src, out, length=1024 * 1024)
        else:
            raise ValueError(f"unknown blob reference kind {kind!r}")


def save_project(path, brain, positions, input_neuron, dictionary=None, ui=None, history=None):
    state = brain_state(brain, positions, input_neuron, dictionary, ui, history)
    _, pattern_blobs = _pattern_bank_metadata(getattr(brain, "neural_pattern_bank", {}))
    _, memory_blobs = _memory_metadata(getattr(brain, "neural_memories", []))
    raw = json.dumps(state, separators=(",", ":"), ensure_ascii=False).encode()
    state["sha256"] = hashlib.sha256(raw).hexdigest()

    tmp = path + ".tmp"
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        z.writestr("project.json", json.dumps(state, separators=(",", ":"), ensure_ascii=False))
        for member, blob in pattern_blobs.items():
            _copy_blob_to_zip(z, member, blob)
        for member, ref in memory_blobs.items():
            _copy_blob_to_zip(z, member, ref)
    os.replace(tmp, path)
    return path


def _restore_brain_common(state, path, Brain):
    bd = state["brain"]
    try:
        from microbrain import Config
        cfg = {k: v for k, v in bd.get("config", {}).items() if k in Config.__dataclass_fields__}
        b = Brain(Config(**cfg))
    except Exception:
        b = Brain()
    b.t = int(bd.get("time_ms", 0))
    for i in range(b.c.n):
        b.edges[i] = []
    b.w.clear(); b.delay.clear(); b.pending.clear()
    for x in bd["synapses"]:
        b.set_synapse(x["pre"], x["post"], x["weight"], x["delay"])
    for i, saved in enumerate(bd.get("neurons", [])):
        if i >= len(b.n):
            break
        for k, v in saved.items():
            if hasattr(b.n[i], k):
                setattr(b.n[i], k, v)
    b.decoder_lexicon = bd.get("decoder_lexicon", {}) or {}
    b.decoder_route_fingerprints = bd.get("decoder_route_fingerprints", {}) or {}
    b.decoder_route_labels = bd.get("decoder_route_labels", {}) or {}
    b.memory_next_id = int(bd.get("memory_next_id", len(bd.get("neural_memories", []))))
    b.document_next_id = int(bd.get("document_next_id", len(bd.get("neural_documents", []))))
    b.neural_documents = bd.get("neural_documents", []) or []
    positions = {int(k): v for k, v in bd["positions"].items()}
    return b, positions, int(bd.get("input_neuron", 0))


def load_project(path, Brain):
    apath = os.path.abspath(path)
    with zipfile.ZipFile(apath, "r") as z:
        state = json.loads(z.read("project.json"))
        fmt = state.get("format")
        if fmt not in (FORMAT, LEGACY_FORMAT):
            raise ValueError("Unsupported LRNR project format")
        b, positions, input_neuron = _restore_brain_common(state, apath, Brain)
        bd = state["brain"]

        if fmt == FORMAT:
            bank = {}
            for pid, value in (bd.get("neural_pattern_bank", {}) or {}).items():
                if isinstance(value, dict) and value.get("kind") == "blob":
                    bank[pid] = z.read(value["member"])
                else:
                    bank[pid] = value
            b.neural_pattern_bank = bank

            memories = []
            for rec in bd.get("neural_memories", []) or []:
                if rec.get("format") == "compact_continuous_v2" and rec.get("engram_member"):
                    x = {k: v for k, v in rec.items() if k != "engram_member"}
                    x["engram"] = {"kind": "zip", "path": apath, "member": rec["engram_member"]}
                    memories.append(x)
                else:
                    memories.append(rec)
            b.neural_memories = memories
            b.neural_memory_index = _memory_index_from_json(bd.get("neural_memory_index", {}) or {})
        else:
            # 0.9.54 and older: leave heavy legacy structures intact long enough for
            # NeuralSentenceMemory to migrate them to compact scratch blobs.
            b.neural_pattern_bank = bd.get("neural_pattern_bank", {}) or {}
            b.neural_memories = bd.get("neural_memories", []) or []
            b.neural_memory_index = {}

    return (
        b, positions, input_neuron,
        state.get("dictionary", {}), state.get("ui", {}), state.get("history", []),
    )
