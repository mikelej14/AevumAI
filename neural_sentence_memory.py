"""Scalable neural episodic memory for MicroBrain 0.9.56.

0.9.56 keeps the continuous-neural-memory semantics while adding lossless
shared-prototype/delta storage on top of the 0.9.55 compact representation:

* New memories are compact MBE2 binary engrams, never giant Python route lists.
* Engrams are spooled to disk immediately; only compact neural indexes stay hot.
* The recall index stores neural route digests + offsets, never source sentences.
* Decoder labels live in the separate neural-language dictionary.
* Bulk text is chunked into bounded continuous neural episodes.
* During supervised ingest, the already-generated continuous neural segments teach
  new decoder vocabulary; unknown words are no longer simulated a second time.

The active fuzzy decoder remains the same 0.5/1/2/4 ms multiscale decoder. Exact
neural-route recall is intentionally the fast path; fuzzy expansion is paid only
when a pattern has not already been learned.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import tempfile
import zipfile

from calibrated_experience import run_calibrated_experience
from continuous_sentence_experience import run_compact_token_experience, CHAR_SLOT_MS, WORD_GAP_MS, TAIL_MS
from fast_compact_encoder import run_fast_compact_token_experience
from current_best_decoder import CurrentDecoder, SCALES, score
from compact_engram import (
    MAGIC,
    active_times,
    discover_regions,
    extract_region as compact_extract_region,
    header as compact_header,
    pack_episode,
    route_fingerprint_region,
    route_signatures_for_regions,
    unpack_episode,
    slice_region_blob,
)
from prototype_engram import (
    factorize_engram,
    layout_regions,
    pack_layout,
    neural_event_digest,
    parse_layout,
    reconstruct_engram,
    prototype_id,
)

# Words (including contractions) plus standalone punctuation tokens.
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*|[^\w\s]", re.UNICODE)
MIN_BOUNDARY_SILENCE_MS = 32
DEFAULT_CHUNK_TOKENS = 256
MAX_LEXICON_EXEMPLARS = 3


def tokenize(text: str):
    return [x.upper() for x in TOKEN_RE.findall(str(text))]


def untokenize(tokens):
    """Readable reconstruction without storing the source string."""
    toks = [str(x) for x in tokens]
    if not toks:
        return ""
    no_space_before = set(".,;:!?%)]}")
    no_space_after = set("([{\"")
    tight = {"-", "/"}
    out = ""
    prev = None
    for tok in toks:
        if not out:
            out = tok
        elif tok in no_space_before or tok in tight or (prev in no_space_after) or (prev in tight):
            out += tok
        else:
            out += " " + tok
        prev = tok
    return out


def _legacy_clean_episode(ep):
    out = {
        "frames": [(int(t), tuple(int(n) for n in active)) for t, active in ep.get("frames", ())],
        "route_events": tuple((int(a), int(b), float(t)) for a, b, t in ep.get("route_events", ())),
        "transmissions": int(ep.get("transmissions", 0)),
    }
    if ep.get("format"):
        out["format"] = str(ep["format"])
    if ep.get("duration_ms") is not None:
        out["duration_ms"] = int(ep["duration_ms"])
    return out


def episode_fingerprint(ep):
    """Historical full-episode content ID retained for 0.9.53 compatibility."""
    h = hashlib.sha256()
    for t, active in ep.get("frames", ()):
        h.update(b"F")
        h.update(struct.pack("<qI", int(t), len(active)))
        for n in active:
            h.update(struct.pack("<I", int(n)))
    for a, b, t in ep.get("route_events", ()):
        h.update(b"R")
        h.update(struct.pack("<IId", int(a), int(b), float(t)))
    return h.hexdigest()


def route_fingerprint(ep):
    routes = ep.get("route_events", ())
    if not routes:
        return None
    t0 = float(routes[0][2])
    h = hashlib.sha256()
    for a, b, t in routes:
        h.update(struct.pack("<IId", int(a), int(b), float(t) - t0))
    return h.hexdigest()


def _blob_ref_bytes(ref):
    """Read one compact blob from bytes, scratch file, or a saved .lrnr ZIP."""
    if isinstance(ref, (bytes, bytearray, memoryview)):
        return bytes(ref)
    if not isinstance(ref, dict):
        raise TypeError(f"unsupported engram reference {type(ref)!r}")
    kind = ref.get("kind")
    if kind == "bytes":
        return bytes(ref["data"])
    if kind == "file":
        with open(ref["path"], "rb") as f:
            return f.read()
    if kind == "zip":
        with zipfile.ZipFile(ref["path"], "r") as z:
            return z.read(ref["member"])
    if kind == "pack":
        with open(ref["path"], "rb") as f:
            f.seek(int(ref["offset"]))
            data = f.read(int(ref["length"]))
        if len(data) != int(ref["length"]):
            raise ValueError("truncated pretrained neural prototype pack")
        return data
    raise ValueError(f"unknown engram reference kind {kind!r}")


class NeuralSentenceMemory:
    def __init__(self, brain):
        self.brain = brain
        for name, default in (
            ("decoder_lexicon", {}),
            ("decoder_route_fingerprints", {}),
            ("decoder_route_labels", {}),
            ("neural_pattern_bank", {}),
            ("neural_prototype_bank", {}),
            ("neural_prototype_routes", {}),
            ("neural_token_prototypes", {}),
            ("neural_prototype_postings", {}),
            ("neural_memories", []),
            ("neural_memory_index", {}),
            ("neural_documents", []),
            ("memory_next_id", 0),
            ("document_next_id", 0),
        ):
            if not hasattr(brain, name):
                setattr(brain, name, default.copy() if isinstance(default, (dict, list)) else default)
        self._scratch_dir = tempfile.mkdtemp(prefix="lrnr_engrams_")
        self.decoder = None
        self.exact_decode = {}
        self.route_decode = {}
        self.label_routes = {}
        self.label_prototypes = {}
        self.route_postings = {}
        self._migrate_or_seed_lexicon()
        self._load_pretrained_vocabulary()
        self._migrate_memories()
        self._rebuild_indexes()

    # ---------- compact blob management ----------
    def _spool_blob(self, blob: bytes, stem: str, ext="mbe"):
        path = os.path.join(self._scratch_dir, f"{stem}.{ext}")
        with open(path, "wb") as f:
            f.write(blob)
        return {"kind": "file", "path": path}

    def _prototype_blob(self, pid):
        return _blob_ref_bytes(self.brain.neural_prototype_bank[str(pid)])

    def _memory_layout(self, rec):
        if rec.get("format") != "prototype_continuous_v3":
            raise ValueError("memory is not prototype_continuous_v3")
        return _blob_ref_bytes(rec["layout"])

    def _memory_blob(self, rec):
        fmt = rec.get("format")
        if fmt == "compact_continuous_v2":
            return _blob_ref_bytes(rec["engram"])
        if fmt == "prototype_continuous_v3":
            return reconstruct_engram(self._memory_layout(rec), self._prototype_blob)
        raise ValueError("memory is not a supported continuous neural format")

    def _find_memory(self, memory_id):
        mid = int(memory_id)
        for rec in self.brain.neural_memories:
            if int(rec.get("id", -1)) == mid:
                return rec
        raise KeyError(f"Unknown neural memory {mid}")

    # ---------- compact V3 postings ----------
    _POSTING = struct.Struct("<IH")

    def _add_prototype_posting(self, pid, memory_id, position):
        pos = int(position)
        if not (0 <= pos <= 0xFFFF):
            raise ValueError("region position exceeds compact posting range")
        pid = str(pid)
        buf = self.brain.neural_prototype_postings.get(pid)
        if buf is None:
            buf = bytearray()
            self.brain.neural_prototype_postings[pid] = buf
        elif isinstance(buf, bytes):
            buf = bytearray(buf)
            self.brain.neural_prototype_postings[pid] = buf
        buf += self._POSTING.pack(int(memory_id), pos)

    def _iter_prototype_postings(self, pid):
        raw = self.brain.neural_prototype_postings.get(str(pid), b"")
        if len(raw) % self._POSTING.size:
            raise ValueError("corrupt compact prototype postings")
        for mid, pos in self._POSTING.iter_unpack(bytes(raw)):
            yield int(mid), int(pos)

    # ---------- decoder dictionary ----------
    def _register_episode_pattern(self, ep):
        ep = _legacy_clean_episode(ep)
        pid = episode_fingerprint(ep)
        if pid not in self.brain.neural_pattern_bank:
            self.brain.neural_pattern_bank[pid] = pack_episode(ep, sparse_frames=False)
        return pid

    def _migrate_or_seed_lexicon(self):
        # Convert any old in-RAM JSON episode patterns to compact blobs.
        for pid, value in list((self.brain.neural_pattern_bank or {}).items()):
            if isinstance(value, dict) and value.get("kind") not in ("region_ref", "prototype_ref"):
                try:
                    self.brain.neural_pattern_bank[pid] = pack_episode(_legacy_clean_episode(value), sparse_frames=False)
                except Exception:
                    pass

        # Migrate older lexicons that embedded episode mappings directly.
        migrated = {}
        for label, entries in (self.brain.decoder_lexicon or {}).items():
            ids = []
            for x in entries:
                if isinstance(x, str):
                    ids.append(x)
                elif isinstance(x, dict) and x.get("kind") == "region_ref":
                    pid = hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()
                    self.brain.neural_pattern_bank[pid] = x
                    ids.append(pid)
                elif isinstance(x, dict):
                    ids.append(self._register_episode_pattern(x))
            if ids:
                migrated[str(label).upper()] = ids
        self.brain.decoder_lexicon = migrated

        if not self.brain.decoder_lexicon:
            compact_path = os.path.join(os.path.dirname(__file__), "baseline_neural_dictionary_compact.json")
            legacy_path = os.path.join(os.path.dirname(__file__), "baseline_neural_dictionary.json")
            if os.path.exists(compact_path):
                import base64
                with open(compact_path, encoding="utf8") as f:
                    raw = json.load(f)
                if tuple(raw.get("scales", ())) != tuple(SCALES):
                    raise RuntimeError(f"Baseline dictionary scale mismatch: {raw.get('scales')} vs {SCALES}")
                self.brain.decoder_lexicon = {str(k).upper(): list(v) for k, v in raw.get("lexicon", {}).items()}
                self.brain.decoder_route_fingerprints.update(raw.get("route_fingerprints", {}))
                for pid, b64 in raw.get("pattern_bank_b64", {}).items():
                    self.brain.neural_pattern_bank[pid] = base64.b64decode(b64)
            elif os.path.exists(legacy_path):
                with open(legacy_path, encoding="utf8") as f:
                    raw = json.load(f)
                if tuple(raw.get("scales", ())) != tuple(SCALES):
                    raise RuntimeError(f"Baseline dictionary scale mismatch: {raw.get('scales')} vs {SCALES}")
                self.brain.decoder_lexicon = {str(k).upper(): list(v) for k, v in raw.get("lexicon", {}).items()}
                self.brain.decoder_route_fingerprints.update(raw.get("route_fingerprints", {}))
                for pid, ep in raw.get("pattern_bank", {}).items():
                    self.brain.neural_pattern_bank[pid] = pack_episode(_legacy_clean_episode(ep), sparse_frames=False)

    def _load_pretrained_vocabulary(self):
        """Attach the shipped read-only neural vocabulary pack lazily.

        The pack contains exact MBE2 neural prototypes and decoder labels only.
        It is not episodic/user memory and is never copied into the user's writable
        neural store. Missing/corrupt optional packs leave runtime learning intact.
        """
        vocab_dir = os.path.join(os.path.dirname(__file__), "pretrained_vocabulary")
        manifest_path = os.path.join(vocab_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            return
        try:
            with open(manifest_path, encoding="utf8") as f:
                manifest = json.load(f)
            if manifest.get("format") != "AEVUM_NEURAL_VOCAB_V1":
                return
            pack_path = os.path.join(vocab_dir, str(manifest.get("pack_file") or "vocabulary.avnp"))
            if not os.path.exists(pack_path):
                return
            for token, meta in (manifest.get("entries") or {}).items():
                token = str(token).upper()
                pid = str(meta["prototype_id"])
                self.brain.neural_prototype_bank.setdefault(pid, {
                    "kind": "pack", "path": pack_path,
                    "offset": int(meta["offset"]), "length": int(meta["length"]),
                    "builtin": True,
                })
                route = str(meta.get("route") or "")
                if route:
                    self.brain.neural_prototype_routes.setdefault(pid, route)
                self.brain.neural_token_prototypes.setdefault(token, {
                    "prototype_id": pid, "lead_ms": int(meta.get("lead_ms", 1)),
                    "builtin": True,
                })
            self.pretrained_vocabulary_count = len(manifest.get("entries") or {})
        except Exception:
            # Vocabulary acceleration is optional. A bad/missing pack must never
            # prevent the base neural memory engine from starting and learning.
            self.pretrained_vocabulary_count = 0

    def _episode(self, pid):
        value = self.brain.neural_pattern_bank[pid]
        if isinstance(value, (bytes, bytearray, memoryview)):
            return unpack_episode(bytes(value))
        if isinstance(value, dict) and value.get("kind") == "region_ref":
            rec = self._find_memory(value["memory_id"])
            blob = self._memory_blob(rec)
            return compact_extract_region(blob, value["start"], value["end"])
        if isinstance(value, dict) and value.get("kind") == "prototype_ref":
            return unpack_episode(self._prototype_blob(value["prototype_id"]))
        if isinstance(value, dict):
            return _legacy_clean_episode(value)
        raise TypeError(f"unsupported pattern-bank value {type(value)!r}")

    def _add_route_label(self, digest: bytes, label: str):
        if not digest:
            return
        label = str(label).upper()
        hx = bytes(digest).hex()
        old = self.brain.decoder_route_labels.get(hx)
        if old is None:
            self.brain.decoder_route_labels[hx] = label
        elif old != label:
            # Preserve ambiguity rather than silently overwriting a collision.
            vals = old if isinstance(old, list) else [old]
            if label not in vals:
                vals.append(label)
            self.brain.decoder_route_labels[hx] = vals

    def _learn_region_label(self, label, memory_id, start, end, digest, prototype_id=None):
        label = str(label).upper()
        self._add_route_label(digest, label)
        bucket = self.brain.decoder_lexicon.setdefault(label, [])
        # Keep only a few full fuzzy exemplars; exact route variants live in the
        # much lighter digest->label dictionary.  V3 exemplars reference the
        # shared neural prototype instead of pinning a whole memory chunk.
        if len(bucket) < MAX_LEXICON_EXEMPLARS:
            if prototype_id is not None:
                key = f"PROTO:{prototype_id}".encode()
                pid = hashlib.sha256(b"PROTOREF\0" + key).hexdigest()
                if pid not in self.brain.neural_pattern_bank:
                    self.brain.neural_pattern_bank[pid] = {
                        "kind": "prototype_ref", "prototype_id": str(prototype_id),
                    }
            else:
                key = f"{int(memory_id)}:{int(start)}:{int(end)}:{bytes(digest).hex()}".encode()
                pid = hashlib.sha256(b"REGIONREF\0" + key).hexdigest()
                if pid not in self.brain.neural_pattern_bank:
                    self.brain.neural_pattern_bank[pid] = {
                        "kind": "region_ref", "memory_id": int(memory_id),
                        "start": int(start), "end": int(end),
                    }
            if pid not in bucket:
                bucket.append(pid)
            self.brain.decoder_route_fingerprints[pid] = bytes(digest).hex()
        self.decoder = None

    # ---------- memory migration / index ----------
    def _migrate_memories(self):
        normalized = []
        for rec in self.brain.neural_memories or []:
            rid = int(rec.get("id", len(normalized)))
            fmt = rec.get("format")
            if fmt == "prototype_continuous_v3" and "layout" in rec:
                normalized.append(rec)
            elif fmt == "compact_continuous_v2" and "engram" in rec:
                normalized.append(rec)
            elif fmt == "continuous_v1" and "episode" in rec:
                ep = _legacy_clean_episode(rec["episode"])
                blob = pack_episode(ep, sparse_frames=True)
                normalized.append({
                    "id": rid, "format": "compact_continuous_v2",
                    "engram": self._spool_blob(blob, f"migrated_{rid}"),
                    "duration_ms": compact_header(blob)["duration_ms"],
                })
            elif "pattern_ids" in rec:
                normalized.append({"id": rid, "format": "segmented_legacy", "pattern_ids": list(rec["pattern_ids"])})
            elif "words" in rec:
                ids = [self._register_episode_pattern(x) for x in rec.get("words", ())]
                normalized.append({"id": rid, "format": "segmented_legacy", "pattern_ids": ids})
        self.brain.neural_memories = normalized
        self.brain.memory_next_id = max([x["id"] for x in normalized], default=-1) + 1

    def _index_compact_memory(self, rec, force=False):
        """Derived neural-region index for both V2 and V3 continuous memories."""
        key = str(int(rec["id"]))
        if rec.get("format") == "prototype_continuous_v3":
            # V3 layouts already carry exact neural prototype IDs.  Do not retain a
            # second Python tuple/digest index for every token; derive rows only for
            # the chunk currently being opened/decoded.
            layout = self._memory_layout(rec)
            rows = []
            for a, b, pid in layout_regions(layout, self._prototype_blob):
                hx = self.brain.neural_prototype_routes.get(pid)
                if not hx:
                    pblob = self._prototype_blob(pid)
                    dig = route_signatures_for_regions(pblob, [(0, compact_header(pblob)["duration_ms"] - 1)], nbytes=32)[0]
                    hx = bytes(dig).hex() if dig else ""
                    self.brain.neural_prototype_routes[pid] = hx
                rows.append((int(a), int(b), bytes.fromhex(hx) if hx else b""))
            return rows
        if not force and key in self.brain.neural_memory_index:
            return self.brain.neural_memory_index[key]
        else:
            blob = self._memory_blob(rec)
            bounds = discover_regions(blob, MIN_BOUNDARY_SILENCE_MS)
            digests = route_signatures_for_regions(blob, bounds, nbytes=32)
            rows = [(int(a), int(b), bytes(d) if d else b"") for (a, b), d in zip(bounds, digests)]
        self.brain.neural_memory_index[key] = rows
        return rows

    def _rebuild_indexes(self):
        self.exact_decode = {}
        self.route_decode = {}
        self.label_routes = {}
        self.label_prototypes = {}
        self.route_postings = {}

        # Historical / exemplar route fingerprints.
        for label, ids in self.brain.decoder_lexicon.items():
            for pid in ids:
                self.exact_decode.setdefault(pid, []).append(label)
                fp = self.brain.decoder_route_fingerprints.get(pid)
                if fp:
                    dig = bytes.fromhex(fp)
                    self.route_decode.setdefault(dig, []).append(label)
                    self.label_routes.setdefault(label, set()).add(dig)

        # Decoder labels -> exact shared neural prototypes.
        for label, ids in self.brain.decoder_lexicon.items():
            for eid in ids:
                val = self.brain.neural_pattern_bank.get(eid)
                if isinstance(val, dict) and val.get("kind") == "prototype_ref":
                    self.label_prototypes.setdefault(str(label).upper(), set()).add(str(val["prototype_id"]))
        for label, meta in (self.brain.neural_token_prototypes or {}).items():
            pid = meta.get("prototype_id") if isinstance(meta, dict) else None
            if pid:
                self.label_prototypes.setdefault(str(label).upper(), set()).add(str(pid))

        # Lightweight learned exact variants.
        for hx, labels in (self.brain.decoder_route_labels or {}).items():
            try:
                dig = bytes.fromhex(hx)
            except Exception:
                continue
            vals = labels if isinstance(labels, list) else [labels]
            for label in vals:
                label = str(label).upper()
                self.route_decode.setdefault(dig, []).append(label)
                self.label_routes.setdefault(label, set()).add(dig)

        # V3 postings are a packed 6-byte occurrence stream keyed by prototype ID.
        # Rebuild them from tiny layouts only when loading an older V3 project that
        # did not persist the packed postings index.  Legacy V2 keeps route postings.
        have_v3_postings = bool(self.brain.neural_prototype_postings)
        for rec in self.brain.neural_memories:
            fmt = rec.get("format")
            if fmt == "prototype_continuous_v3":
                if not have_v3_postings:
                    parsed = parse_layout(self._memory_layout(rec))
                    for pos, (_gap, pid) in enumerate(parsed["placements"]):
                        self._add_prototype_posting(pid, int(rec["id"]), pos)
                continue
            if fmt == "compact_continuous_v2":
                rows = self._index_compact_memory(rec)
                for pos, (_, _, dig) in enumerate(rows):
                    if dig:
                        self.route_postings.setdefault(bytes(dig), []).append((int(rec["id"]), pos))

    # ---------- fuzzy decoder (lazy) ----------
    def _ensure_decoder(self, labels=None):
        # Full dictionary construction is retained for compatibility and small
        # vocabularies. Normal stored-memory recall does not need it.
        dec = CurrentDecoder()
        wanted = set(labels) if labels else None
        for label, ids in self.brain.decoder_lexicon.items():
            if wanted is not None and label not in wanted:
                continue
            for i, pid in enumerate(ids):
                try:
                    dec.store(label, self._episode(pid), {"kind": "brain_lexicon", "pattern_id": pid, "index": i})
                except Exception:
                    continue
        return dec

    def _decode_episode(self, ep):
        fp = route_fingerprint(ep)
        if fp:
            dig = bytes.fromhex(fp)
            exact = list(dict.fromkeys(self.route_decode.get(dig, ())))
            if len(exact) == 1:
                return {"token": exact[0], "score": 1.0, "margin": 1.0, "runner": None, "mode": "exact-recurrent-route"}
        dec = self._ensure_decoder()
        if not dec.families:
            return {"token": "<?>", "score": 0.0, "margin": 0.0, "runner": None, "mode": "unknown"}
        rows = dec.recall(ep)
        best = rows[0]
        second = rows[1] if len(rows) > 1 else None
        return {
            "token": best["label"], "score": float(best["score"]),
            "margin": float(best["score"] - (second["score"] if second else 0.0)),
            "runner": second["label"] if second else None, "mode": "multiscale-fuzzy",
        }

    # ---------- compiled deterministic neural ingest ----------
    def _register_prototype_blob(self, pid, pblob):
        pid = str(pid)
        added = 0
        if pid not in self.brain.neural_prototype_bank:
            self.brain.neural_prototype_bank[pid] = self._spool_blob(pblob, f"prototype_{pid}")
            added = len(pblob)
        if pid not in self.brain.neural_prototype_routes:
            dig = route_signatures_for_regions(pblob, [(0, compact_header(pblob)["duration_ms"] - 1)], nbytes=32)[0]
            self.brain.neural_prototype_routes[pid] = bytes(dig).hex() if dig else ""
        return added

    def _prototype_from_existing_v2(self, token):
        """Promote an exact deterministic V2 occurrence to the V3 prototype cache.

        Aevum 0.2.0/0.2.1 memories were produced by the same fixed, non-plastic
        encoder with the same 96 ms reset. Their discovered token regions therefore
        contain the exact deterministic prototype already; re-simulating that token
        during migration would only waste CPU and make the UI lag.
        """
        token = str(token).upper()
        for dig in self.label_routes.get(token, set()):
            for mid, pos in self.route_postings.get(dig, ()): 
                try:
                    rec = self._find_memory(mid)
                    if rec.get("format") != "compact_continuous_v2":
                        continue
                    rows = self._index_compact_memory(rec)
                    if not (0 <= int(pos) < len(rows)):
                        continue
                    a, b, _ = rows[int(pos)]
                    pblob = slice_region_blob(self._memory_blob(rec), int(a), int(b))
                    pid = prototype_id(pblob)
                    self._register_prototype_blob(pid, pblob)
                    meta = {"prototype_id": pid, "lead_ms": 1, "migrated_from_v2": True}
                    self.brain.neural_token_prototypes[token] = meta
                    self.label_prototypes.setdefault(token, set()).add(pid)
                    return meta
                except Exception:
                    continue
        return None

    def _ensure_token_prototype(self, token):
        """Learn one deterministic token response once and cache its neural prototype.

        This optimization is valid only for the current non-plastic, deterministic
        ingest regime.  If plasticity is enabled, caller must use direct simulation.
        """
        token = str(token).upper()
        meta = (self.brain.neural_token_prototypes or {}).get(token)
        if meta and meta.get("prototype_id") in self.brain.neural_prototype_bank:
            return meta, 0, False

        migrated = self._prototype_from_existing_v2(token)
        if migrated is not None:
            return migrated, 0, False

        # Exact optimized deterministic encoder. Regression tests require its full
        # neural-event digest to match the canonical Brain implementation.
        blob = run_fast_compact_token_experience([token])
        times = active_times(blob)
        if not times:
            raise RuntimeError(f"Token {token!r} produced no neural activity")
        start, end = int(times[0]), int(times[-1])
        trailing_silence = int(compact_header(blob)["duration_ms"]) - end - 1
        if trailing_silence < 16:
            raise RuntimeError(f"Token {token!r} lacks the minimum validated neural reset interval")
        # A word may legitimately contain an internal quiet interval >32 ms. Keep
        # first..last activity as one exact token prototype rather than incorrectly
        # splitting the word into multiple decoder tokens.
        pblob = slice_region_blob(blob, start, end)
        pid = prototype_id(pblob)
        added = self._register_prototype_blob(pid, pblob)
        meta = {"prototype_id": pid, "lead_ms": start}
        self.brain.neural_token_prototypes[token] = meta
        return meta, added, True

    def _compile_token_layout(self, toks):
        """Build an exact continuous neural layout from cached deterministic prototypes."""
        placements = []
        proto_rows = []
        cursor = 0
        clock = 0
        new_prototype_bytes = 0
        newly_simulated = 0
        for i, token in enumerate(toks):
            if i:
                clock += WORD_GAP_MS
            meta, added, learned = self._ensure_token_prototype(token)
            new_prototype_bytes += int(added)
            newly_simulated += int(bool(learned))
            pid = meta["prototype_id"]
            start = int(clock) + int(meta.get("lead_ms", 1))
            pblob = self._prototype_blob(pid)
            pdur = int(compact_header(pblob)["duration_ms"] )
            end = start + pdur - 1
            gap = start - cursor
            if gap < 0:
                raise RuntimeError(f"Cached neural prototype overlap for {token!r}")
            placements.append((gap, pid))
            proto_rows.append((start, end, pid))
            cursor = end + 1
            clock += len(str(token)) * CHAR_SLOT_MS
        duration = int(clock + TAIL_MS)
        return pack_layout(duration, placements), proto_rows, new_prototype_bytes, newly_simulated

    # ---------- store ----------
    def _store_tokens(self, tokens, document_id=None, chunk_index=None, force_simulation=False):
        toks = [str(x).upper() for x in tokens if str(x)]
        if not toks:
            raise ValueError("Nothing to store")

        # Under the current deterministic/non-plastic regime, a learned token's
        # recurrent neural region is invariant across the validated quiet reset.
        # Simulate each previously unseen token once, then compile exact subsequent
        # occurrences from the shared prototype.  If plasticity is ever enabled,
        # fall back to the direct continuous Brain simulation path.
        compiled = (not force_simulation) and (not bool(getattr(self.brain.c, "plasticity", False)))
        compile_error = None
        if compiled:
            try:
                layout_blob, proto_rows, new_prototype_bytes, newly_simulated = self._compile_token_layout(toks)
                source_compact_bytes = None
            except RuntimeError as ex:
                # A token whose activity does not settle within the validated macro
                # envelope is encoded through the original continuous Brain path.
                compiled = False
                compile_error = str(ex)
        if not compiled:
            source_blob = run_compact_token_experience(toks, seed=None)
            layout_blob, prototypes, proto_rows = factorize_engram(source_blob)
            if len(proto_rows) != len(toks):
                raise RuntimeError(f"Neural segmentation mismatch: source tokens={len(toks)} discovered_regions={len(proto_rows)}")
            new_prototype_bytes = 0
            for pid, pblob in prototypes.items():
                new_prototype_bytes += self._register_prototype_blob(pid, pblob)
            newly_simulated = len(toks)
            source_compact_bytes = len(source_blob)

        rid = int(self.brain.memory_next_id)
        self.brain.memory_next_id = rid + 1
        rec = {
            "id": rid, "format": "prototype_continuous_v3",
            "layout": self._spool_blob(layout_blob, f"layout_{rid}", ext="mbl"),
            "duration_ms": int(parse_layout(layout_blob)["duration_ms"]),
        }
        if document_id is not None:
            rec["document_id"] = int(document_id)
            rec["chunk_index"] = int(chunk_index or 0)
        self.brain.neural_memories.append(rec)

        learned_words = 0
        for pos, (token, (a, b, pid)) in enumerate(zip(toks, proto_rows)):
            hx = self.brain.neural_prototype_routes.get(pid, "")
            dig = bytes.fromhex(hx) if hx else b""
            self._add_prototype_posting(pid, rid, pos)
            self.label_prototypes.setdefault(token, set()).add(str(pid))
            if dig:
                before = token in self.brain.decoder_lexicon
                self._learn_region_label(token, rid, a, b, dig, prototype_id=pid)
                if not before:
                    learned_words += 1
                self.route_decode.setdefault(dig, [])
                if token not in self.route_decode[dig]:
                    self.route_decode[dig].append(token)
                self.label_routes.setdefault(token, set()).add(dig)

        # Lossless reconstruction is exhaustively checked by the codec/regression
        # tests.  Do not rebuild the just-created engram here: bulk ingest must not
        # pay a second full decode/hash pass for every chunk.
        physical_added = len(layout_blob) + new_prototype_bytes
        return {
            "id": rid, "format": "prototype_continuous_v3",
            "duration_ms": rec["duration_ms"], "tokens": len(toks),
            "new_words_learned": learned_words,
            "engram_bytes": physical_added, "layout_bytes": len(layout_blob),
            "new_prototype_bytes": new_prototype_bytes,
            "source_compact_bytes": source_compact_bytes,
            "compiled_reuse": bool(compiled),
            "new_prototypes_simulated": int(newly_simulated),
            "compiled_fallback_reason": compile_error,
        }

    def store_sentence(self, text, force_simulation=False):
        toks = tokenize(text)
        if not toks:
            raise ValueError("Nothing to store")
        return self._store_tokens(toks, force_simulation=force_simulation)

    def store_text(self, text, chunk_tokens=DEFAULT_CHUNK_TOKENS, source_name=None, progress=None, force_simulation=False):
        """Bulk ingest text as bounded continuous neural chunks.

        `source_name` is optional external metadata (for example a filename); source
        content is never persisted.  Chunk boundaries are storage boundaries only.
        """
        toks = tokenize(text)
        if not toks:
            raise ValueError("Nothing to store")
        size = max(1, int(chunk_tokens))
        if force_simulation:
            size = min(size, 12)
        else:
            size = min(size, 0xFFFF)
        chunks = [toks[i:i + size] for i in range(0, len(toks), size)]
        did = int(self.brain.document_next_id)
        self.brain.document_next_id = did + 1
        chunk_ids = []
        total_bytes = 0
        new_words = 0
        total_duration = 0
        total_layout = 0
        total_new_prototypes = 0
        prototypes_simulated = 0
        compiled_chunks = 0
        for i, chunk in enumerate(chunks):
            info = self._store_tokens(chunk, document_id=did, chunk_index=i, force_simulation=force_simulation)
            chunk_ids.append(info["id"])
            total_bytes += info["engram_bytes"]
            new_words += info["new_words_learned"]
            total_duration += info["duration_ms"]
            total_layout += int(info.get("layout_bytes", 0))
            total_new_prototypes += int(info.get("new_prototype_bytes", 0))
            prototypes_simulated += int(info.get("new_prototypes_simulated", 0))
            compiled_chunks += int(bool(info.get("compiled_reuse")))
            if progress:
                progress(i + 1, len(chunks), info)
        doc = {"id": did, "chunk_ids": chunk_ids, "token_count": len(toks)}
        if source_name:
            doc["source_name"] = os.path.basename(str(source_name))
        self.brain.neural_documents.append(doc)
        return {
            "document_id": did, "chunks": len(chunk_ids), "chunk_ids": chunk_ids,
            "tokens": len(toks), "new_words_learned": new_words,
            "engram_bytes": total_bytes, "duration_ms": total_duration,
            "layout_bytes": total_layout, "new_prototype_bytes": total_new_prototypes,
            "prototypes_simulated": prototypes_simulated, "compiled_chunks": compiled_chunks,
            "strict_simulation": bool(force_simulation),
        }

    # ---------- recall ----------
    def _decode_compact_memory(self, rec):
        rows = self._index_compact_memory(rec)
        decoded = []
        blob = None
        for pos, (a, b, dig) in enumerate(rows):
            labels = list(dict.fromkeys(self.route_decode.get(bytes(dig), ()))) if dig else []
            if len(labels) == 1:
                row = {"token": labels[0], "score": 1.0, "margin": 1.0, "runner": None, "mode": "exact-neural-route-digest"}
            else:
                if blob is None:
                    blob = self._memory_blob(rec)
                ep = compact_extract_region(blob, a, b)
                row = self._decode_episode(ep)
            row["region_ms"] = [int(a), int(b)]
            row["region"] = pos
            decoded.append(row)
        return decoded

    def open_memory(self, memory_id):
        rec = self._find_memory(memory_id)
        if rec.get("format") in ("compact_continuous_v2", "prototype_continuous_v3"):
            decoded = self._decode_compact_memory(rec)
            return {
                "id": int(rec["id"]), "format": rec.get("format"),
                "sentence": untokenize([x["token"] for x in decoded]),
                "decoded": decoded, "discovered_regions": len(decoded),
                "engram_bytes": self.memory_storage_bytes(rec["id"]),
                "document_id": rec.get("document_id"), "chunk_index": rec.get("chunk_index"),
            }
        decoded = []
        for pid in rec.get("pattern_ids", ()):
            ep = self._episode(pid)
            decoded.append(self._decode_episode(ep))
        return {"id": int(rec["id"]), "format": "segmented_legacy", "sentence": untokenize([x["token"] for x in decoded]), "decoded": decoded}

    def open_document(self, document_id):
        did = int(document_id)
        doc = next((x for x in self.brain.neural_documents if int(x.get("id", -1)) == did), None)
        if doc is None:
            raise KeyError(f"Unknown neural document {did}")
        decoded = []
        for mid in doc.get("chunk_ids", ()):
            decoded.extend(self.open_memory(mid)["decoded"])
        return {
            "document_id": did,
            "source_name": doc.get("source_name"),
            "sentence": untokenize([x["token"] for x in decoded]),
            "decoded": decoded,
            "chunks": len(doc.get("chunk_ids", ())),
        }

    def search(self, cue_text, top_k=20, min_score=0.75):
        cue_tokens = tokenize(cue_text)
        if len(cue_tokens) != 1:
            raise ValueError("Search currently accepts one token at a time")
        token = cue_tokens[0]

        # Fastest V3 path: decoder label -> exact neural prototype(s) -> compact
        # 6-byte occurrence postings.  Episodic memory still contains no English.
        hits = {}
        for pid in self.label_prototypes.get(token, set()):
            for mid, pos in self._iter_prototype_postings(pid):
                key = int(mid)
                old = hits.get(key)
                if old is None or pos < old["position"]:
                    rec = self._find_memory(mid)
                    hits[key] = {
                        "memory_id": key, "position": int(pos), "score": 1.0,
                        "state": 1.0, "diff": 1.0, "temporal": 1.0,
                        "fine_0_5ms": 1.0, "isi": 1.0,
                        "mode": "indexed-neural-prototype",
                        "document_id": rec.get("document_id"),
                    }

        # Legacy / noisy exact-route variants remain supported.
        variants = self.label_routes.get(token, set())
        for dig in variants:
            for mid, pos in self.route_postings.get(dig, ()):
                key = int(mid)
                old = hits.get(key)
                if old is None or pos < old["position"]:
                    rec = self._find_memory(mid)
                    hits[key] = {
                        "memory_id": key, "position": int(pos), "score": 1.0,
                        "state": 1.0, "diff": 1.0, "temporal": 1.0,
                        "fine_0_5ms": 1.0, "isi": 1.0,
                        "mode": "indexed-neural-route-digest",
                        "document_id": rec.get("document_id"),
                    }
        if hits:
            rows = sorted(hits.values(), key=lambda x: x["memory_id"])
            return rows[:max(1, int(top_k))]

        # If the label exists but no route postings matched (legacy or unusual
        # state), use its exemplar as a cue and scan only on demand.
        ids = self.brain.decoder_lexicon.get(token, ())
        if ids:
            cue = self._episode(ids[0])
        else:
            cue = _legacy_clean_episode(run_calibrated_experience(token, seed=None))

        best_by_memory = {}
        for rec in self.brain.neural_memories:
            if rec.get("format") in ("compact_continuous_v2", "prototype_continuous_v3"):
                blob = None
                for pos, (a, b, _) in enumerate(self._index_compact_memory(rec)):
                    if blob is None:
                        blob = self._memory_blob(rec)
                    ep = compact_extract_region(blob, a, b)
                    total, state, diff, temporal = score(cue, ep)
                    row = {
                        "memory_id": int(rec["id"]), "position": pos,
                        "score": float(total), "state": float(state), "diff": float(diff),
                        "temporal": float(temporal["combined"]), "fine_0_5ms": float(temporal["fine"]),
                        "isi": float(temporal["isi"]), "mode": "multiscale-fuzzy",
                        "document_id": rec.get("document_id"),
                    }
                    old = best_by_memory.get(row["memory_id"])
                    if old is None or row["score"] > old["score"]:
                        best_by_memory[row["memory_id"]] = row
            else:
                for pos, pid in enumerate(rec.get("pattern_ids", ())):
                    ep = self._episode(pid)
                    total, state, diff, temporal = score(cue, ep)
                    row = {
                        "memory_id": int(rec["id"]), "position": pos,
                        "score": float(total), "state": float(state), "diff": float(diff),
                        "temporal": float(temporal["combined"]), "fine_0_5ms": float(temporal["fine"]),
                        "isi": float(temporal["isi"]), "mode": "multiscale-fuzzy",
                        "document_id": None,
                    }
                    old = best_by_memory.get(row["memory_id"])
                    if old is None or row["score"] > old["score"]:
                        best_by_memory[row["memory_id"]] = row
        rows = [x for x in best_by_memory.values() if x["score"] >= float(min_score)]
        rows.sort(key=lambda x: x["score"], reverse=True)
        return rows[:max(1, int(top_k))]

    # ---------- stats / maintenance ----------
    def vocabulary(self):
        return sorted(self.brain.decoder_lexicon)

    def memory_count(self):
        return len(self.brain.neural_memories)

    def document_count(self):
        return len(self.brain.neural_documents)

    def memory_storage_bytes(self, memory_id):
        rec = self._find_memory(memory_id)
        ref = None
        if rec.get("format") == "compact_continuous_v2":
            ref = rec.get("engram")
        elif rec.get("format") == "prototype_continuous_v3":
            ref = rec.get("layout")
        if ref is None:
            return 0
        if isinstance(ref, (bytes, bytearray, memoryview)):
            return len(ref)
        if isinstance(ref, dict) and ref.get("kind") == "file":
            try:
                return os.path.getsize(ref["path"])
            except OSError:
                return 0
        if isinstance(ref, dict) and ref.get("kind") == "zip":
            try:
                with zipfile.ZipFile(ref["path"], "r") as z:
                    return z.getinfo(ref["member"]).file_size
            except Exception:
                return 0
        return 0

    def _refs_total_bytes(self, refs):
        total = 0
        zip_members = {}
        for ref in refs:
            if isinstance(ref, (bytes, bytearray, memoryview)):
                total += len(ref)
            elif isinstance(ref, dict) and ref.get("kind") == "file":
                try: total += os.path.getsize(ref["path"])
                except OSError: pass
            elif isinstance(ref, dict) and ref.get("kind") == "zip":
                zip_members.setdefault(ref["path"], []).append(ref["member"])
        for path, members in zip_members.items():
            try:
                with zipfile.ZipFile(path, "r") as z:
                    for member in members:
                        try: total += z.getinfo(member).file_size
                        except KeyError: pass
            except Exception:
                pass
        return total

    def prototype_storage_bytes(self):
        return self._refs_total_bytes((self.brain.neural_prototype_bank or {}).values())

    def postings_storage_bytes(self):
        return sum(len(x) for x in (self.brain.neural_prototype_postings or {}).values())

    def total_engram_bytes(self):
        # Physical neural store estimate: layouts/legacy engrams + each shared
        # prototype once + the compact exact-recall postings index.
        refs = []
        for rec in self.brain.neural_memories:
            if rec.get("format") == "prototype_continuous_v3": refs.append(rec.get("layout"))
            elif rec.get("format") == "compact_continuous_v2": refs.append(rec.get("engram"))
        return self._refs_total_bytes(refs) + self.prototype_storage_bytes() + self.postings_storage_bytes()

    def clear_memories(self, keep_lexicon=True):
        self.brain.neural_memories = []
        self.brain.neural_memory_index = {}
        self.brain.neural_documents = []
        self.brain.memory_next_id = 0
        self.brain.document_next_id = 0
        self.route_postings.clear()
        self.label_prototypes.clear()
        self.brain.neural_prototype_postings = {}
        if not keep_lexicon:
            self.brain.decoder_lexicon = {}
            self.brain.decoder_route_fingerprints = {}
            self.brain.decoder_route_labels = {}
            self.brain.neural_pattern_bank = {}
            self.brain.neural_prototype_bank = {}
            self.brain.neural_prototype_routes = {}
            self.brain.neural_token_prototypes = {}
            self._migrate_or_seed_lexicon()
        self._rebuild_indexes()
