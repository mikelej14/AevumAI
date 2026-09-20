from __future__ import annotations

import gzip
import json
import os
import shutil
import struct
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from microbrain import Brain
from neural_sentence_memory import NeuralSentenceMemory, tokenize

STATE_VERSION = 2
POSTINGS_MAGIC = b"MBP1"
STOPWORDS = {
    "A","AN","THE","OF","TO","AND","OR","BUT","ABOUT","ON","IN","AT","FOR","FROM","WITH",
    "WHAT","WHEN","WHERE","WHO","WHY","HOW","WAS","WERE","IS","ARE","BE","BEEN","SAID","SAY",
    "TELL","ME","MY","YOUR","OUR","THAT","THIS","IT","LAST","EARLIER","BEFORE","AFTER",
}


def _iso_now():
    return datetime.now(timezone.utc).isoformat()




def _pack_prototype_postings(mapping):
    items = [(str(pid), bytes(raw)) for pid, raw in (mapping or {}).items() if raw]
    out = bytearray(POSTINGS_MAGIC)
    out += struct.pack("<I", len(items))
    for pid, raw in sorted(items):
        key = bytes.fromhex(pid)
        if len(key) != 32:
            raise ValueError("prototype posting key must be SHA-256")
        out += key
        out += struct.pack("<I", len(raw))
        out += raw
    return bytes(out)


def _unpack_prototype_postings(blob):
    data = bytes(blob)
    if len(data) < 8 or data[:4] != POSTINGS_MAGIC:
        raise ValueError("bad prototype postings index")
    count = struct.unpack_from("<I", data, 4)[0]
    pos = 8
    out = {}
    for _ in range(count):
        if pos + 36 > len(data):
            raise ValueError("truncated prototype postings index")
        pid = data[pos:pos+32].hex(); pos += 32
        n = struct.unpack_from("<I", data, pos)[0]; pos += 4
        if pos + n > len(data):
            raise ValueError("truncated prototype postings payload")
        out[pid] = bytearray(data[pos:pos+n]); pos += n
    if pos != len(data):
        raise ValueError("trailing bytes in prototype postings index")
    return out


def _ref_bytes(ref):
    if isinstance(ref, (bytes, bytearray, memoryview)):
        return bytes(ref)
    if not isinstance(ref, dict):
        raise TypeError(f"unsupported neural blob reference {type(ref)!r}")
    kind = ref.get("kind")
    if kind == "bytes":
        return bytes(ref["data"])
    if kind == "file":
        with open(ref["path"], "rb") as f:
            return f.read()
    if kind == "zip":
        import zipfile
        with zipfile.ZipFile(ref["path"], "r") as z:
            return z.read(ref["member"])
    if kind == "pack":
        with open(ref["path"], "rb") as f:
            f.seek(int(ref["offset"]))
            data = f.read(int(ref["length"]))
        if len(data) != int(ref["length"]):
            raise ValueError("truncated pretrained neural prototype pack")
        return data
    raise ValueError(f"unknown neural blob reference kind {kind!r}")

def _digest_index_to_json(index):
    out = {}
    for key, rows in (index or {}).items():
        out[str(key)] = [[int(a), int(b), (bytes(d).hex() if not isinstance(d, str) else d)] for a, b, d in rows]
    return out


def _digest_index_from_json(index):
    out = {}
    for key, rows in (index or {}).items():
        vals = []
        for a, b, hx in rows:
            try:
                dig = bytes.fromhex(hx) if isinstance(hx, str) else bytes(hx)
            except Exception:
                dig = b""
            vals.append((int(a), int(b), dig))
        out[str(key)] = vals
    return out


class NeuralMemoryRuntime:
    """Persistent product runtime around the verified 0.9.55 neural memory core.

    Content is recalled from compact neural engrams.  Human-readable metadata is
    deliberately limited to provenance (speaker, timestamp, role, chat/message IDs,
    source name, and optional semantic annotations); source message/document text is not stored
    in the neural-memory manifest or search index.
    """

    def __init__(self, data_dir: str | Path, *, chunk_tokens: int = 12):
        self.data_dir = Path(data_dir)
        self.engram_dir = self.data_dir / "engrams"
        self.layout_dir = self.data_dir / "layouts"
        self.prototype_dir = self.data_dir / "prototypes"
        self.postings_path = self.data_dir / "prototype_postings.mbp"
        self.state_path = self.data_dir / "neural_state.json.gz"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.engram_dir.mkdir(parents=True, exist_ok=True)
        self.layout_dir.mkdir(parents=True, exist_ok=True)
        self.prototype_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.chunk_tokens = max(1, int(chunk_tokens))
        self.brain = Brain()
        # Seed the shipped neural codec before overlaying learned state.
        self.memory = NeuralSentenceMemory(self.brain)
        self._load_state()

    # ---------- persistence ----------
    def _load_state(self):
        if not self.state_path.exists():
            return
        try:
            with gzip.open(self.state_path, "rt", encoding="utf-8") as f:
                state = json.load(f)
            version = int(state.get("version", 0))
            if version not in (1, STATE_VERSION):
                raise ValueError("unsupported neural runtime state")

            # Merge saved lexicon over the shipped baseline so baseline neural blobs
            # do not need to be copied into every user profile.
            for label, ids in (state.get("decoder_lexicon") or {}).items():
                bucket = self.brain.decoder_lexicon.setdefault(str(label).upper(), [])
                for pid in ids:
                    if pid not in bucket:
                        bucket.append(pid)
            self.brain.decoder_route_fingerprints.update(state.get("decoder_route_fingerprints") or {})
            self.brain.decoder_route_labels.update(state.get("decoder_route_labels") or {})
            for pid, ref in (state.get("learned_pattern_refs") or {}).items():
                self.brain.neural_pattern_bank[pid] = ref

            self.brain.neural_prototype_routes.update(state.get("neural_prototype_routes") or {})
            self.brain.neural_token_prototypes.update(state.get("neural_token_prototypes") or {})
            for pid, rel in (state.get("prototype_files") or {}).items():
                path = self.data_dir / str(rel)
                if path.exists():
                    self.brain.neural_prototype_bank[str(pid)] = {"kind": "file", "path": str(path)}
            if self.postings_path.exists():
                try:
                    self.brain.neural_prototype_postings = _unpack_prototype_postings(self.postings_path.read_bytes())
                except Exception:
                    self.brain.neural_prototype_postings = {}

            memories = []
            for rec in state.get("neural_memories") or []:
                x = dict(rec)
                rel = x.pop("engram_file", None)
                layout_rel = x.pop("layout_file", None)
                if rel:
                    x["engram"] = {"kind": "file", "path": str(self.data_dir / rel)}
                if layout_rel:
                    x["layout"] = {"kind": "file", "path": str(self.data_dir / layout_rel)}
                memories.append(x)
            self.brain.neural_memories = memories
            self.brain.neural_memory_index = _digest_index_from_json(state.get("neural_memory_index") or {})
            self.brain.neural_documents = list(state.get("neural_documents") or [])
            self.brain.memory_next_id = int(state.get("memory_next_id", len(memories)))
            self.brain.document_next_id = int(state.get("document_next_id", len(self.brain.neural_documents)))
            self.memory._rebuild_indexes()
        except Exception:
            # Preserve a corrupt state for diagnosis rather than deleting it.
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            bad = self.state_path.with_name(f"neural_state.corrupt-{stamp}.json.gz")
            try:
                self.state_path.replace(bad)
            except Exception:
                pass

    def _persist_neural_files(self):
        # Shared exact neural prototypes.
        for pid, ref in list((self.brain.neural_prototype_bank or {}).items()):
            # Shipped vocabulary prototypes stay in the read-only application pack;
            # do not copy thousands of built-ins into every user profile.
            if isinstance(ref, dict) and ref.get("kind") == "pack":
                continue
            dst = self.prototype_dir / f"{pid}.mbe"
            try:
                src_path = Path(ref.get("path", "")) if isinstance(ref, dict) and ref.get("kind") == "file" else None
                if not dst.exists() or src_path is None or src_path.resolve() != dst.resolve():
                    dst.write_bytes(_ref_bytes(ref))
                self.brain.neural_prototype_bank[str(pid)] = {"kind": "file", "path": str(dst)}
            except Exception:
                continue

        # Episodic V2 engrams and V3 timing layouts.
        for rec in self.brain.neural_memories:
            fmt = rec.get("format")
            if fmt == "compact_continuous_v2":
                ref = rec.get("engram")
                dst = self.engram_dir / f"{int(rec['id']):08d}.mbe"
                if ref is not None:
                    src = Path(ref.get("path", "")) if isinstance(ref, dict) and ref.get("kind") == "file" else None
                    if not dst.exists() or src is None or src.resolve() != dst.resolve():
                        dst.write_bytes(_ref_bytes(ref))
                    rec["engram"] = {"kind": "file", "path": str(dst)}
            elif fmt == "prototype_continuous_v3":
                ref = rec.get("layout")
                dst = self.layout_dir / f"{int(rec['id']):08d}.mbl"
                if ref is not None:
                    src = Path(ref.get("path", "")) if isinstance(ref, dict) and ref.get("kind") == "file" else None
                    if not dst.exists() or src is None or src.resolve() != dst.resolve():
                        dst.write_bytes(_ref_bytes(ref))
                    rec["layout"] = {"kind": "file", "path": str(dst)}

        postings = self.brain.neural_prototype_postings or {}
        if postings:
            tmp = self.postings_path.with_suffix(self.postings_path.suffix + ".tmp")
            tmp.write_bytes(_pack_prototype_postings(postings))
            os.replace(tmp, self.postings_path)
        elif self.postings_path.exists():
            try: self.postings_path.unlink()
            except OSError: pass

    def save(self):
        with self.lock:
            self._persist_neural_files()
            memories = []
            for rec in self.brain.neural_memories:
                x = {k: v for k, v in rec.items() if k not in {"engram", "layout"}}
                ref = rec.get("engram")
                if isinstance(ref, dict) and ref.get("kind") == "file":
                    try:
                        x["engram_file"] = str(Path(ref["path"]).resolve().relative_to(self.data_dir.resolve()))
                    except Exception:
                        x["engram_file"] = str(Path("engrams") / f"{int(rec['id']):08d}.mbe")
                lref = rec.get("layout")
                if isinstance(lref, dict) and lref.get("kind") == "file":
                    try:
                        x["layout_file"] = str(Path(lref["path"]).resolve().relative_to(self.data_dir.resolve()))
                    except Exception:
                        x["layout_file"] = str(Path("layouts") / f"{int(rec['id']):08d}.mbl")
                memories.append(x)

            # Only persist region references learned from real memories.  Shipped
            # baseline blobs are reloaded from baseline_neural_dictionary_compact.json.
            pattern_refs = {
                pid: value for pid, value in (self.brain.neural_pattern_bank or {}).items()
                if isinstance(value, dict)
            }
            state = {
                "version": STATE_VERSION,
                "saved_utc": _iso_now(),
                "decoder_lexicon": self.brain.decoder_lexicon,
                "decoder_route_fingerprints": self.brain.decoder_route_fingerprints,
                "decoder_route_labels": self.brain.decoder_route_labels,
                "learned_pattern_refs": pattern_refs,
                "prototype_files": {str(pid): str(Path(ref["path"]).resolve().relative_to(self.data_dir.resolve()))
                                    for pid, ref in (self.brain.neural_prototype_bank or {}).items()
                                    if isinstance(ref, dict) and ref.get("kind") == "file"},
                "neural_prototype_routes": self.brain.neural_prototype_routes,
                "neural_token_prototypes": self.brain.neural_token_prototypes,
                "neural_memories": memories,
                "neural_memory_index": _digest_index_to_json(self.brain.neural_memory_index),
                "neural_documents": self.brain.neural_documents,
                "memory_next_id": int(self.brain.memory_next_id),
                "document_next_id": int(self.brain.document_next_id),
            }
            tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
            with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as f:
                json.dump(state, f, ensure_ascii=False, separators=(",", ":"))
            os.replace(tmp, self.state_path)

    # ---------- metadata helpers ----------
    def _doc(self, document_id: int):
        did = int(document_id)
        for doc in self.brain.neural_documents:
            if int(doc.get("id", -1)) == did:
                return doc
        raise KeyError(f"Unknown neural document {did}")

    def _rec(self, memory_id: int):
        return self.memory._find_memory(int(memory_id))

    @staticmethod
    def _in_range(doc, start_utc=None, end_utc=None):
        stamp = str(doc.get("timestamp_utc") or "")
        if not stamp:
            return True
        if start_utc and stamp < str(start_utc):
            return False
        if end_utc and stamp > str(end_utc):
            return False
        return True

    # ---------- ingest ----------
    def store_message(self, text: str, *, role: str, speaker: str, chat_id: str = "",
                      message_id: str = "", timestamp_utc: str | None = None,
                      progress=None):
        text = str(text or "").strip()
        if not text:
            raise ValueError("Cannot store an empty message")
        with self.lock:
            before = {int(x["id"]) for x in self.brain.neural_memories}
            info = self.memory.store_text(
                text, chunk_tokens=self.chunk_tokens,
                source_name=f"chat:{chat_id or 'default'}", progress=progress,
            )
            doc = self._doc(info["document_id"])
            doc.update({
                "kind": "chat_message",
                "role": str(role or "user"),
                "speaker": str(speaker or role or "Unknown"),
                "timestamp_utc": timestamp_utc or _iso_now(),
                "chat_id": str(chat_id or ""),
                "message_id": str(message_id or ""),
            })
            self.save()
            new_ids = [int(x["id"]) for x in self.brain.neural_memories if int(x["id"]) not in before]
            return info | {"memory_ids": new_ids, "metadata": dict(doc)}

    def store_text(self, text: str, *, source_name: str = "manual", timestamp_utc: str | None = None,
                   speaker: str = "Imported", progress=None):
        text = str(text or "").strip()
        if not text:
            raise ValueError("Nothing to remember")
        with self.lock:
            before = {int(x["id"]) for x in self.brain.neural_memories}
            info = self.memory.store_text(text, chunk_tokens=self.chunk_tokens, source_name=source_name, progress=progress)
            doc = self._doc(info["document_id"])
            doc.update({
                "kind": "manual_memory",
                "role": "source",
                "speaker": str(speaker or "Imported"),
                "timestamp_utc": timestamp_utc or _iso_now(),
            })
            self.save()
            new_ids = [int(x["id"]) for x in self.brain.neural_memories if int(x["id"]) not in before]
            return info | {"memory_ids": new_ids, "metadata": dict(doc)}

    def store_tool_note(self, text: str, *, tool: str, chat_id: str = "", turn_id: str = "",
                        timestamp_utc: str | None = None, progress=None):
        """Store a compact tool-use receipt, never the fetched/opened evidence body."""
        text = str(text or "").strip()
        if not text:
            return None
        with self.lock:
            before = {int(x["id"]) for x in self.brain.neural_memories}
            info = self.memory.store_text(text, chunk_tokens=self.chunk_tokens, source_name=f"tool:{tool}", progress=progress)
            doc = self._doc(info["document_id"])
            doc.update({
                "kind": "tool_note", "role": "tool", "speaker": "Runtime",
                "tool": str(tool or "tool"), "chat_id": str(chat_id or ""),
                "turn_id": str(turn_id or ""), "timestamp_utc": timestamp_utc or _iso_now(),
            })
            self.save()
            new_ids = [int(x["id"]) for x in self.brain.neural_memories if int(x["id"]) not in before]
            return info | {"memory_ids": new_ids, "metadata": dict(doc)}

    def annotate_document(self, document_id: int, annotation: dict[str, Any]):
        """Attach bounded semantic context to provenance without altering the neural engram."""
        with self.lock:
            doc = self._doc(document_id)
            clean = dict(annotation or {})
            clean["topics"] = [str(x)[:80] for x in list(clean.get("topics") or [])[:8]]
            clean["entities"] = [str(x)[:100] for x in list(clean.get("entities") or [])[:12]]
            clean["intent"] = str(clean.get("intent", "") or "")[:160]
            clean["context"] = str(clean.get("context", "") or "")[:320]
            doc["semantic"] = clean
            self.save()
            return dict(doc)

    # ---------- neural recall ----------
    def _query_tokens(self, query: str):
        toks = tokenize(query)
        content = [t for t in toks if t not in STOPWORDS]
        return content or toks

    def _decode_preview(self, doc, matched_memory_ids):
        # Chat messages are intentionally opened whole; large imported documents
        # return only the matching neural chunk(s) so a search doesn't decode a manual.
        if doc.get("kind") == "chat_message" or int(doc.get("token_count", 0) or 0) <= 80:
            try:
                return self.memory.open_document(doc["id"])["sentence"]
            except Exception:
                pass
        parts = []
        for mid in matched_memory_ids[:3]:
            try:
                parts.append(self.memory.open_memory(mid)["sentence"])
            except Exception:
                continue
        return " … ".join(parts)

    def search(self, query: str, *, start_utc: str | None = None, end_utc: str | None = None,
               speaker: str | None = None, role: str | None = None, top_k: int = 12):
        tokens = self._query_tokens(query)
        if not tokens:
            return []
        with self.lock:
            grouped: dict[int, dict[str, Any]] = {}
            for token in tokens:
                try:
                    hits = self.memory.search(token, top_k=max(50, int(top_k) * 10), min_score=0.72)
                except Exception:
                    continue
                for hit in hits:
                    rec = self._rec(hit["memory_id"])
                    did = rec.get("document_id")
                    if did is None:
                        continue
                    doc = self._doc(did)
                    if not self._in_range(doc, start_utc, end_utc):
                        continue
                    if speaker and str(doc.get("speaker", "")).lower() != str(speaker).lower():
                        continue
                    if role and str(doc.get("role", "")).lower() != str(role).lower():
                        continue
                    row = grouped.setdefault(int(did), {
                        "document_id": int(did), "match_tokens": set(), "memory_ids": set(),
                        "neural_score": 0.0, "metadata": dict(doc),
                    })
                    row["match_tokens"].add(token)
                    row["memory_ids"].add(int(hit["memory_id"]))
                    row["neural_score"] = max(float(row["neural_score"]), float(hit.get("score", 0.0)))

            rows = []
            for row in grouped.values():
                mids = sorted(row.pop("memory_ids"))
                mtoks = sorted(row.pop("match_tokens"))
                doc = row["metadata"]
                row["matched_memory_ids"] = mids
                row["match_tokens"] = mtoks
                row["match_count"] = len(mtoks)
                row["score"] = len(mtoks) + float(row["neural_score"])
                row["preview"] = self._decode_preview(doc, mids)
                rows.append(row)
            rows.sort(key=lambda x: (x["match_count"], x["neural_score"], x["metadata"].get("timestamp_utc", "")), reverse=True)
            return rows[:max(1, int(top_k))]

    def hint_search(self, query: str, *, start_utc: str | None = None, end_utc: str | None = None,
                    speaker: str | None = None, role: str | None = None, top_k: int = 6,
                    include_tool_notes: bool = False):
        """Fast automatic recall hints: IDs/provenance/semantic tags only, never decoded body text."""
        tokens = self._query_tokens(query)
        if not tokens:
            return []
        with self.lock:
            grouped: dict[int, dict[str, Any]] = {}
            for token in tokens:
                # Automatic/product recall never simulates an unseen query token. If a
                # word occurred in stored content it is already part of the hidden
                # decoder codec; unknown current words cannot have exact old postings.
                if token not in self.brain.decoder_lexicon:
                    continue
                try:
                    hits = self.memory.search(token, top_k=max(40, int(top_k) * 8), min_score=0.72)
                except Exception:
                    continue
                for hit in hits:
                    rec = self._rec(hit["memory_id"])
                    did = rec.get("document_id")
                    if did is None:
                        continue
                    doc = self._doc(did)
                    if doc.get("kind") == "tool_note" and not include_tool_notes:
                        continue
                    if not self._in_range(doc, start_utc, end_utc):
                        continue
                    if speaker and str(doc.get("speaker", "")).lower() != str(speaker).lower():
                        continue
                    if role and str(doc.get("role", "")).lower() != str(role).lower():
                        continue
                    row = grouped.setdefault(int(did), {
                        "document_id": int(did), "match_tokens": set(), "matched_memory_ids": set(), "neural_score": 0.0,
                        "timestamp_utc": doc.get("timestamp_utc", ""), "speaker": doc.get("speaker", ""),
                        "role": doc.get("role", ""), "kind": doc.get("kind", ""),
                        "source_name": doc.get("source_name", ""), "token_count": int(doc.get("token_count", 0) or 0),
                        "semantic": dict(doc.get("semantic") or {}),
                    })
                    row["match_tokens"].add(token)
                    row["matched_memory_ids"].add(int(hit["memory_id"]))
                    row["neural_score"] = max(float(row["neural_score"]), float(hit.get("score", 0.0)))
            rows = []
            for row in grouped.values():
                row["match_tokens"] = sorted(row["match_tokens"])
                row["matched_memory_ids"] = sorted(row["matched_memory_ids"])
                row["match_count"] = len(row["match_tokens"])
                row["score"] = row["match_count"] + float(row["neural_score"])
                rows.append(row)
            rows.sort(key=lambda x: (x["match_count"], x["neural_score"], x.get("timestamp_utc", "")), reverse=True)
            return rows[:max(1, int(top_k))]

    def search_full_documents(self, query: str, *, start_utc: str | None = None, end_utc: str | None = None,
                              speaker: str | None = None, role: str | None = None, top_k: int = 6,
                              include_tool_notes: bool = False):
        """Explicit AI recall: each hit contains the COMPLETE decoded neural document.

        A document may span many compact neural chunks.  This method deliberately opens
        the whole document rather than returning a matching chunk/snippet.
        """
        hints = self.hint_search(
            query, start_utc=start_utc, end_utc=end_utc, speaker=speaker, role=role,
            top_k=top_k, include_tool_notes=include_tool_notes,
        )
        out = []
        for hint in hints:
            try:
                opened = self.open_document(int(hint["document_id"]))
            except Exception as exc:
                out.append({**hint, "ok": False, "error": str(exc), "text": ""})
                continue
            out.append({**hint, "ok": True, "text": str(opened.get("sentence", "") or ""),
                        "metadata": opened.get("metadata", {})})
        return out

    def open_document(self, document_id: int):
        with self.lock:
            doc = dict(self._doc(document_id))
            decoded = self.memory.open_document(document_id)
            return {"metadata": doc, **decoded}

    def recent(self, *, limit: int = 10, speaker: str | None = None, role: str | None = None):
        with self.lock:
            docs = list(self.brain.neural_documents)
            if speaker:
                docs = [d for d in docs if str(d.get("speaker", "")).lower() == str(speaker).lower()]
            if role:
                docs = [d for d in docs if str(d.get("role", "")).lower() == str(role).lower()]
            docs.sort(key=lambda d: str(d.get("timestamp_utc", "")), reverse=True)
            out = []
            for doc in docs[:max(1, int(limit))]:
                try:
                    text = self.memory.open_document(doc["id"])["sentence"]
                except Exception:
                    text = "<decode failed>"
                out.append({"document_id": int(doc["id"]), "metadata": dict(doc), "text": text})
            return out

    def chat_message_ids(self):
        """IDs already represented in neural memory; used for crash-safe transcript reconciliation."""
        with self.lock:
            return {str(d.get("message_id")) for d in self.brain.neural_documents
                    if d.get("kind") == "chat_message" and str(d.get("message_id", "")).strip()}

    def stats(self):
        with self.lock:
            return {
                "documents": self.memory.document_count(),
                "neural_chunks": self.memory.memory_count(),
                "engram_bytes": self.memory.total_engram_bytes(),
                "decoder_labels": len(self.memory.vocabulary()),
                "route_patterns": len(self.brain.decoder_route_labels),
                "scales_ms": [0.5, 1.0, 2.0, 4.0],
            }

    # ---------- visualization ----------
    def memory_blob(self, memory_id: int):
        with self.lock:
            rec = self._rec(memory_id)
            return self.memory._memory_blob(rec)

    def document_first_blob(self, document_id: int):
        with self.lock:
            doc = self._doc(document_id)
            ids = list(doc.get("chunk_ids") or [])
            return self.memory_blob(ids[0]) if ids else None
