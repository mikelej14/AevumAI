from __future__ import annotations

import json
import math
import os
import re
import struct
import threading
import time
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

MAGIC = b"CBRMEM01"
VERSION = 1
HEADER = struct.Struct("<8sHHQ")  # magic, version, flags, created_ns
RECORD_HEADER = struct.Struct("<BBHQQIII")
# type, flags, reserved, record_id, timestamp_ns, raw_len, comp_len, crc32(raw)

TYPE_EPISODE = 1
TYPE_TURN = 2
TYPE_STATE = 3
TYPE_LESSON = 4
TYPE_SOURCE = 5
TYPE_FEEDBACK = 6
TYPE_NAMES = {
    TYPE_EPISODE: "episode",
    TYPE_TURN: "turn",
    TYPE_STATE: "state",
    TYPE_LESSON: "lesson",
    TYPE_SOURCE: "source",
    TYPE_FEEDBACK: "feedback",
}
NAME_TYPES = {v: k for k, v in TYPE_NAMES.items()}

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.'-]{1,63}")
STOP = {
    "the","and","for","that","with","this","from","was","were","are","you","your","but","not","have","has",
    "had","what","when","where","who","why","how","into","about","they","them","their","then","than","can","could",
    "would","should","will","just","its","it's","our","out","all","any","some","more","very","there","here","been",
    "being","also","because","does","did","doing","make","made","like","want","need","use","using","used","user",
    "assistant","model","system","turn","message","response","said","says","say","get","got","getting","over","under",
}


@dataclass
class MemoryHit:
    record_id: int
    record_type: str
    timestamp: float
    score: float
    salience: float
    text: str
    payload: Dict[str, Any]

    def compact(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "type": self.record_type,
            "timestamp": self.timestamp,
            "score": round(self.score, 5),
            "salience": round(self.salience, 4),
            "text": self.text,
        }


class RMEMError(RuntimeError):
    pass


class CognitiveRMEM:
    """Append-only proprietary cognitive memory store.

    The file is canonical. In-memory indexes are disposable and rebuilt by scanning
    the file at startup. Records are individually compressed and CRC protected so
    a damaged tail cannot invalidate earlier experience.
    """

    def __init__(self, path: str | Path, compression_level: int = 3, recency_half_life_days: float = 45.0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.compression_level = max(1, min(9, int(compression_level)))
        self.recency_half_life_days = max(1.0, float(recency_half_life_days))
        self.lock = threading.RLock()
        self.records: Dict[int, Dict[str, Any]] = {}
        self.record_meta: Dict[int, Tuple[int, float]] = {}
        self.postings: Dict[str, set[int]] = defaultdict(set)
        self.term_freqs: Dict[int, Counter[str]] = {}
        self.doc_lengths: Dict[int, int] = {}
        self.avg_doc_len: float = 1.0
        self._next_id = 1
        self._open_or_create()
        self._scan()

    def _open_or_create(self) -> None:
        if not self.path.exists():
            with self.path.open("wb") as f:
                f.write(HEADER.pack(MAGIC, VERSION, 0, time.time_ns()))
                f.flush()
                os.fsync(f.fileno())
            return
        with self.path.open("rb") as f:
            head = f.read(HEADER.size)
        if len(head) != HEADER.size:
            raise RMEMError("RMEM header is incomplete")
        magic, version, _flags, _created = HEADER.unpack(head)
        if magic != MAGIC:
            raise RMEMError("Not a Cognitive Brain RMEM file")
        if version != VERSION:
            raise RMEMError(f"Unsupported RMEM version {version}; expected {VERSION}")

    @staticmethod
    def tokenize(text: str) -> List[str]:
        toks = []
        for m in TOKEN_RE.finditer(text.lower()):
            t = m.group(0).strip(".'-")
            if len(t) < 2 or t in STOP:
                continue
            toks.append(t)
        return toks

    @staticmethod
    def _record_text(payload: Dict[str, Any]) -> str:
        preferred = []
        for key in (
            "user_text", "assistant_text", "content", "summary", "lesson", "focus", "context",
            "expectation", "outcome", "interpretation", "text",
        ):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                preferred.append(value.strip())
        for key in ("entities", "goals", "connections", "open_loops", "tags"):
            value = payload.get(key)
            if isinstance(value, list):
                preferred.extend(str(x) for x in value if str(x).strip())
        if not preferred:
            preferred.append(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return "\n".join(preferred)

    def _index(self, rid: int, payload: Dict[str, Any]) -> None:
        text = self._record_text(payload)
        tf = Counter(self.tokenize(text))
        self.term_freqs[rid] = tf
        self.doc_lengths[rid] = sum(tf.values())
        for term in tf:
            self.postings[term].add(rid)

    def _recalc_avg(self) -> None:
        self.avg_doc_len = (sum(self.doc_lengths.values()) / max(1, len(self.doc_lengths))) or 1.0

    def _scan(self) -> None:
        with self.lock:
            self.records.clear()
            self.record_meta.clear()
            self.postings.clear()
            self.term_freqs.clear()
            self.doc_lengths.clear()
            max_id = 0
            with self.path.open("rb") as f:
                f.seek(HEADER.size)
                while True:
                    pos = f.tell()
                    raw_hdr = f.read(RECORD_HEADER.size)
                    if not raw_hdr:
                        break
                    if len(raw_hdr) != RECORD_HEADER.size:
                        # Interrupted append: preserve valid prefix and ignore tail.
                        break
                    rtype, flags, _reserved, rid, ts_ns, raw_len, comp_len, crc = RECORD_HEADER.unpack(raw_hdr)
                    blob = f.read(comp_len)
                    if len(blob) != comp_len:
                        break
                    try:
                        raw = zlib.decompress(blob) if (flags & 0x01) else blob
                    except zlib.error:
                        break
                    if len(raw) != raw_len or (zlib.crc32(raw) & 0xFFFFFFFF) != crc:
                        break
                    try:
                        payload = json.loads(raw.decode("utf-8"))
                    except Exception:
                        break
                    self.records[rid] = payload
                    self.record_meta[rid] = (rtype, ts_ns / 1e9)
                    self._index(rid, payload)
                    max_id = max(max_id, rid)
            self._next_id = max_id + 1
            self._recalc_avg()

    def append(self, record_type: str, payload: Dict[str, Any], timestamp: Optional[float] = None) -> int:
        rtype = NAME_TYPES.get(record_type)
        if rtype is None:
            raise ValueError(f"Unknown record type: {record_type}")
        with self.lock:
            rid = self._next_id
            self._next_id += 1
            ts = float(timestamp if timestamp is not None else time.time())
            body = dict(payload)
            body.setdefault("record_id", rid)
            body.setdefault("record_type", record_type)
            body.setdefault("timestamp", ts)
            raw = json.dumps(body, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
            comp = zlib.compress(raw, self.compression_level)
            use_comp = len(comp) + 8 < len(raw)
            blob = comp if use_comp else raw
            flags = 0x01 if use_comp else 0x00
            hdr = RECORD_HEADER.pack(
                rtype, flags, 0, rid, int(ts * 1e9), len(raw), len(blob), zlib.crc32(raw) & 0xFFFFFFFF
            )
            with self.path.open("ab") as f:
                f.write(hdr)
                f.write(blob)
                f.flush()
                os.fsync(f.fileno())
            self.records[rid] = body
            self.record_meta[rid] = (rtype, ts)
            self._index(rid, body)
            self._recalc_avg()
            return rid

    def get(self, record_id: int) -> Optional[Dict[str, Any]]:
        with self.lock:
            record = self.records.get(int(record_id))
            return dict(record) if isinstance(record, dict) else None

    def latest(self, count: int = 50, types: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
        with self.lock:
            allowed = None if types is None else {NAME_TYPES[x] for x in types if x in NAME_TYPES}
            out = []
            for rid in sorted(self.records, reverse=True):
                rtype, _ts = self.record_meta[rid]
                if allowed is not None and rtype not in allowed:
                    continue
                out.append(dict(self.records[rid]))
                if len(out) >= count:
                    break
            return out

    def _score_candidates(self, query_terms: List[str], candidate_ids: set[int], now: float) -> List[Tuple[float, int]]:
        n_docs = max(1, len(self.records))
        k1, b = 1.35, 0.72
        qtf = Counter(query_terms)
        scored: List[Tuple[float, int]] = []
        for rid in candidate_ids:
            tf = self.term_freqs.get(rid, Counter())
            dl = max(1, self.doc_lengths.get(rid, 1))
            lexical = 0.0
            matches = 0
            for term, qcount in qtf.items():
                freq = tf.get(term, 0)
                if not freq:
                    continue
                matches += 1
                df = len(self.postings.get(term, ()))
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                lexical += idf * ((freq * (k1 + 1.0)) / (freq + k1 * (1.0 - b + b * dl / self.avg_doc_len))) * min(2, qcount)
            if lexical <= 0:
                continue
            payload = self.records[rid]
            salience = float(payload.get("salience", payload.get("importance", 0.5)) or 0.5)
            ts = self.record_meta[rid][1]
            age_days = max(0.0, (now - ts) / 86400.0)
            recency = math.exp(-math.log(2.0) * age_days / self.recency_half_life_days)
            convergence = matches / max(1, len(set(query_terms)))
            score = lexical * (1.0 + 0.30 * convergence) + 0.35 * salience + 0.10 * recency
            scored.append((score, rid))
        scored.sort(reverse=True)
        return scored

    def retrieve(self, query: str, limit: int = 18, extra_queries: Optional[Iterable[str]] = None, exclude_ids: Optional[Iterable[int]] = None) -> List[MemoryHit]:
        with self.lock:
            return self._retrieve_locked(query, limit=limit, extra_queries=extra_queries, exclude_ids=exclude_ids)

    def _retrieve_locked(self, query: str, limit: int = 18, extra_queries: Optional[Iterable[str]] = None, exclude_ids: Optional[Iterable[int]] = None) -> List[MemoryHit]:
        terms = self.tokenize(query)
        for q in extra_queries or ():
            terms.extend(self.tokenize(str(q)))
        if not terms:
            return []
        candidates: set[int] = set()
        # prioritize discriminative terms while still allowing convergence
        unique = list(dict.fromkeys(terms))
        unique.sort(key=lambda t: len(self.postings.get(t, ())))
        for term in unique[:48]:
            candidates.update(self.postings.get(term, ()))
        if exclude_ids:
            candidates.difference_update(int(x) for x in exclude_ids)
        scored = self._score_candidates(terms, candidates, time.time())
        hits: List[MemoryHit] = []
        for score, rid in scored[: max(1, int(limit))]:
            payload = self.records[rid]
            rtype, ts = self.record_meta[rid]
            text = self._record_text(payload)
            if len(text) > 2400:
                text = text[:2397] + "..."
            salience = float(payload.get("salience", payload.get("importance", 0.5)) or 0.5)
            hits.append(MemoryHit(rid, TYPE_NAMES.get(rtype, "unknown"), ts, score, salience, text, payload))
        return hits

    def stats(self) -> Dict[str, Any]:
        with self.lock:
            try:
                size = self.path.stat().st_size
            except OSError:
                size = 0
            type_counts = Counter(TYPE_NAMES.get(self.record_meta[rid][0], "unknown") for rid in self.records)
            return {
                "path": str(self.path),
                "size_bytes": size,
                "records": len(self.records),
                "terms": len(self.postings),
                "type_counts": dict(type_counts),
                "next_record_id": self._next_id,
            }

    def validate(self) -> Dict[str, Any]:
        with self.lock:
            return self._validate_locked()

    def _validate_locked(self) -> Dict[str, Any]:
        checked = 0
        valid_bytes = HEADER.size
        error = None
        with self.path.open("rb") as f:
            f.seek(HEADER.size)
            while True:
                start = f.tell()
                hdr = f.read(RECORD_HEADER.size)
                if not hdr:
                    valid_bytes = start
                    break
                if len(hdr) != RECORD_HEADER.size:
                    error = f"Incomplete record header at byte {start}"
                    valid_bytes = start
                    break
                _rtype, flags, _res, _rid, _ts, raw_len, comp_len, crc = RECORD_HEADER.unpack(hdr)
                blob = f.read(comp_len)
                if len(blob) != comp_len:
                    error = f"Incomplete record payload at byte {start}"
                    valid_bytes = start
                    break
                try:
                    raw = zlib.decompress(blob) if flags & 1 else blob
                except zlib.error as exc:
                    error = f"Decompression failure at byte {start}: {exc}"
                    valid_bytes = start
                    break
                if len(raw) != raw_len or (zlib.crc32(raw) & 0xFFFFFFFF) != crc:
                    error = f"CRC/length failure at byte {start}"
                    valid_bytes = start
                    break
                checked += 1
                valid_bytes = f.tell()
        return {"ok": error is None, "records_checked": checked, "valid_bytes": valid_bytes, "error": error}


class ResilientMemory:
    """Failure-isolated RMEM facade.

    Chat/session continuity must not depend on long-term memory availability. If the
    RMEM pack is missing, corrupt, or becomes inaccessible, calls degrade to neutral
    results until `reopen()` succeeds. In particular, a pack deleted while the app is
    running is never accidentally recreated as a headerless append-only file.
    """

    def __init__(self, path: str | Path, compression_level: int = 3, recency_half_life_days: float = 45.0):
        self.path = Path(path)
        self.compression_level = compression_level
        self.recency_half_life_days = recency_half_life_days
        self.lock = threading.RLock()
        self.backend: Optional[CognitiveRMEM] = None
        self.last_error = ""
        self.reopen()

    @property
    def available(self) -> bool:
        with self.lock:
            return self.backend is not None and self.path.exists()

    def _fail(self, exc: Exception | str) -> None:
        self.last_error = str(exc)
        self.backend = None

    def reopen(self) -> bool:
        with self.lock:
            try:
                self.backend = CognitiveRMEM(
                    self.path,
                    compression_level=self.compression_level,
                    recency_half_life_days=self.recency_half_life_days,
                )
                self.last_error = ""
                return True
            except Exception as exc:
                self._fail(exc)
                return False

    def _live(self) -> Optional[CognitiveRMEM]:
        with self.lock:
            if self.backend is None:
                return None
            if not self.path.exists():
                self._fail("RMEM file is missing. Chat remains available; reconnect or create a new RMEM pack from the Memory page.")
                return None
            return self.backend

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Public tokenizer shared with the RMEM backend for scheduler-side relevance checks."""
        return CognitiveRMEM.tokenize(text)

    def append(self, record_type: str, payload: Dict[str, Any], timestamp: Optional[float] = None) -> int:
        mem = self._live()
        if mem is None:
            return 0
        try:
            return mem.append(record_type, payload, timestamp=timestamp)
        except Exception as exc:
            with self.lock:
                self._fail(exc)
            return 0

    def get(self, record_id: int) -> Optional[Dict[str, Any]]:
        mem = self._live()
        if mem is None or not record_id:
            return None
        try:
            return mem.get(record_id)
        except Exception as exc:
            with self.lock:
                self._fail(exc)
            return None

    def latest(self, count: int = 50, types: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
        mem = self._live()
        if mem is None:
            return []
        try:
            return mem.latest(count=count, types=types)
        except Exception as exc:
            with self.lock:
                self._fail(exc)
            return []

    def retrieve(self, query: str, limit: int = 18, extra_queries: Optional[Iterable[str]] = None, exclude_ids: Optional[Iterable[int]] = None) -> List[MemoryHit]:
        mem = self._live()
        if mem is None:
            return []
        try:
            return mem.retrieve(query, limit=limit, extra_queries=extra_queries, exclude_ids=exclude_ids)
        except Exception as exc:
            with self.lock:
                self._fail(exc)
            return []

    def stats(self) -> Dict[str, Any]:
        mem = self._live()
        if mem is None:
            return {
                "path": str(self.path), "size_bytes": 0, "records": 0, "terms": 0,
                "type_counts": {}, "next_record_id": 0, "available": False,
                "error": self.last_error or "RMEM unavailable",
            }
        try:
            out = mem.stats()
            out.update({"available": True, "error": ""})
            return out
        except Exception as exc:
            with self.lock:
                self._fail(exc)
            return self.stats()

    def validate(self) -> Dict[str, Any]:
        mem = self._live()
        if mem is None:
            return {"ok": False, "records_checked": 0, "valid_bytes": 0, "error": self.last_error or "RMEM unavailable"}
        try:
            return mem.validate()
        except Exception as exc:
            with self.lock:
                self._fail(exc)
            return {"ok": False, "records_checked": 0, "valid_bytes": 0, "error": self.last_error}
