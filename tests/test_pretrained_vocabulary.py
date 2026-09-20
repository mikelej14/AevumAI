import hashlib
import json
from pathlib import Path

from core.neural_memory_runtime import NeuralMemoryRuntime

ROOT = Path(__file__).resolve().parents[1]


def test_pretrained_pack_contract(tmp_path):
    vocab = ROOT / "pretrained_vocabulary"
    manifest = json.loads((vocab / "manifest.json").read_text(encoding="utf-8"))
    pack = vocab / manifest["pack_file"]
    assert manifest["word_entry_count"] == 10000
    assert manifest["supplement_entry_count"] >= 200
    assert manifest["extra_entry_count"] == 28
    assert manifest["entry_count"] == len(manifest["entries"])
    assert hashlib.sha256(pack.read_bytes()).hexdigest() == manifest["pack_sha256"]
    quarantined = {row["token"] for row in json.loads((vocab / "quarantine.json").read_text(encoding="utf-8"))}
    assert {"ENGAGE", "ENGAGED", "ENGAGEMENT", "ENGAGING", "ENCAPSULATED"} <= quarantined
    assert quarantined.isdisjoint(manifest["entries"])

    runtime = NeuralMemoryRuntime(tmp_path / "nm")
    assert runtime.memory.pretrained_vocabulary_count == manifest["entry_count"]
    info = runtime.memory.store_text("THE RED FISH LIKES COLD WATER STORY MANUAL", chunk_tokens=64)
    assert info["prototypes_simulated"] == 0
    assert runtime.memory.open_document(info["document_id"])["sentence"] == "THE RED FISH LIKES COLD WATER STORY MANUAL"
