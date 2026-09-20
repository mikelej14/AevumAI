from __future__ import annotations
import argparse, json, random, tempfile, time
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.memory import CognitiveRMEM


def main():
    ap = argparse.ArgumentParser(description="Synthetic Cognitive RMEM storage/retrieval stress test")
    ap.add_argument("--records", type=int, default=12000)
    ap.add_argument("--queries", type=int, default=300)
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "stress.rmem"
        m = CognitiveRMEM(p)
        raw = 0
        t0 = time.perf_counter()
        for i in range(args.records):
            key = f"projectkey{i:06d}"
            payload = {
                "summary": f"Episode {i} concerned {key} with native cognitive memory and persistent context for Executive.",
                "lesson": f"{key} must preserve provenance and experience state.",
                "entities": [key, "Executive", "RMEM"],
                "salience": 0.2 + (i % 8) / 10,
            }
            raw += len(json.dumps(payload, separators=(",", ":")).encode())
            m.append("episode", payload, timestamp=1700000000 + i)
        build = time.perf_counter() - t0
        rng = random.Random(1234)
        sample = rng.sample(range(args.records), min(args.queries, args.records))
        correct = 0; q0 = time.perf_counter()
        for i in sample:
            key = f"projectkey{i:06d}"
            hits = m.retrieve(key, limit=8)
            correct += bool(hits and key in hits[0].text.lower())
        qt = time.perf_counter() - q0
        report = {
            "records": args.records,
            "build_seconds": build,
            "records_per_second": args.records / build,
            "raw_payload_bytes": raw,
            "rmem_bytes": p.stat().st_size,
            "rmem_over_raw": p.stat().st_size / raw,
            "recall_at_8": correct / len(sample),
            "queries_per_second": len(sample) / qt,
            "validation": m.validate(),
        }
        print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
