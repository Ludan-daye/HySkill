"""Aggregate SR-Agents retrieval JSONs into a Markdown comparison table."""

import json
import sys
from collections import defaultdict
from pathlib import Path

METRICS = ["Recall@1", "Recall@5", "Recall@10", "Recall@50", "nDCG@10"]


def main(result_dir: str) -> None:
    table = defaultdict(dict)  # (dataset, retriever) -> metrics
    for p in sorted(Path(result_dir).glob("*.json")):
        d = json.loads(p.read_text())
        meta, metrics = d.get("metadata", {}), d.get("metrics") or {}
        table[(meta.get("dataset", p.stem), meta.get("retriever", "?"))] = metrics
    datasets = sorted({k[0] for k in table})
    retrievers = sorted({k[1] for k in table})
    for ds in datasets:
        print(f"\n## {ds}\n")
        print("| retriever | " + " | ".join(METRICS) + " |")
        print("|" + "---|" * (len(METRICS) + 1))
        for r in retrievers:
            m = table.get((ds, r), {})
            row = " | ".join(f"{m[x]:.3f}" if x in m else "—" for x in METRICS)
            print(f"| {r} | {row} |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/retrieval")
