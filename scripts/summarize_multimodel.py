#!/usr/bin/env python
"""Package a community multi-model run into a small summary JSON.

Reads results/multimodel/<tag>/ (big, local-only) and writes
community-results/<tag>/summary.json (small, meant to be PRed back).

Usage: python scripts/summarize_multimodel.py <tag> [<model-name>]
"""

import json
import sys
from pathlib import Path

VARIANTS = ["naive_sentence", "naive_passage", "naive_skill",
            "hyskill", "two_stage", "bm25", "llm_rerank", "routed"]


def main() -> None:
    tag = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else tag
    src = Path("results/multimodel") / tag
    out = {"tag": tag, "generator": model,
           "encoder": "sentence-transformers/all-MiniLM-L6-v2",
           "k_samples": 4, "retrieval": {}, "router": {}, "gating": {}}

    for p in sorted(src.glob("*.json")):
        if p.name.endswith(".eval.json"):
            continue
        ds, _, variant = p.stem.partition("-")
        if variant not in VARIANTS:
            continue
        d = json.loads(p.read_text())
        out["retrieval"].setdefault(ds, {})[variant] = d.get("metrics", {})
        if variant == "routed":
            out["router"][ds] = d.get("metadata", {}).get("router", {})

    for p in sorted(src.glob("*.eval.json")):
        ds, _, arm = p.stem.removesuffix(".eval").partition("-")
        m = json.loads(p.read_text())["metrics"]
        out["gating"].setdefault(ds, {})[arm] = {
            "accuracy": round(m["accuracy"], 4),
            "correct": m["correct"], "total": m["total"]}

    for p in sorted(src.glob("*-taus.json")):
        ds = p.stem.partition("-")[0]
        t = json.loads(p.read_text())
        out["gating"].setdefault(ds, {})["taus"] = {
            "tau1": t["tau1"], "tau2": t["tau2"]}
    for p in sorted(src.glob("*-gated.json")):
        ds = p.stem.partition("-")[0]
        g = json.loads(p.read_text()).get("metadata", {}).get("gate")
        if g:
            out["gating"].setdefault(ds, {})["gate"] = g

    dst = Path("community-results") / tag
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "summary.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {dst / 'summary.json'}")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
