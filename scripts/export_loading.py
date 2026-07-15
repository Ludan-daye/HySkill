#!/usr/bin/env python
"""Export per-instance LOADING data (which skill each arm actually put in
context) into community-results/<TAG>/loading_per_instance.jsonl.gz.

Row: {instance_id, domain, arm, loaded: [skill_ids], gold: [...],
      hit: 1 if any loaded skill is gold else 0 (null when nothing loaded)}

Covers every answering arm whose jsonl exists. `loaded` is empty for bare and
for gate-blocked instances — that IS the loading decision record.

Usage:
  fleet model :   python scripts/export_loading.py <TAG>
  4B reference:   python scripts/export_loading.py qwen3.5-4b-reference --phase2
"""

import ast
import gzip
import json
import sys
from pathlib import Path

RULE_DOMAINS = ["theoremqa", "logicbench", "medcalcbench", "champ"]
FLEET_ARMS = ["bare", "always", "gated", "select", "always_rerank", "select_bm25"]
P2_ARMS = ["bare", "always", "gated", "select", "oracle", "always_r", "gated_r"]


def parse_used(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str) and v.strip():
        try:
            out = ast.literal_eval(v)
            return list(out) if isinstance(out, (list, tuple)) else [str(out)]
        except (ValueError, SyntaxError):
            return [v]
    return []


def main():
    tag = sys.argv[1]
    phase2 = "--phase2" in sys.argv
    out_dir = Path(f"community-results/{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with gzip.open(out_dir / "loading_per_instance.jsonl.gz", "wt", encoding="utf-8") as f:
        for ds in RULE_DOMAINS:
            if phase2:
                gold_src = Path(f"results/phase2/{ds}-routed.json")
                arm_paths = {a: Path(f"results/phase2/{ds}-{a}.jsonl") for a in P2_ARMS}
                arm_paths["always_rerank"] = Path(f"results/multimodel/qwen35-4b-baselines/{ds}-always_rerank.jsonl")
                arm_paths["select_bm25"] = Path(f"results/multimodel/qwen35-4b-baselines/{ds}-select_bm25.jsonl")
            else:
                gold_src = Path(f"results/multimodel/{tag}/{ds}-routed.json")
                arm_paths = {a: Path(f"results/multimodel/{tag}/{ds}-{a}.jsonl") for a in FLEET_ARMS}
            gold = {r["instance_id"]: set(r["gold_skill_ids"])
                    for r in json.loads(gold_src.read_text())["results"]}
            for arm, p in arm_paths.items():
                if not p.exists():
                    continue
                for line in p.open():
                    r = json.loads(line)
                    iid = r["instance_id"]
                    loaded = parse_used(r.get("skill_ids_used"))
                    g = gold.get(iid, set())
                    f.write(json.dumps({
                        "instance_id": iid, "domain": ds, "arm": arm,
                        "loaded": loaded, "gold": sorted(g),
                        "hit": (1 if any(s in g for s in loaded) else 0) if loaded else None,
                    }, ensure_ascii=False) + "\n")
                    n += 1
    print(json.dumps({"loading_per_instance.jsonl.gz": n}))


if __name__ == "__main__":
    main()
