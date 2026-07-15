#!/usr/bin/env python
"""Export FULL top-50 retrieval lists (instance x method) into
community-results/<TAG>/retrieval_top50.jsonl.gz.

Complements retrieval_top10.jsonl.gz (which carries names + per-row nDCG):
this file carries the complete 50-deep ranking as (skill_id, score, is_gold)
triples — skill names/descriptions are resolvable from the SRA corpus by id,
so they are not duplicated here to keep the artifact small.

Usage:
  fleet model :   python scripts/export_top50.py <TAG>
  4B reference:   python scripts/export_top50.py qwen3.5-4b-reference --phase1
"""

import gzip
import json
import sys
from pathlib import Path

DOMAINS = ["theoremqa", "logicbench", "medcalcbench", "champ", "bigcodebench"]
FLEET_METHODS = ["naive_sentence", "naive_passage", "naive_skill", "hyskill",
                 "two_stage", "bm25", "llm_rerank", "routed"]
P1_METHODS = ["bm25", "dense", "hybrid", "llm_rerank", "naive_sentence",
              "naive_passage", "naive_skill", "hyskill", "two_stage"]


def main():
    tag = sys.argv[1]
    phase1 = "--phase1" in sys.argv
    out_dir = Path(f"community-results/{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    with gzip.open(out_dir / "retrieval_top50.jsonl.gz", "wt", encoding="utf-8") as f:
        for ds in DOMAINS:
            sources = {}
            if phase1:
                for m in P1_METHODS:
                    sources[m] = Path(f"results/phase1/{ds}-{m}.json")
                sources["routed"] = Path(f"results/phase2/{ds}-routed.json")
            else:
                for m in FLEET_METHODS:
                    sources[m] = Path(f"results/multimodel/{tag}/{ds}-{m}.json")
            for m, p in sources.items():
                if not p.exists():
                    continue
                for r in json.loads(p.read_text())["results"]:
                    gold = set(r["gold_skill_ids"])
                    f.write(json.dumps({
                        "instance_id": r["instance_id"], "domain": ds, "variant": m,
                        "gold": sorted(gold),
                        "top50": [[x["skill_id"], round(float(x["score"]), 5),
                                   1 if x["skill_id"] in gold else 0]
                                  for x in r["retrieved"][:50]],
                    }, ensure_ascii=False) + "\n")
                    n_rows += 1
    print(json.dumps({"retrieval_top50.jsonl.gz": n_rows}))


if __name__ == "__main__":
    main()
