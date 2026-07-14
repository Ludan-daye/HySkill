#!/usr/bin/env python
"""L1 domain-level granularity router (multimodel v2 spec).

Per dataset: score every imagination variant's retrieval file by mean
nDCG@10 on the 20% validation split (same split rule and seed as
gate.py calibrate), pick the winner, and write a routed retrieval file —
a copy of the winning file with metadata.router recording the decision.
Downstream test reporting must exclude the validation ids (recomputable
from the seed; also stored in the gate taus file).

Validation offline replay (results/route_pilot.json): 20% val identifies
the full-data champion in 5/5 domains; macro nDCG@10 0.591 -> 0.636.

Usage:
  python scripts/route_variant.py --dir results/phase1 --dataset theoremqa \
      --out results/phase2/theoremqa-routed.json
"""

import argparse
import json
import math
import random
from pathlib import Path

DEFAULT_VARIANTS = ["naive_skill", "naive_passage", "naive_sentence",
                    "hyskill", "two_stage"]
FALLBACK = "naive_skill"


def ndcg_at_10(gold: set, ranked: list) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, s in enumerate(ranked[:10]) if s in gold)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), 10)))
    return dcg / idcg if idcg > 0 else 0.0


def route(files: dict[str, str], val_frac: float = 0.2, seed: int = 0):
    """files: variant name -> retrieval json path (missing ones already
    filtered out by caller). Returns (routed_dict, decision_dict)."""
    data = {v: json.load(open(p)) for v, p in files.items()}
    per = {v: {r["instance_id"]:
               ndcg_at_10(set(r["gold_skill_ids"]),
                          [x["skill_id"] for x in r["retrieved"]])
               for r in d["results"]} for v, d in data.items()}
    ids = sorted(set.intersection(*[set(m) for m in per.values()]))
    rng = random.Random(seed)
    val = rng.sample(ids, max(1, int(len(ids) * val_frac)))

    scores = {v: sum(per[v][i] for i in val) / len(val) for v in per}
    degenerate = max(scores.values()) <= 0.0
    pick = FALLBACK if degenerate and FALLBACK in files else \
        max(scores, key=lambda v: scores[v])

    routed = data[pick]
    routed.setdefault("metadata", {})["router"] = {
        "pick": pick, "val_ndcg": {v: round(s, 4) for v, s in scores.items()},
        "n_val": len(val), "val_frac": val_frac, "seed": seed,
        "degenerate": degenerate}
    return routed, routed["metadata"]["router"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="dir with <ds>-<variant>.json")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    files = {}
    for v in a.variants.split(","):
        p = Path(a.dir) / f"{a.dataset}-{v}.json"
        if p.exists():
            files[v] = str(p)
        else:
            print(f"  missing variant file, skipping: {p}")
    if not files:
        raise SystemExit("no variant files found")

    routed, decision = route(files, a.val_frac, a.seed)
    Path(a.out).write_text(json.dumps(routed))
    print(f"routed {a.dataset}: pick={decision['pick']} "
          f"val_ndcg={decision['val_ndcg']} -> {a.out}")


if __name__ == "__main__":
    main()
