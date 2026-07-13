#!/usr/bin/env python
"""Paired bootstrap significance test between two retrieval result files.

Recomputes per-instance metrics from the `results` records (instance_id,
gold_skill_ids, retrieved), sanity-checks the aggregate against the file's
stored metrics, then reports mean difference, 95% CI and two-sided p-value
from a paired bootstrap over instances.

Usage:
  python scripts/significance.py A.json B.json --metric ndcg@10 [--boot 10000]
"""

import argparse
import json
import math

import numpy as np


def per_instance_metric(record: dict, metric: str) -> float:
    gold = set(record["gold_skill_ids"])
    ranked = [r["skill_id"] for r in record["retrieved"]]
    if metric.startswith("recall@"):
        k = int(metric.split("@")[1])
        return len(set(ranked[:k]) & gold) / max(1, len(gold))
    if metric.startswith("ndcg@"):
        k = int(metric.split("@")[1])
        dcg = sum(1.0 / math.log2(i + 2)
                  for i, sid in enumerate(ranked[:k]) if sid in gold)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
        return dcg / idcg if idcg > 0 else 0.0
    raise ValueError(f"unknown metric {metric}")


def load_scores(path: str, metric: str) -> dict[str, float]:
    data = json.load(open(path))
    scores = {r["instance_id"]: per_instance_metric(r, metric)
              for r in data["results"]}
    stored = (data.get("metrics") or {})
    key = metric.replace("recall", "Recall").replace("ndcg", "nDCG")
    if key in stored:
        recomputed = float(np.mean(list(scores.values())))
        if abs(recomputed - stored[key]) > 0.02:
            print(f"  WARNING {path}: recomputed {metric}={recomputed:.4f} "
                  f"vs stored {stored[key]:.4f} — convention mismatch?")
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("file_a")
    ap.add_argument("file_b")
    ap.add_argument("--metric", default="ndcg@10")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    a, b = load_scores(args.file_a, args.metric), load_scores(args.file_b, args.metric)
    common = sorted(set(a) & set(b))
    if not common:
        raise SystemExit("no common instance_ids")
    da = np.array([a[i] for i in common])
    db = np.array([b[i] for i in common])
    diff = da - db

    rng = np.random.default_rng(args.seed)
    n = len(diff)
    boots = np.array([diff[rng.integers(0, n, n)].mean()
                      for _ in range(args.boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())

    print(f"n={n}  {args.metric}: A={da.mean():.4f}  B={db.mean():.4f}  "
          f"diff={diff.mean():+.4f}  95%CI=[{lo:+.4f},{hi:+.4f}]  p={p:.4f}"
          f"  {'SIGNIFICANT' if p < 0.05 else 'n.s.'}")


if __name__ == "__main__":
    main()
