#!/usr/bin/env python
"""Offline replay: can per-instance signals route among imagination variants?

Compares routing levels on existing Phase 1 retrieval files (zero LLM calls;
generations read from cache, embeddings computed fresh on GPU):

  L0      each fixed variant (reference)
  L1      domain-level routing: variant with best mean nDCG@10 on the 20%
          val split (seed 0, same split rule as gate.py calibrate)
  L2a     difficulty routing: agreement of the 4 skill-template imaginations
          (mean pairwise cos); threshold calibrated on val; agreement >= t
          -> naive_skill (single-path), else -> hyskill (4-path fusion)
  L2b1    uniform-judge routing: per instance pick the variant whose top-1
          maximizes cos(skill-imagination centroid + query anchor, top-1
          full text) — the same quantity as the gate's S1
  L2b3    same judge but DCG-weighted over each variant's top-3
  ORACLE  per-instance best variant (routing upper bound)

All levels are evaluated on the SAME 80% test split. Output: printed table
+ results/route_pilot.json.
"""

import json
import math
import random
from itertools import combinations
from pathlib import Path

import numpy as np

from sragents.cli.retrieve import _build_query
from sragents.corpus import load_corpus_dict, skill_text
from hyskill.embedder import Embedder
from hyskill.generator import HypotheticalGenerator, SKILL_TEMPLATE

DOMAINS = ["theoremqa", "logicbench", "medcalcbench", "champ", "bigcodebench"]
VARIANTS = ["naive_skill", "naive_passage", "naive_sentence", "hyskill", "two_stage"]
SRA = "external/SR-Agents"
MODEL = "qwen3.5-4b"
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


class _NoNet:
    def complete(self, *_a, **_k):
        raise RuntimeError("cache miss — pilot must be cache-only")


def ndcg_at_10(gold: set, ranked: list) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, s in enumerate(ranked[:10]) if s in gold)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), 10)))
    return dcg / idcg if idcg > 0 else 0.0


def main() -> None:
    emb = Embedder(model_name=ENCODER)
    gen = HypotheticalGenerator(client=_NoNet(), k_samples=4,
                                template=SKILL_TEMPLATE,
                                cache_dir="results/hyp_cache",
                                model_tag=f"{MODEL}|{SKILL_TEMPLATE[:20]}")
    corpus = load_corpus_dict(f"{SRA}/data/bench/corpus/corpus.json")

    report = {}
    for ds in DOMAINS:
        ret = {v: {r["instance_id"]: r for r in
                   json.load(open(f"results/phase1/{ds}-{v}.json"))["results"]}
               for v in VARIANTS}
        ids = sorted(set.intersection(*[set(m) for m in ret.values()]))
        instances = {i["instance_id"]: i for i in
                     json.load(open(f"{SRA}/data/bench/instances/{ds}.json"))}

        # per-instance ndcg + top-k skill ids per variant
        nd = {v: {i: ndcg_at_10(set(ret[v][i]["gold_skill_ids"]),
                                [r["skill_id"] for r in ret[v][i]["retrieved"]])
                  for i in ids} for v in VARIANTS}
        topk = {v: {i: [r["skill_id"] for r in ret[v][i]["retrieved"][:3]]
                    for i in ids} for v in VARIANTS}

        # centroids + agreement from cached imaginations (cache-only)
        cents, agree, miss = {}, {}, 0
        for i in ids:
            q = _build_query(instances[i])
            try:
                docs = gen.generate(q)
            except RuntimeError:
                miss += 1
                continue
            if not docs:
                miss += 1
                continue
            dv = emb.encode(docs)
            qv = emb.encode([q])[0]
            c = np.vstack([dv, qv[None, :]]).mean(axis=0)
            cents[i] = c / max(np.linalg.norm(c), 1e-9)
            pairs = [float(dv[a] @ dv[b]) for a, b in combinations(range(len(dv)), 2)]
            agree[i] = float(np.mean(pairs)) if pairs else 1.0
        ids = [i for i in ids if i in cents]

        # embed unique top-3 skills across variants
        uniq = sorted({s for v in VARIANTS for i in ids for s in topk[v][i]})
        svecs = dict(zip(uniq, emb.encode([skill_text(corpus[s]) for s in uniq])))

        # val/test split — same rule and seed as gate.py calibrate
        rng = random.Random(0)
        val = set(rng.sample(sorted(ids), max(1, int(len(ids) * 0.2))))
        test = [i for i in ids if i not in val]

        def mean_nd(picks: dict, subset: list) -> float:
            return float(np.mean([nd[picks[i]][i] for i in subset]))

        # L1: domain champion on val
        l1_pick = max(VARIANTS, key=lambda v: np.mean([nd[v][i] for i in val]))

        # L2a: agreement threshold calibrated on val (single vs fusion)
        def l2a_picks(t: float) -> dict:
            return {i: ("naive_skill" if agree[i] >= t else "hyskill") for i in ids}
        cand_ts = sorted({round(agree[i], 3) for i in val})
        best_t = max(cand_ts, key=lambda t: mean_nd(l2a_picks(t), sorted(val)))

        # L2b: uniform judge (S1-style) over each variant's top-1 / dcg-top-3
        def judge(i: str, v: str, k: int) -> float:
            sids = topk[v][i][:k]
            if not sids:
                return -1.0
            w = [1.0 / math.log2(r + 2) for r in range(len(sids))]
            return sum(wr * float(cents[i] @ svecs[s]) for wr, s in zip(w, sids)) / sum(w)
        l2b1 = {i: max(VARIANTS, key=lambda v: judge(i, v, 1)) for i in ids}
        l2b3 = {i: max(VARIANTS, key=lambda v: judge(i, v, 3)) for i in ids}
        oracle = {i: max(VARIANTS, key=lambda v: nd[v][i]) for i in ids}

        row = {"n_test": len(test), "cache_misses": miss,
               "L0": {v: round(float(np.mean([nd[v][i] for i in test])), 4)
                      for v in VARIANTS},
               "L1": {"pick": l1_pick,
                      "ndcg": round(mean_nd({i: l1_pick for i in ids}, test), 4)},
               "L2a": {"tau": best_t,
                       "ndcg": round(mean_nd(l2a_picks(best_t), test), 4),
                       "fusion_share": round(np.mean(
                           [l2a_picks(best_t)[i] == "hyskill" for i in test]), 3)},
               "L2b1": {"ndcg": round(mean_nd(l2b1, test), 4),
                        "picks": {v: sum(1 for i in test if l2b1[i] == v)
                                  for v in VARIANTS}},
               "L2b3": {"ndcg": round(mean_nd(l2b3, test), 4)},
               "ORACLE": round(mean_nd(oracle, test), 4)}
        report[ds] = row
        print(ds, json.dumps(row), flush=True)

    macro = {lvl: round(float(np.mean([
        report[d]["L0"]["naive_skill"] if lvl == "L0-skill"
        else report[d]["L1"]["ndcg"] if lvl == "L1"
        else report[d]["L2a"]["ndcg"] if lvl == "L2a"
        else report[d]["L2b1"]["ndcg"] if lvl == "L2b1"
        else report[d]["L2b3"]["ndcg"] if lvl == "L2b3"
        else report[d]["ORACLE"] for d in DOMAINS])), 4)
        for lvl in ["L0-skill", "L1", "L2a", "L2b1", "L2b3", "ORACLE"]}
    report["macro"] = macro
    Path("results/route_pilot.json").write_text(json.dumps(report, indent=1))
    print("MACRO", json.dumps(macro))
    print("PILOT-DONE")


if __name__ == "__main__":
    main()
