#!/usr/bin/env python
"""Export the qwen3.5-4b MAIN-EXPERIMENT reference pack (Phase 1 + 2 + 2c)
into community-results/qwen3.5-4b-reference/ — same shapes as the fleet
analysis packs, but covering the full 9-method grid and all 7 answering arms.

Files:
  retrieval_top10.jsonl.gz     (instance x method) for ALL 9 Phase-1 methods
                               + 5 routed files: gold, top-10, nDCG@10
  gating_per_instance.jsonl.gz fixed-gate + routed-gate signals per instance,
                               taus, decisions, correctness for 7 arms
                               (bare/always/gated/select/oracle/always_r/gated_r)
  imagination_samples.jsonl.gz same fixed 50 instances as fleet packs,
                               3 templates x K=4 raw texts, top-3 per method
  router_decisions.json        per-domain routed pick + val scores
  metrics_flat.jsonl.gz        every (domain x method x metric) number
  significance.json            copy of phase2 significance (incl. pooled)
  MANIFEST.md

Run on the lab server: .venv/bin/python scripts/export_reference_pack.py
"""

import gzip
import json
import math
import random
from pathlib import Path

DOMAINS = ["theoremqa", "logicbench", "medcalcbench", "champ", "bigcodebench"]
RULE_DOMAINS = ["theoremqa", "logicbench", "medcalcbench", "champ"]
METHODS = ["bm25", "dense", "hybrid", "llm_rerank", "naive_sentence",
           "naive_passage", "naive_skill", "hyskill", "two_stage"]
ARMS = ["bare", "always", "gated", "select", "oracle", "always_r", "gated_r"]
SRA = "external/SR-Agents"
MODEL = "qwen3.5-4b"
OUT = Path("community-results/qwen3.5-4b-reference")
SAMPLES_PER_DOMAIN = 10


def ndcg10(gold, ranked):
    dcg = sum(1.0 / math.log2(i + 2) for i, s in enumerate(ranked[:10]) if s in gold)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), 10)))
    return dcg / idcg if idcg > 0 else 0.0


def jl(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def main():
    from sragents.cli.retrieve import _build_query
    from sragents.corpus import load_corpus_dict
    from hyskill.generator import (HypotheticalGenerator, PASSAGE_TEMPLATE,
                                   SENTENCE_TEMPLATE, SKILL_TEMPLATE)

    class _NoNet:
        def complete(self, *_a, **_k):
            raise RuntimeError("cache-only")

    OUT.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus_dict(f"{SRA}/data/bench/corpus/corpus.json")
    counts = {}

    def load_results(path):
        return {r["instance_id"]: r for r in json.loads(Path(path).read_text())["results"]}

    # retrieval_top10: 9 methods (phase1) + routed (phase2)
    rows = []
    common_by_domain = {}
    method_files = {}
    for ds in DOMAINS:
        per = {}
        for m in METHODS:
            p = Path(f"results/phase1/{ds}-{m}.json")
            if p.exists():
                per[m] = load_results(p)
        rp = Path(f"results/phase2/{ds}-routed.json")
        if rp.exists():
            per["routed"] = load_results(rp)
        method_files[ds] = per
        common_by_domain[ds] = sorted(per["naive_skill"])
        for m, recs in per.items():
            for iid, r in recs.items():
                gold = set(r["gold_skill_ids"])
                ranked = [x["skill_id"] for x in r["retrieved"]]
                rows.append({"instance_id": iid, "domain": ds, "variant": m,
                             "gold": sorted(gold),
                             "top10": [{"skill_id": x["skill_id"],
                                        "score": round(float(x["score"]), 5),
                                        "is_gold": x["skill_id"] in gold}
                                       for x in r["retrieved"][:10]],
                             "ndcg10": round(ndcg10(gold, ranked), 4)})
    counts["retrieval_top10.jsonl.gz"] = jl(OUT / "retrieval_top10.jsonl.gz", rows)

    # gating_per_instance: fixed + routed signals, 7 arms
    rows = []
    for ds in RULE_DOMAINS:
        sigs = {}
        for kind, sp, tp in [("fixed", f"results/phase2/{ds}-signals.json",
                              f"results/phase2/{ds}-taus.json"),
                             ("routed", f"results/phase2/{ds}-routed-signals.json",
                              f"results/phase2/{ds}-routed-taus.json")]:
            if Path(sp).exists():
                sigs[kind] = ({s["instance_id"]: s for s in
                               json.loads(Path(sp).read_text())["signals"]},
                              json.loads(Path(tp).read_text()))
        arms = {}
        for arm in ARMS:
            p = Path(f"results/phase2/{ds}-{arm}.eval.json")
            if p.exists():
                arms[arm] = {d["instance_id"]: bool(d["correct"])
                             for d in json.loads(p.read_text())["details"]}
        ids = sorted(arms["bare"])
        for iid in ids:
            row = {"instance_id": iid, "domain": ds}
            for kind, (sm, taus) in sigs.items():
                s = sm.get(iid)
                if s:
                    t1, t2 = taus.get("tau1"), taus.get("tau2")
                    row[f"S1_{kind}"] = s["S1"]
                    row[f"S2_{kind}"] = s["S2"]
                    row[f"top1_{kind}"] = s["top1"]
                    row[f"retrieval_wrong_{kind}"] = s["rel_truth_wrong"]
                    row[f"gate_blocked_{kind}"] = (
                        (t1 is not None and s["S1"] < t1) or
                        (t2 is not None and s["S2"] < t2))
                    row[f"in_calibration_split_{kind}"] = \
                        iid in set(taus.get("val_ids", []))
            for arm, m in arms.items():
                row[f"correct_{arm}"] = m.get(iid)
            rows.append(row)
    counts["gating_per_instance.jsonl.gz"] = jl(
        OUT / "gating_per_instance.jsonl.gz", rows)

    # imagination_samples: same fixed picks (seed 0 over naive_skill ids)
    gens = {name: HypotheticalGenerator(
                client=_NoNet(), k_samples=4, template=tpl,
                cache_dir="results/hyp_cache",
                model_tag=f"{MODEL}|{tpl[:20]}")
            for name, tpl in [("sentence", SENTENCE_TEMPLATE),
                              ("passage", PASSAGE_TEMPLATE),
                              ("skill", SKILL_TEMPLATE)]}
    rows = []
    for ds in DOMAINS:
        instances = {i["instance_id"]: i for i in
                     json.loads(Path(f"{SRA}/data/bench/instances/{ds}.json").read_text())}
        ids = common_by_domain[ds]
        rng = random.Random(0)
        picks = rng.sample(ids, min(SAMPLES_PER_DOMAIN, len(ids)))
        for iid in picks:
            q = _build_query(instances[iid])
            imag = {}
            for name, g in gens.items():
                try:
                    imag[name] = g.generate(q)
                except RuntimeError:
                    imag[name] = []
            gold = set()
            tops = {}
            for m, recs in method_files[ds].items():
                rec = recs.get(iid)
                if not rec:
                    continue
                gold = set(rec["gold_skill_ids"])
                tops[m] = [{"skill_id": x["skill_id"],
                            "name": corpus.get(x["skill_id"], {}).get("name", ""),
                            "description": (corpus.get(x["skill_id"], {}).get("description", "") or "")[:300],
                            "score": round(float(x["score"]), 5),
                            "is_gold": x["skill_id"] in gold}
                           for x in rec["retrieved"][:3]]
            rows.append({"instance_id": iid, "domain": ds, "query": q,
                         "imaginations": imag, "top3_per_variant": tops,
                         "gold": sorted(gold)})
    counts["imagination_samples.jsonl.gz"] = jl(
        OUT / "imagination_samples.jsonl.gz", rows)

    # router_decisions + metrics_flat + significance copy
    router = {}
    for ds in DOMAINS:
        p = Path(f"results/phase2/{ds}-routed.json")
        if p.exists():
            router[ds] = json.loads(p.read_text()).get("metadata", {}).get("router", {})
    (OUT / "router_decisions.json").write_text(json.dumps(router, indent=1))
    counts["router_decisions.json"] = len(router)

    rows = []
    for ds in DOMAINS:
        for m, recs_path in [(m, f"results/phase1/{ds}-{m}.json") for m in METHODS] + \
                            [("routed", f"results/phase2/{ds}-routed.json")]:
            p = Path(recs_path)
            if p.exists():
                for metric, val in (json.loads(p.read_text()).get("metrics") or {}).items():
                    rows.append({"domain": ds, "method": m, "metric": metric,
                                 "value": round(float(val), 5)})
        for arm in ARMS:
            p = Path(f"results/phase2/{ds}-{arm}.eval.json")
            if p.exists():
                mm = json.loads(p.read_text())["metrics"]
                rows.append({"domain": ds, "method": f"arm:{arm}",
                             "metric": "accuracy", "value": round(mm["accuracy"], 5)})
                rows.append({"domain": ds, "method": f"arm:{arm}",
                             "metric": "n", "value": mm["total"]})
    counts["metrics_flat.jsonl.gz"] = jl(OUT / "metrics_flat.jsonl.gz", rows)

    sig = Path("results/phase2/significance.json")
    if sig.exists():
        (OUT / "significance.json").write_text(sig.read_text())
        counts["significance.json"] = "copied"

    (OUT / "MANIFEST.md").write_text(f"""# qwen3.5-4b 主实验参考包（自动生成）

主实验（Phase 1 全量 9 方法 + Phase 2 五臂 + Phase 2c 路由臂）的逐题级数据。

| 文件 | 规模 | 内容 |
|---|---|---|
| retrieval_top10.jsonl.gz | {counts['retrieval_top10.jsonl.gz']} 行 | （实例 × 方法）×{{9 方法 + routed}}：金标、top-10、逐题 nDCG@10 |
| gating_per_instance.jsonl.gz | {counts['gating_per_instance.jsonl.gz']} 行 | 固定门+路由门双份 S1/S2/τ/拦截决定/标定集标记 + **7 臂逐题对错**（bare/always/gated/select/oracle/always_r/gated_r） |
| imagination_samples.jsonl.gz | {counts['imagination_samples.jsonl.gz']} 行 | 与舰队包同一批 50 题：3 模板 × K=4 想象全文 + 各方法 top-3 |
| router_decisions.json | {counts['router_decisions.json']} 域 | 路由决策与验证集比分 |
| metrics_flat.jsonl.gz | {counts['metrics_flat.jsonl.gz']} 行 | 全部（域 × 方法 × 指标）数字拍平 |
| significance.json | — | Phase 2 全部配对 bootstrap（分域 + 合并 + 剔标定集版） |

pandas: `pd.read_json("<file>.jsonl.gz", lines=True)`
""")
    print(json.dumps(counts, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
