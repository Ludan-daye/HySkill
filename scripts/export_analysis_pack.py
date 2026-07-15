#!/usr/bin/env python
"""Export a per-model ANALYSIS PACK into community-results/<tag>/.

Produces small, GitHub-committable, analysis-ready files (gzip JSONL):

  retrieval_top10.jsonl.gz   one row per (instance x variant): gold ids,
                             top-10 [skill_id, score, is_gold], nDCG@10
  gating_per_instance.jsonl.gz  one row per instance: S1/S2, top1,
                             retrieval-truth flag, taus, gate decision,
                             per-arm correctness (bare/always/gated[/select])
  imagination_samples.jsonl.gz  10 instances/domain (seed 0, fixed across
                             models): query, K=4 imagined SKILL.md texts
                             (skill template), naive_skill top-3 with
                             names/descriptions, gold — for qualitative
                             imagination-vs-match analysis
  MANIFEST.md                what each file is + row counts + pandas recipe

Raw full artifacts (top-50 rankings, answer jsonl, logs, full 47k-doc
imagination cache) stay on the run server; this pack is the analyzable core.

Usage (run on the machine that holds results/multimodel/<tag>):
  .venv/bin/python scripts/export_analysis_pack.py <tag> <model-name-for-cache>
"""

import gzip
import json
import math
import random
import sys
from pathlib import Path

DOMAINS = ["theoremqa", "logicbench", "medcalcbench", "champ", "bigcodebench"]
RULE_DOMAINS = ["theoremqa", "logicbench", "medcalcbench", "champ"]
VARIANTS = ["naive_sentence", "naive_passage", "naive_skill",
            "hyskill", "two_stage", "llm_rerank", "routed"]
SRA = "external/SR-Agents"
SAMPLES_PER_DOMAIN = 10


def ndcg10(gold: set, ranked: list) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, s in enumerate(ranked[:10]) if s in gold)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), 10)))
    return dcg / idcg if idcg > 0 else 0.0


def jl(path: Path, rows: list) -> int:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> None:
    tag = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else tag
    res = Path(f"results/multimodel/{tag}")
    out = Path(f"community-results/{tag}")
    out.mkdir(parents=True, exist_ok=True)

    from sragents.cli.retrieve import _build_query
    from sragents.corpus import load_corpus_dict
    from hyskill.generator import (HypotheticalGenerator, PASSAGE_TEMPLATE,
                                   SENTENCE_TEMPLATE, SKILL_TEMPLATE)

    class _NoNet:
        def complete(self, *_a, **_k):
            raise RuntimeError("cache-only")

    corpus = load_corpus_dict(f"{SRA}/data/bench/corpus/corpus.json")
    counts = {}

    # ---------- retrieval_top10 ----------
    rows = []
    common_by_domain = {}
    for ds in DOMAINS:
        per_variant = {}
        for v in VARIANTS:
            p = res / f"{ds}-{v}.json"
            if p.exists():
                per_variant[v] = {r["instance_id"]: r for r in
                                  json.loads(p.read_text())["results"]}
        if not per_variant:
            continue
        base = per_variant.get("naive_skill") or next(iter(per_variant.values()))
        common_by_domain[ds] = sorted(base)
        for v, recs in per_variant.items():
            for iid, r in recs.items():
                gold = set(r["gold_skill_ids"])
                ranked = [x["skill_id"] for x in r["retrieved"]]
                rows.append({
                    "instance_id": iid, "domain": ds, "variant": v,
                    "gold": sorted(gold),
                    "top10": [{"skill_id": x["skill_id"],
                               "score": round(float(x["score"]), 5),
                               "is_gold": x["skill_id"] in gold}
                              for x in r["retrieved"][:10]],
                    "ndcg10": round(ndcg10(gold, ranked), 4)})
    counts["retrieval_top10.jsonl.gz"] = jl(out / "retrieval_top10.jsonl.gz", rows)

    # ---------- gating_per_instance ----------
    rows = []
    for ds in RULE_DOMAINS:
        sig_p = res / f"{ds}-signals.json"
        if not sig_p.exists():
            continue
        sig = {s["instance_id"]: s for s in
               json.loads(sig_p.read_text())["signals"]}
        taus = json.loads((res / f"{ds}-taus.json").read_text())
        val_ids = set(taus.get("val_ids", []))
        arms = {}
        for arm in ["bare", "always", "gated", "select", "oracle", "always_rerank", "select_bm25"]:
            p = res / f"{ds}-{arm}.eval.json"
            if p.exists():
                arms[arm] = {d["instance_id"]: bool(d["correct"])
                             for d in json.loads(p.read_text())["details"]}
        t1, t2 = taus.get("tau1"), taus.get("tau2")
        for iid, s in sig.items():
            blocked = (t1 is not None and s["S1"] < t1) or \
                      (t2 is not None and s["S2"] < t2)
            row = {"instance_id": iid, "domain": ds,
                   "S1": s["S1"], "S2": s["S2"], "top1": s["top1"],
                   "retrieval_wrong": s["rel_truth_wrong"],
                   "tau1": t1, "tau2": t2, "gate_blocked": blocked,
                   "in_calibration_split": iid in val_ids}
            for arm, m in arms.items():
                row[f"correct_{arm}"] = m.get(iid)
            rows.append(row)
    counts["gating_per_instance.jsonl.gz"] = jl(
        out / "gating_per_instance.jsonl.gz", rows)

    # ---------- imagination_samples (ALL 3 templates) ----------
    gens = {name: HypotheticalGenerator(
                client=_NoNet(), k_samples=4, template=tpl,
                cache_dir="results/hyp_cache",
                model_tag=f"{model}|{tpl[:20]}")
            for name, tpl in [("sentence", SENTENCE_TEMPLATE),
                              ("passage", PASSAGE_TEMPLATE),
                              ("skill", SKILL_TEMPLATE)]}
    rows = []
    for ds in DOMAINS:
        if ds not in common_by_domain:
            continue
        instances = {i["instance_id"]: i for i in
                     json.loads(Path(f"{SRA}/data/bench/instances/{ds}.json").read_text())}
        ids = common_by_domain[ds]
        rng = random.Random(0)                      # fixed across ALL models
        picks = rng.sample(ids, min(SAMPLES_PER_DOMAIN, len(ids)))
        variant_recs = {}
        for v in VARIANTS:
            p = res / f"{ds}-{v}.json"
            if p.exists():
                variant_recs[v] = {r["instance_id"]: r for r in
                                   json.loads(p.read_text())["results"]}
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
            for v, recs in variant_recs.items():
                rec = recs.get(iid)
                if not rec:
                    continue
                gold = set(rec["gold_skill_ids"])
                tops[v] = []
                for x in rec["retrieved"][:3]:
                    sk = corpus.get(x["skill_id"], {})
                    tops[v].append({"skill_id": x["skill_id"],
                                    "name": sk.get("name", ""),
                                    "description": (sk.get("description", "") or "")[:300],
                                    "score": round(float(x["score"]), 5),
                                    "is_gold": x["skill_id"] in gold})
            rows.append({"instance_id": iid, "domain": ds, "query": q,
                         "imaginations": imag,           # {sentence|passage|skill: [4 texts]}
                         "top3_per_variant": tops,       # incl. routed/rerank when present
                         "gold": sorted(gold)})
    counts["imagination_samples.jsonl.gz"] = jl(
        out / "imagination_samples.jsonl.gz", rows)

    # ---------- router_decisions + metrics_flat ----------
    router = {}
    for ds in DOMAINS:
        p = res / f"{ds}-routed.json"
        if p.exists():
            router[ds] = json.loads(p.read_text()).get("metadata", {}).get("router", {})
    (out / "router_decisions.json").write_text(
        json.dumps(router, ensure_ascii=False, indent=1))
    counts["router_decisions.json"] = len(router)

    rows = []
    for ds in DOMAINS:
        for v in VARIANTS + ["bm25"]:
            p = res / f"{ds}-{v}.json"
            if p.exists():
                for metric, val in (json.loads(p.read_text()).get("metrics") or {}).items():
                    rows.append({"domain": ds, "method": v, "metric": metric,
                                 "value": round(float(val), 5)})
        for arm in ["bare", "always", "gated", "select", "oracle", "always_rerank", "select_bm25"]:
            p = res / f"{ds}-{arm}.eval.json"
            if p.exists():
                m = json.loads(p.read_text())["metrics"]
                rows.append({"domain": ds, "method": f"arm:{arm}",
                             "metric": "accuracy", "value": round(m["accuracy"], 5)})
                rows.append({"domain": ds, "method": f"arm:{arm}",
                             "metric": "n", "value": m["total"]})
    counts["metrics_flat.jsonl.gz"] = jl(out / "metrics_flat.jsonl.gz", rows)

    # ---------- MANIFEST ----------
    manifest = f"""# {tag} 分析数据包清单（自动生成）

| 文件 | 行数 | 内容 |
|---|---|---|
| retrieval_top10.jsonl.gz | {counts['retrieval_top10.jsonl.gz']} | 每行=（实例×变体）：金标、top-10（id/分数/是否金标）、逐题 nDCG@10。变体含 5 想象变体 + routed{'+llm_rerank' if any((res / f'{d}-llm_rerank.json').exists() for d in DOMAINS) else '（本模型无重排臂）'} |
| gating_per_instance.jsonl.gz | {counts['gating_per_instance.jsonl.gz']} | 每行=实例：S1/S2、top1、检索是否错、τ、门控是否拦截、是否标定集、各臂对错（bare/always/gated{'/select' if counts['gating_per_instance.jsonl.gz'] and 'select' in str(rows[:1]) else ''}） |
| imagination_samples.jsonl.gz | {counts['imagination_samples.jsonl.gz']} | 每域固定 10 题（seed 0，跨模型同题可比）：查询原文、**三种模板 × K=4 份想象全文**、每个变体（含 routed/rerank）的 top-3（名称/简介/分数/命中）、金标 |
| router_decisions.json | {counts['router_decisions.json']} 域 | 路由决策账：每域选中的变体 + 全部变体的验证集 nDCG 比分 + 切分参数 |
| metrics_flat.jsonl.gz | {counts['metrics_flat.jsonl.gz']} | **全部分数拍平**：（域 × 方法 × 指标）一行一个数——检索 Recall@1/5/10/50、nDCG@k 全量 + 各臂 accuracy/n |
| summary.json | — | 聚合指标（检索/路由/门控/成本审计） |

## 用 pandas 读取

```python
import pandas as pd
top10 = pd.read_json("retrieval_top10.jsonl.gz", lines=True)
gate  = pd.read_json("gating_per_instance.jsonl.gz", lines=True)
imag  = pd.read_json("imagination_samples.jsonl.gz", lines=True)
```

全量原始件（top-50 榜单、做题 jsonl、日志、完整想象缓存）留存跑批服务器 `results/multimodel/{tag}/`，见本目录 README。
"""
    (out / "MANIFEST.md").write_text(manifest)
    print(json.dumps(counts, indent=1))
    print(f"pack written to {out}/")


if __name__ == "__main__":
    main()
