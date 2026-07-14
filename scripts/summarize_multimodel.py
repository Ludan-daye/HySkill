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

CHARS_PER_TOKEN = 3.8  # rough English estimate; same estimator everywhere


def cost_audit(model: str, sample_n: int = 150):
    """Measured per-query token budget from THIS run's actual artifacts:
    imagination templates (prompt + cached generation lengths, K=4) and the
    rerank prompt (instruction + query + 50 candidate lines). Cost rule:
    every method comparison that goes into the paper must carry the
    same-batch measured cost from this block."""
    try:
        import hashlib
        import random
        from sragents.cli.retrieve import _build_query
        from sragents.corpus import load_corpus_dict
        import sragents.retrieve.llm_rerank as lr
        from hyskill.generator import (PASSAGE_TEMPLATE, SENTENCE_TEMPLATE,
                                       SKILL_TEMPLATE)
    except ImportError:
        return None

    def key(tag, temp, template, query, i):
        raw = json.dumps([tag, temp, template, query, i])
        return hashlib.sha256(raw.encode()).hexdigest()

    queries = []
    for ds in ["theoremqa", "logicbench", "medcalcbench", "champ", "bigcodebench"]:
        p = Path(f"external/SR-Agents/data/bench/instances/{ds}.json")
        if not p.exists():
            continue
        for inst in json.loads(p.read_text()):
            if inst.get("skill_annotations"):
                queries.append(_build_query(inst))
    if not queries:
        return None
    random.Random(0).shuffle(queries)
    sample = queries[:sample_n]
    T = CHARS_PER_TOKEN
    cost = {"note": f"tokens ~= chars/{T}; K=4; per fresh query",
            "avg_query_tok": round(sum(len(q) for q in sample) / len(sample) / T)}

    cache = Path("results/hyp_cache")
    for name, tpl in [("skill", SKILL_TEMPLATE), ("passage", PASSAGE_TEMPLATE),
                      ("sentence", SENTENCE_TEMPLATE)]:
        outs, plens = [], []
        for q in sample:
            plens.append(len(tpl.format(q=q)))
            for i in range(4):
                f = cache / (key(f"{model}|{tpl[:20]}", 0.7, tpl, q, i) + ".txt")
                if f.exists():
                    outs.append(len(f.read_text()))
        if outs:
            pin = round(sum(plens) / len(plens) / T)
            pout = round(sum(outs) / len(outs) / T)
            cost[f"imagine_{name}"] = {
                "in_per_gen": pin, "out_per_gen": pout, "n_sampled": len(outs),
                "total_K4": 4 * (pin + pout)}

    try:
        corpus = load_corpus_dict("external/SR-Agents/data/bench/corpus/corpus.json")
        skills = random.Random(1).sample(list(corpus.values()),
                                         min(500, len(corpus)))
        cand = sum(len(f"[{i}] {s.get('name', '')}: {s.get('description', '')}")
                   for i, s in enumerate(skills)) / len(skills) / T
        instr = len(lr._RERANK_PROMPT.format(query="", candidates="")) / T
        cost["rerank"] = {"in": round(instr + cost["avg_query_tok"] + 50 * cand),
                          "out_typical": 200, "candidates": 50}
    except Exception:
        pass
    cost["gate_and_router"] = {"extra_llm_tokens": 0,
                               "note": "signals/routing reuse cached imaginations"}
    return cost


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

    cost = cost_audit(model)
    if cost:
        out["cost"] = cost

    dst = Path("community-results") / tag
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "summary.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {dst / 'summary.json'}")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
