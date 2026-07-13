#!/usr/bin/env python
"""S1/S2 loading gate over precomputed retrieval results (Phase 2).

Subcommands:
  signals    Compute per-instance S1 (coverage: cos(hypothetical centroid,
             top-1 skill)) and S2 (gain: fraction of top-1 sentences NOT
             covered by the hypothetical) from cached generations. Zero LLM
             calls; embeddings computed on the fly (GPU, minutes).
  calibrate  Pick conservative taus on a validation split:
             tau1 = largest t s.t. precision(top1-is-wrong | S1<t) >= p_min;
             tau2 = largest t s.t. precision(bare-correct  | S2<t) >= p_min
             (needs --bare-eval for the second; otherwise tau2 disabled).
  apply      Write a gated retrieval file: blocked instances get retrieved=[]
             (SR-Agents' topk provider then injects nothing = bare behaviour).

Gate decision: BLOCK if S1 < tau1 (no suitable skill) OR S2 < tau2
(model already knows). Instances in the validation split are never counted
in test reporting downstream (their ids are recorded in the taus file).
"""

import argparse
import json
import math
import re
from pathlib import Path

_SENT = re.compile(r"[.\n;]+")
TAU_C = 0.6  # sentence-level "covered" similarity threshold for S2


def split_sentences(text: str, min_len: int = 15) -> list[str]:
    return [s.strip() for s in _SENT.split(text or "") if len(s.strip()) >= min_len]


def novelty_ratio(cand_sims_max: list[float]) -> float:
    """S2 = fraction of candidate sentences whose best match to the
    hypothetical is below TAU_C (i.e., content the model did not imagine)."""
    if not cand_sims_max:
        return 0.0
    return sum(1 for s in cand_sims_max if s < TAU_C) / len(cand_sims_max)


def pick_tau(values: list[float], is_positive: list[bool],
             p_min: float = 0.9) -> float | None:
    """Largest threshold t such that precision(positive | value < t) >= p_min.

    Returns None if no threshold achieves p_min (gate disabled).
    """
    pairs = sorted(zip(values, is_positive))
    best = None
    pos = 0
    for i, (v, p) in enumerate(pairs, start=1):
        pos += 1 if p else 0
        if pos / i >= p_min:
            best = v  # blocking everything strictly below the NEXT value
    return best


# --------------------------------------------------------------------- signals

def cmd_signals(a) -> None:
    import numpy as np
    from sragents.cli.retrieve import _build_query
    from sragents.corpus import load_corpus_dict, skill_text
    from hyskill.embedder import Embedder
    from hyskill.generator import HypotheticalGenerator, SKILL_TEMPLATE

    class _NoNet:
        def complete(self, *_a, **_k):
            raise RuntimeError("cache miss — signals must be cache-only")

    corpus = load_corpus_dict(a.corpus)
    instances = {i["instance_id"]: i for i in json.load(open(a.instances))}
    retrieval = json.load(open(a.retrieval))
    emb = Embedder(model_name=a.encoder)
    gen = HypotheticalGenerator(client=_NoNet(), k_samples=a.k,
                                template=SKILL_TEMPLATE, cache_dir=a.cache_dir,
                                model_tag=f"{a.model}|{SKILL_TEMPLATE[:20]}")

    out, misses = [], 0
    for rec in retrieval["results"]:
        iid = rec["instance_id"]
        if not rec["retrieved"]:
            continue
        top1 = rec["retrieved"][0]["skill_id"]
        inst = instances[iid]
        query = _build_query(inst)
        try:
            docs = gen.generate(query)
        except RuntimeError:
            misses += 1
            continue
        if not docs:
            misses += 1
            continue

        skill = corpus[top1]
        full = skill_text(skill)
        doc_vecs = emb.encode(docs)
        q_vec = emb.encode([query])[0]
        centroid = np.vstack([doc_vecs, q_vec[None, :]]).mean(axis=0)
        centroid /= max(np.linalg.norm(centroid), 1e-9)
        top1_vec = emb.encode([full])[0]
        s1 = float(centroid @ top1_vec)

        cand_sents = split_sentences(skill.get("content", ""))[:40]
        hyp_sents = [s for d in docs for s in split_sentences(d)][:60]
        if cand_sents and hyp_sents:
            cs = emb.encode(cand_sents)
            hs = emb.encode(hyp_sents)
            maxsims = (cs @ hs.T).max(axis=1).tolist()
            s2 = novelty_ratio(maxsims)
        else:
            s2 = 1.0  # nothing comparable: treat as novel (never skip)

        out.append({"instance_id": iid, "top1": top1, "S1": round(s1, 4),
                    "S2": round(s2, 4),
                    "rel_truth_wrong": top1 not in rec["gold_skill_ids"]})

    Path(a.out).write_text(json.dumps({"signals": out, "cache_misses": misses,
                                       "encoder": a.encoder}, indent=1))
    print(f"signals: {len(out)} instances, cache_misses={misses} -> {a.out}")


# ------------------------------------------------------------------ calibrate

def cmd_calibrate(a) -> None:
    import random
    sig = json.load(open(a.signals))["signals"]
    rng = random.Random(a.seed)
    ids = sorted(s["instance_id"] for s in sig)
    val_ids = set(rng.sample(ids, max(1, int(len(ids) * a.val_frac))))
    val = [s for s in sig if s["instance_id"] in val_ids]

    tau1 = pick_tau([s["S1"] for s in val],
                    [s["rel_truth_wrong"] for s in val], a.p_min)

    tau2 = None
    if a.bare_eval:
        bare = {d["instance_id"]: d["correct"]
                for d in json.load(open(a.bare_eval))["details"]}
        vb = [s for s in val if s["instance_id"] in bare]
        tau2 = pick_tau([s["S2"] for s in vb],
                        [bare[s["instance_id"]] for s in vb], a.p_min)

    res = {"tau1": tau1, "tau2": tau2, "p_min": a.p_min,
           "val_ids": sorted(val_ids), "n_val": len(val)}
    Path(a.out).write_text(json.dumps(res, indent=1))
    n1 = sum(1 for s in sig if tau1 is not None and s["S1"] < tau1)
    n2 = sum(1 for s in sig if tau2 is not None and s["S2"] < tau2)
    print(f"tau1={tau1} (would block {n1}/{len(sig)})  "
          f"tau2={tau2} (would skip {n2}/{len(sig)}) -> {a.out}")


# ---------------------------------------------------------------------- apply

def cmd_apply(a) -> None:
    sig = {s["instance_id"]: s for s in json.load(open(a.signals))["signals"]}
    taus = json.load(open(a.taus))
    tau1, tau2 = taus.get("tau1"), taus.get("tau2")
    retrieval = json.load(open(a.retrieval))

    blocked = skipped = kept = 0
    for rec in retrieval["results"]:
        s = sig.get(rec["instance_id"])
        if s is None:
            continue
        if tau1 is not None and s["S1"] < tau1:
            rec["retrieved"] = []
            blocked += 1
        elif tau2 is not None and s["S2"] < tau2:
            rec["retrieved"] = []
            skipped += 1
        else:
            kept += 1
    retrieval.setdefault("metadata", {})["gate"] = {
        "tau1": tau1, "tau2": tau2, "blocked": blocked,
        "skipped": skipped, "kept": kept}
    Path(a.out).write_text(json.dumps(retrieval))
    print(f"gated: kept={kept} blocked={blocked} skipped={skipped} -> {a.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("signals")
    s.add_argument("--retrieval", required=True)
    s.add_argument("--corpus", required=True)
    s.add_argument("--instances", required=True)
    s.add_argument("--cache-dir", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--k", type=int, default=4)
    s.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_signals)

    c = sub.add_parser("calibrate")
    c.add_argument("--signals", required=True)
    c.add_argument("--bare-eval", default=None)
    c.add_argument("--val-frac", type=float, default=0.2)
    c.add_argument("--p-min", type=float, default=0.9)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", required=True)
    c.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("apply")
    p.add_argument("--signals", required=True)
    p.add_argument("--taus", required=True)
    p.add_argument("--retrieval", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_apply)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
