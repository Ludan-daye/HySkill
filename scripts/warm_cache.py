#!/usr/bin/env python
"""Parallel hypothetical-document cache warmer.

Pre-fills results/hyp_cache with all (template x query x K) generations using
a thread pool, so subsequent `sragents retrieve` runs are 100% cache hits.
Cache keys are identical to hyskill.plugin._generator's (same model_tag,
temperature, template), and queries are built with SR-Agents' own
_build_query so they match the retrieve CLI byte-for-byte.

Resumable: already-cached items are skipped for free.

Usage:
  .venv/bin/python scripts/warm_cache.py \
      --instances external/SR-Agents/data/bench/instances/*.json \
      --model qwen3.5-4b --api-base http://localhost:8311/v1 \
      --no-think --workers 32
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from sragents.cli.retrieve import _build_query

from hyskill.generator import (HypotheticalGenerator, OpenAIClient,
                               PASSAGE_TEMPLATE, SENTENCE_TEMPLATE,
                               SKILL_TEMPLATE)

TEMPLATES = {"passage": PASSAGE_TEMPLATE, "skill": SKILL_TEMPLATE,
             "sentence": SENTENCE_TEMPLATE}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="+", required=True)
    ap.add_argument("--templates", default="passage,skill,sentence")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-base", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--cache-dir", default="results/hyp_cache")
    ap.add_argument("--no-think", action="store_true")
    args = ap.parse_args()

    queries: list[str] = []
    for path in args.instances:
        for inst in json.load(open(path)):
            if inst.get("skill_annotations"):
                queries.append(_build_query(inst))
    print(f"queries: {len(queries)}", flush=True)

    client = OpenAIClient(model=args.model, api_base=args.api_base,
                          no_think=args.no_think)
    jobs = []
    for name in args.templates.split(","):
        tpl = TEMPLATES[name]
        gen = HypotheticalGenerator(
            client=client, k_samples=args.k, temperature=args.temperature,
            template=tpl, cache_dir=args.cache_dir,
            model_tag=f"{args.model}|{tpl[:20]}")
        jobs += [(gen, q) for q in queries]
    print(f"jobs: {len(jobs)} (= {len(queries)} x {len(args.templates.split(','))})",
          flush=True)

    done = empty = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(g.generate, q) for g, q in jobs]
        for fut in as_completed(futures):
            docs = fut.result()
            done += 1
            if not docs:
                empty += 1
            if done % 500 == 0:
                print(f"progress {done}/{len(futures)} empty={empty}", flush=True)
    print(f"WARMUP-DONE jobs={done} empty={empty}", flush=True)


if __name__ == "__main__":
    main()
