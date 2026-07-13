#!/usr/bin/env bash
# Phase 0: retrieval-stage comparison on SRA-Bench (5 datasets, ToolQA deferred).
# Usage: MODEL=... API_BASE=... [PILOT=1] ./scripts/run_phase0.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SRA=external/SR-Agents
CORPUS=$SRA/data/bench/corpus/corpus.json
DATASETS=(theoremqa logicbench medcalcbench champ bigcodebench)
mkdir -p results/retrieval results/hyp_cache results/emb_cache

for DS in "${DATASETS[@]}"; do
  INST=$SRA/data/bench/instances/$DS.json
  if [[ "${PILOT:-0}" == "1" ]]; then
    .venv/bin/python -c "
import json; d=json.load(open('$INST')); json.dump(d[:20], open('results/pilot_$DS.json','w'))"
    INST=results/pilot_$DS.json
  fi
  for R in bm25 bge hybrid; do
    .venv/bin/sragents retrieve --retriever $R \
      --corpus $CORPUS --instances $INST \
      --output results/retrieval/$DS-$R.json --top-k 50
  done
  .venv/bin/sragents --plugin hyskill.plugin retrieve --retriever naive_hyde \
    --retriever-arg model="$MODEL" --retriever-arg api_base="$API_BASE" \
    --retriever-arg k_samples=4 --retriever-arg cache_dir=results/hyp_cache \
    --retriever-arg emb_cache_dir=results/emb_cache \
    --corpus $CORPUS --instances $INST \
    --output results/retrieval/$DS-naive_hyde.json --top-k 50
  .venv/bin/sragents --plugin hyskill.plugin retrieve --retriever hyskill \
    --retriever-arg corpus_path=$CORPUS \
    --retriever-arg model="$MODEL" --retriever-arg api_base="$API_BASE" \
    --retriever-arg k_samples=4 --retriever-arg cache_dir=results/hyp_cache \
    --retriever-arg emb_cache_dir=results/emb_cache \
    --corpus $CORPUS --instances $INST \
    --output results/retrieval/$DS-hyskill.json --top-k 50
done
.venv/bin/python scripts/analyze.py results/retrieval
