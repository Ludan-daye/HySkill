#!/usr/bin/env bash
# Phase 2c: routed-retrieval P2 arms. For each rule domain: route among the
# 5 Phase-1 imagination variants on the 20% val split, recompute gate
# signals/taus against the routed top-1, then run always_r + gated_r arms.
# Compare against phase2's naive_skill-based always/gated to quantify the
# routing dividend on end-task accuracy.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/mnt/d5f4cfb6-8afe-40a4-8650-2965046cd208/hf_cache}
export HF_HUB_OFFLINE=1
export OPENAI_API_KEY=EMPTY
SRA=external/SR-Agents
CORPUS=$SRA/data/bench/corpus/corpus.json
MODEL=qwen3.5-4b
API=http://localhost:8311/v1

for DS in theoremqa logicbench medcalcbench champ; do
  INST=$SRA/data/bench/instances/$DS.json
  echo "### $DS route"
  .venv/bin/python scripts/route_variant.py --dir results/phase1 --dataset $DS \
    --out results/phase2/$DS-routed.json
  echo "### $DS routed-signals"
  .venv/bin/python scripts/gate.py signals \
    --retrieval results/phase2/$DS-routed.json \
    --corpus $CORPUS --instances $INST \
    --cache-dir results/hyp_cache --model $MODEL --k 4 \
    --out results/phase2/$DS-routed-signals.json
  echo "### $DS routed-calibrate+apply"
  .venv/bin/python scripts/gate.py calibrate \
    --signals results/phase2/$DS-routed-signals.json \
    --bare-eval results/phase2/$DS-bare.eval.json \
    --out results/phase2/$DS-routed-taus.json
  .venv/bin/python scripts/gate.py apply \
    --signals results/phase2/$DS-routed-signals.json \
    --taus results/phase2/$DS-routed-taus.json \
    --retrieval results/phase2/$DS-routed.json \
    --out results/phase2/$DS-routed-gated.json

  for ARM in always_r gated_r; do
    SRC=results/phase2/$DS-routed.json
    [[ $ARM == gated_r ]] && SRC=results/phase2/$DS-routed-gated.json
    echo "### $DS $ARM-infer"
    .venv/bin/sragents infer --instances $INST \
      --output results/phase2/$DS-$ARM.jsonl \
      --model $MODEL --api-base $API \
      --provider topk --provider-arg source=$SRC --provider-arg k=1 \
      --engine direct --workers 48 --max-tokens 2048 --label $ARM
    echo "### $DS $ARM-eval"
    .venv/bin/sragents evaluate --input results/phase2/$DS-$ARM.jsonl \
      --instances $INST --output results/phase2/$DS-$ARM.eval.json
  done
done
echo PHASE2C-DONE
