#!/usr/bin/env bash
# Phase 2 part A: bare arm (no-skill baseline) + gate signals for 4 rule-based
# domains. Bare-eval feeds tau2 calibration before part B.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/mnt/d5f4cfb6-8afe-40a4-8650-2965046cd208/hf_cache}
export HF_HUB_OFFLINE=1
export OPENAI_API_KEY=EMPTY
SRA=external/SR-Agents
CORPUS=$SRA/data/bench/corpus/corpus.json
MODEL=qwen3.5-4b
API=http://localhost:8311/v1
mkdir -p results/phase2

for DS in theoremqa logicbench medcalcbench champ; do
  INST=$SRA/data/bench/instances/$DS.json
  echo "### $DS bare-infer"
  .venv/bin/sragents infer --instances $INST \
    --output results/phase2/$DS-bare.jsonl \
    --model $MODEL --api-base $API \
    --provider none --engine direct \
    --workers 48 --max-tokens 2048 --label bare
  echo "### $DS bare-eval"
  .venv/bin/sragents evaluate --input results/phase2/$DS-bare.jsonl \
    --instances $INST --output results/phase2/$DS-bare.eval.json
  echo "### $DS signals"
  .venv/bin/python scripts/gate.py signals \
    --retrieval results/phase1/$DS-naive_skill.json \
    --corpus $CORPUS --instances $INST \
    --cache-dir results/hyp_cache --model $MODEL --k 4 \
    --out results/phase2/$DS-signals.json
done
echo PHASE2A-DONE
