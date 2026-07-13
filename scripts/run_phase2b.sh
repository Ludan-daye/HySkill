#!/usr/bin/env bash
# Phase 2 part B: remaining four arms (always / gated / oracle / model-select)
# for 4 rule-based domains. Requires part A outputs + calibrated taus + gated
# retrieval files (scripts/gate.py calibrate + apply).
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/mnt/d5f4cfb6-8afe-40a4-8650-2965046cd208/hf_cache}
export HF_HUB_OFFLINE=1
export OPENAI_API_KEY=EMPTY
SRA=external/SR-Agents
CORPUS=$SRA/data/bench/corpus/corpus.json
MODEL=qwen3.5-4b
API=http://localhost:8311/v1

run_arm() {  # DS ARM PROVIDER EXTRA_PROVIDER_ARGS...
  local DS=$1 ARM=$2 PROVIDER=$3; shift 3
  local INST=$SRA/data/bench/instances/$DS.json
  echo "### $DS $ARM-infer"
  .venv/bin/sragents infer --instances $INST \
    --output results/phase2/$DS-$ARM.jsonl \
    --model $MODEL --api-base $API \
    --provider $PROVIDER "$@" --engine direct \
    --workers 48 --max-tokens 2048 --label $ARM
  echo "### $DS $ARM-eval"
  .venv/bin/sragents evaluate --input results/phase2/$DS-$ARM.jsonl \
    --instances $INST --output results/phase2/$DS-$ARM.eval.json
}

for DS in theoremqa logicbench medcalcbench champ; do
  run_arm $DS always topk \
    --provider-arg source=results/phase1/$DS-naive_skill.json --provider-arg k=1
  run_arm $DS gated topk \
    --provider-arg source=results/phase2/$DS-gated.json --provider-arg k=1
  run_arm $DS oracle oracle
  run_arm $DS select llm_select \
    --provider-arg source=results/phase1/$DS-naive_skill.json \
    --provider-arg model=$MODEL --provider-arg api_base=$API \
    --provider-arg pool=50
done
echo PHASE2B-DONE
