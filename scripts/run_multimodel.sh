#!/usr/bin/env bash
# Community multi-model runner: one command reproduces HySkill Track A
# (imagination retrieval) and optionally Track B (loading gate) for ANY
# OpenAI-compatible model endpoint. Fully resumable: generations and
# embeddings are disk-cached; finished retrieval/infer files are skipped.
#
# Usage:
#   TAG=llama31-8b MODEL=llama31-8b API_BASE=http://localhost:8000/v1 \
#     [TRACKB=1] [NO_THINK=1] [WORKERS=32] [INFER_WORKERS=48] \
#     ./scripts/run_multimodel.sh
#
# TAG    short id, becomes results/multimodel/<TAG>/ and community-results/<TAG>/
# MODEL  the --served-model-name of your endpoint (keep MODEL == TAG; it is
#        baked into cache keys, so it must stay identical across reruns)
# Long runs over SSH: launch with  (nohup ./scripts/run_multimodel.sh > run.log 2>&1 &)
set -uo pipefail
cd "$(dirname "$0")/.."
: "${TAG:?set TAG=<short-model-tag>, e.g. llama31-8b}"
: "${MODEL:?set MODEL=<served model name> (use the same string as TAG)}"
: "${API_BASE:?set API_BASE=http://localhost:PORT/v1}"
WORKERS=${WORKERS:-32}
INFER_WORKERS=${INFER_WORKERS:-48}
NO_THINK=${NO_THINK:-0}
TRACKB=${TRACKB:-0}
export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}

SRA=external/SR-Agents
CORPUS=$SRA/data/bench/corpus/corpus.json
OUT=results/multimodel/$TAG
mkdir -p "$OUT" results/hyp_cache results/emb_cache
DATASETS=(theoremqa logicbench medcalcbench champ bigcodebench)
RULE_DATASETS=(theoremqa logicbench medcalcbench champ)

NOTHINK_FLAG=""; NOTHINK_ARG=""
if [[ "$NO_THINK" == "1" ]]; then NOTHINK_FLAG="--no-think"; NOTHINK_ARG="--retriever-arg no_think=1"; fi

echo "=== [1/4] Track A warmup: imagination cache (skill template, K=4, temp 0.7) ==="
.venv/bin/python scripts/warm_cache.py \
  --instances $(for d in "${DATASETS[@]}"; do echo "$SRA/data/bench/instances/$d.json"; done) \
  --templates skill --k 4 --model "$MODEL" --api-base "$API_BASE" \
  --workers "$WORKERS" $NOTHINK_FLAG || exit 1

echo "=== [2/4] Track A retrieval: naive_skill (full-skill imagination) x 5 domains ==="
for DS in "${DATASETS[@]}"; do
  if [[ -s $OUT/$DS-naive_skill.json ]]; then echo "skip $DS (exists)"; continue; fi
  .venv/bin/sragents --plugin hyskill.plugin retrieve --retriever naive_hyde \
    --retriever-arg template=skill \
    --retriever-arg model="$MODEL" --retriever-arg api_base="$API_BASE" \
    --retriever-arg k_samples=4 --retriever-arg cache_dir=results/hyp_cache \
    --retriever-arg emb_cache_dir=results/emb_cache $NOTHINK_ARG \
    --corpus $CORPUS --instances $SRA/data/bench/instances/$DS.json \
    --output "$OUT/$DS-naive_skill.json" --top-k 50 || exit 1
done
echo TRACKA-DONE

if [[ "$TRACKB" == "1" ]]; then
  echo "=== [3/4] Track B gate: bare -> signals -> calibrate -> always/gated ==="
  for DS in "${RULE_DATASETS[@]}"; do
    INST=$SRA/data/bench/instances/$DS.json
    echo "### $DS bare"
    .venv/bin/sragents infer --instances "$INST" --output "$OUT/$DS-bare.jsonl" \
      --model "$MODEL" --api-base "$API_BASE" --provider none --engine direct \
      --workers "$INFER_WORKERS" --max-tokens 2048 --label bare || exit 1
    .venv/bin/sragents evaluate --input "$OUT/$DS-bare.jsonl" --instances "$INST" \
      --output "$OUT/$DS-bare.eval.json" || exit 1
    echo "### $DS signals + calibrate + apply"
    .venv/bin/python scripts/gate.py signals --retrieval "$OUT/$DS-naive_skill.json" \
      --corpus $CORPUS --instances "$INST" --cache-dir results/hyp_cache \
      --model "$MODEL" --k 4 --out "$OUT/$DS-signals.json" || exit 1
    .venv/bin/python scripts/gate.py calibrate --signals "$OUT/$DS-signals.json" \
      --bare-eval "$OUT/$DS-bare.eval.json" --out "$OUT/$DS-taus.json" || exit 1
    .venv/bin/python scripts/gate.py apply --signals "$OUT/$DS-signals.json" \
      --taus "$OUT/$DS-taus.json" --retrieval "$OUT/$DS-naive_skill.json" \
      --out "$OUT/$DS-gated.json" || exit 1
    for ARM in always gated; do
      SRC=$OUT/$DS-naive_skill.json; [[ $ARM == gated ]] && SRC=$OUT/$DS-gated.json
      echo "### $DS $ARM"
      .venv/bin/sragents infer --instances "$INST" --output "$OUT/$DS-$ARM.jsonl" \
        --model "$MODEL" --api-base "$API_BASE" \
        --provider topk --provider-arg source="$SRC" --provider-arg k=1 \
        --engine direct --workers "$INFER_WORKERS" --max-tokens 2048 --label $ARM || exit 1
      .venv/bin/sragents evaluate --input "$OUT/$DS-$ARM.jsonl" --instances "$INST" \
        --output "$OUT/$DS-$ARM.eval.json" || exit 1
    done
  done
  echo TRACKB-DONE
fi

echo "=== [4/4] summary -> community-results/$TAG/summary.json ==="
.venv/bin/python scripts/summarize_multimodel.py "$TAG" "$MODEL"
echo ALL-DONE
