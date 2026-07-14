#!/usr/bin/env bash
# Community multi-model runner v2: full comparison menu for ANY
# OpenAI-compatible model endpoint.
#   Track A: imagination x 5 variants x 5 domains + llm_rerank comparison
#   Route:   val-split granularity routing (route_variant.py)
#   Track B: bare / always / gated answering arms on routed retrieval
# Fully resumable: generations + embeddings are disk-cached, finished
# output files are skipped.
#
# Usage:
#   TAG=llama31-8b MODEL=llama31-8b API_BASE=http://localhost:8000/v1 \
#     [TRACKB=1] [SELECT=1] [NO_THINK=1] [RERANK_DOMAINS="all"] \
#     [WORKERS=32] [INFER_WORKERS=48] [RERANK_WORKERS=8] \
#     ./scripts/run_multimodel.sh
#
# TAG    short id -> results/multimodel/<TAG>/, community-results/<TAG>/
# MODEL  the --served-model-name of your endpoint. KEEP MODEL == TAG and
#        never change it mid-run (it is baked into cache keys).
# Long runs over SSH: (nohup ./scripts/run_multimodel.sh > run.log 2>&1 &)
set -uo pipefail
cd "$(dirname "$0")/.."
: "${TAG:?set TAG=<short-model-tag>, e.g. llama31-8b}"
: "${MODEL:?set MODEL=<served model name> (use the same string as TAG)}"
: "${API_BASE:?set API_BASE=http://localhost:PORT/v1}"
WORKERS=${WORKERS:-32}
INFER_WORKERS=${INFER_WORKERS:-48}
RERANK_WORKERS=${RERANK_WORKERS:-8}
NO_THINK=${NO_THINK:-0}
TRACKB=${TRACKB:-0}
SELECT=${SELECT:-0}
RERANK_DOMAINS=${RERANK_DOMAINS:-"theoremqa logicbench bigcodebench"}
export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}

SRA=external/SR-Agents
CORPUS=$SRA/data/bench/corpus/corpus.json
ENC=sentence-transformers/all-MiniLM-L6-v2
OUT=results/multimodel/$TAG
mkdir -p "$OUT" results/hyp_cache results/emb_cache
DATASETS=(theoremqa logicbench medcalcbench champ bigcodebench)
RULE_DATASETS=(theoremqa logicbench medcalcbench champ)
[[ "$RERANK_DOMAINS" == "all" ]] && RERANK_DOMAINS="${DATASETS[*]}"

NOTHINK_FLAG=""; NOTHINK_ARG=""
if [[ "$NO_THINK" == "1" ]]; then NOTHINK_FLAG="--no-think"; NOTHINK_ARG="--retriever-arg no_think=1"; fi
GEN_ARGS="--retriever-arg model=$MODEL --retriever-arg api_base=$API_BASE \
  --retriever-arg k_samples=4 --retriever-arg cache_dir=results/hyp_cache \
  --retriever-arg emb_cache_dir=results/emb_cache $NOTHINK_ARG"

echo "=== [1/6] warmup: imagination cache, 3 templates x K=4 x 3,970 queries ==="
.venv/bin/python scripts/warm_cache.py \
  --instances $(for d in "${DATASETS[@]}"; do echo "$SRA/data/bench/instances/$d.json"; done) \
  --templates passage,skill,sentence --k 4 --model "$MODEL" --api-base "$API_BASE" \
  --workers "$WORKERS" $NOTHINK_FLAG || exit 1

echo "=== [2/6] Track A: 5 imagination variants x 5 domains ==="
for DS in "${DATASETS[@]}"; do
  INST=$SRA/data/bench/instances/$DS.json
  for T in sentence passage skill; do
    O=$OUT/$DS-naive_$T.json
    [[ -s $O ]] || { echo "### $DS naive_$T"; .venv/bin/sragents --plugin hyskill.plugin retrieve \
      --retriever naive_hyde --retriever-arg template=$T --retriever-arg encoder_name=$ENC \
      $GEN_ARGS --corpus $CORPUS --instances $INST --output $O --top-k 50 || exit 1; }
  done
  O=$OUT/$DS-hyskill.json
  [[ -s $O ]] || { echo "### $DS hyskill"; .venv/bin/sragents --plugin hyskill.plugin retrieve \
    --retriever hyskill --retriever-arg corpus_path=$CORPUS --retriever-arg encoder_name=$ENC \
    $GEN_ARGS --corpus $CORPUS --instances $INST --output $O --top-k 50 || exit 1; }
  O=$OUT/$DS-two_stage.json
  [[ -s $O ]] || { echo "### $DS two_stage"; .venv/bin/sragents --plugin hyskill.plugin retrieve \
    --retriever two_stage --retriever-arg corpus_path=$CORPUS --retriever-arg encoder_name=$ENC \
    --retriever-arg recall_k=50 $GEN_ARGS --corpus $CORPUS --instances $INST --output $O --top-k 50 || exit 1; }
done
echo TRACKA-VARIANTS-DONE

echo "=== [3/6] Track A: llm_rerank comparison (domains: $RERANK_DOMAINS) ==="
for DS in $RERANK_DOMAINS; do
  INST=$SRA/data/bench/instances/$DS.json
  O=$OUT/$DS-bm25.json
  [[ -s $O ]] || { echo "### $DS fast_bm25 (rerank candidates)"; \
    .venv/bin/sragents --plugin hyskill.plugin retrieve --retriever fast_bm25 \
    --corpus $CORPUS --instances $INST --output $O --top-k 50 || exit 1; }
  O=$OUT/$DS-llm_rerank.json
  [[ -s $O ]] || { echo "### $DS llm_rerank"; \
    .venv/bin/sragents --plugin hyskill.plugin rerank \
    --input $OUT/$DS-bm25.json --output $O --instances $INST --corpus $CORPUS \
    --model "$MODEL" --api-base "$API_BASE" --workers "$RERANK_WORKERS" || exit 1; }
done
echo TRACKA-RERANK-DONE

echo "=== [4/6] granularity routing (20% val split) ==="
for DS in "${DATASETS[@]}"; do
  O=$OUT/$DS-routed.json
  [[ -s $O ]] || .venv/bin/python scripts/route_variant.py \
    --dir "$OUT" --dataset $DS --out $O || exit 1
done
echo ROUTE-DONE

if [[ "$TRACKB" == "1" ]]; then
  echo "=== [5/6] Track B: bare / always / gated on routed retrieval ==="
  for DS in "${RULE_DATASETS[@]}"; do
    INST=$SRA/data/bench/instances/$DS.json
    echo "### $DS bare"
    .venv/bin/sragents infer --instances "$INST" --output "$OUT/$DS-bare.jsonl" \
      --model "$MODEL" --api-base "$API_BASE" --provider none --engine direct \
      --workers "$INFER_WORKERS" --max-tokens 2048 --label bare || exit 1
    .venv/bin/sragents evaluate --input "$OUT/$DS-bare.jsonl" --instances "$INST" \
      --output "$OUT/$DS-bare.eval.json" || exit 1
    echo "### $DS signals + calibrate + apply (on routed top-1)"
    .venv/bin/python scripts/gate.py signals --retrieval "$OUT/$DS-routed.json" \
      --corpus $CORPUS --instances "$INST" --cache-dir results/hyp_cache \
      --model "$MODEL" --k 4 --out "$OUT/$DS-signals.json" || exit 1
    .venv/bin/python scripts/gate.py calibrate --signals "$OUT/$DS-signals.json" \
      --bare-eval "$OUT/$DS-bare.eval.json" --out "$OUT/$DS-taus.json" || exit 1
    .venv/bin/python scripts/gate.py apply --signals "$OUT/$DS-signals.json" \
      --taus "$OUT/$DS-taus.json" --retrieval "$OUT/$DS-routed.json" \
      --out "$OUT/$DS-gated.json" || exit 1
    for ARM in always gated; do
      SRC=$OUT/$DS-routed.json; [[ $ARM == gated ]] && SRC=$OUT/$DS-gated.json
      echo "### $DS $ARM"
      .venv/bin/sragents infer --instances "$INST" --output "$OUT/$DS-$ARM.jsonl" \
        --model "$MODEL" --api-base "$API_BASE" \
        --provider topk --provider-arg source="$SRC" --provider-arg k=1 \
        --engine direct --workers "$INFER_WORKERS" --max-tokens 2048 --label $ARM || exit 1
      .venv/bin/sragents evaluate --input "$OUT/$DS-$ARM.jsonl" --instances "$INST" \
        --output "$OUT/$DS-$ARM.eval.json" || exit 1
    done
    if [[ "$SELECT" == "1" ]]; then
      echo "### $DS select"
      .venv/bin/sragents infer --instances "$INST" --output "$OUT/$DS-select.jsonl" \
        --model "$MODEL" --api-base "$API_BASE" \
        --provider llm_select --provider-arg source="$OUT/$DS-routed.json" \
        --provider-arg model="$MODEL" --provider-arg api_base="$API_BASE" \
        --provider-arg pool=50 \
        --engine direct --workers "$INFER_WORKERS" --max-tokens 2048 --label select || exit 1
      .venv/bin/sragents evaluate --input "$OUT/$DS-select.jsonl" --instances "$INST" \
        --output "$OUT/$DS-select.eval.json" || exit 1
    fi
  done
  echo TRACKB-DONE
fi

echo "=== [6/6] summary -> community-results/$TAG/summary.json ==="
.venv/bin/python scripts/summarize_multimodel.py "$TAG" "$MODEL"
echo ALL-DONE
