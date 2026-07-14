#!/usr/bin/env bash
# Community multi-model runner v2.1: full comparison menu for ANY
# OpenAI-compatible model endpoint, with three-level parallelism:
#   - rerank stream runs CONCURRENTLY with the warmup (it needs no
#     imaginations — only BM25 candidates + the endpoint);
#   - the 5 domains' variant retrievals run as parallel streams;
#   - Track B runs the 4 rule domains as parallel streams (arms serial
#     within a domain: bare -> signals -> calibrate -> always/gated).
# Fully resumable: generations + embeddings are disk-cached, finished
# output files are skipped. Per-stream logs in results/multimodel/<TAG>/logs/.
#
# Usage:
#   TAG=llama31-8b MODEL=llama31-8b API_BASE=http://localhost:8000/v1 \
#     [TRACKB=1] [SELECT=1] [NO_THINK=1] [RERANK_DOMAINS="all"] \
#     [WORKERS=32] [INFER_WORKERS=24] [RERANK_WORKERS=8] \
#     ./scripts/run_multimodel.sh
#
# TAG    short id -> results/multimodel/<TAG>/, community-results/<TAG>/
# MODEL  the --served-model-name of your endpoint. KEEP MODEL == TAG and
#        never change it mid-run (it is baked into cache keys).
# INFER_WORKERS is PER Track-B stream (4 streams run at once -> ~4x in flight).
# Long runs over SSH: (nohup ./scripts/run_multimodel.sh > run.log 2>&1 &)
set -uo pipefail
cd "$(dirname "$0")/.."
: "${TAG:?set TAG=<short-model-tag>, e.g. llama31-8b}"
: "${MODEL:?set MODEL=<served model name> (use the same string as TAG)}"
: "${API_BASE:?set API_BASE=http://localhost:PORT/v1}"
WORKERS=${WORKERS:-32}
INFER_WORKERS=${INFER_WORKERS:-24}
RERANK_WORKERS=${RERANK_WORKERS:-8}
NO_THINK=${NO_THINK:-0}
TRACKB=${TRACKB:-0}
SELECT=${SELECT:-0}
RERANK_DOMAINS=${RERANK_DOMAINS:-"theoremqa logicbench bigcodebench"}
export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}

SRA=external/SR-Agents
CORPUS=$SRA/data/bench/corpus/corpus.json
# Encoder: HF id by default; override with ENC=/local/path when the box
# cannot reach HF (e.g. use a ModelScope-downloaded copy).
ENC=${ENC:-sentence-transformers/all-MiniLM-L6-v2}
OUT=results/multimodel/$TAG
LOGS=$OUT/logs
mkdir -p "$OUT" "$LOGS" results/hyp_cache results/emb_cache
DATASETS=(theoremqa logicbench medcalcbench champ bigcodebench)
RULE_DATASETS=(theoremqa logicbench medcalcbench champ)
[[ "$RERANK_DOMAINS" == "all" ]] && RERANK_DOMAINS="${DATASETS[*]}"

NOTHINK_FLAG=""; NOTHINK_ARG=""
if [[ "$NO_THINK" == "1" ]]; then NOTHINK_FLAG="--no-think"; NOTHINK_ARG="--retriever-arg no_think=1"; fi
GEN_ARGS="--retriever-arg model=$MODEL --retriever-arg api_base=$API_BASE \
  --retriever-arg k_samples=4 --retriever-arg cache_dir=results/hyp_cache \
  --retriever-arg emb_cache_dir=results/emb_cache $NOTHINK_ARG"

# ---------------------------------------------------------------- streams
rerank_stream() {  # no imaginations needed -> overlaps with warmup
  for DS in $RERANK_DOMAINS; do
    local INST=$SRA/data/bench/instances/$DS.json
    local O=$OUT/$DS-bm25.json
    [[ -s $O ]] || { echo "### $DS fast_bm25"; \
      .venv/bin/sragents --plugin hyskill.plugin retrieve --retriever fast_bm25 \
      --corpus $CORPUS --instances $INST --output $O --top-k 50 || return 1; }
    O=$OUT/$DS-llm_rerank.json
    [[ -s $O ]] || { echo "### $DS llm_rerank"; \
      .venv/bin/sragents --plugin hyskill.plugin rerank \
      --input $OUT/$DS-bm25.json --output $O --instances $INST --corpus $CORPUS \
      --model "$MODEL" --api-base "$API_BASE" --workers "$RERANK_WORKERS" || return 1; }
  done
}

variant_stream() {  # one domain, 5 variants serial (cache-hit generations)
  local DS=$1 INST=$SRA/data/bench/instances/$1.json O T
  for T in sentence passage skill; do
    O=$OUT/$DS-naive_$T.json
    [[ -s $O ]] || { echo "### $DS naive_$T"; .venv/bin/sragents --plugin hyskill.plugin retrieve \
      --retriever naive_hyde --retriever-arg template=$T --retriever-arg encoder_name=$ENC \
      $GEN_ARGS --corpus $CORPUS --instances $INST --output $O --top-k 50 || return 1; }
  done
  O=$OUT/$DS-hyskill.json
  [[ -s $O ]] || { echo "### $DS hyskill"; .venv/bin/sragents --plugin hyskill.plugin retrieve \
    --retriever hyskill --retriever-arg corpus_path=$CORPUS --retriever-arg encoder_name=$ENC \
    $GEN_ARGS --corpus $CORPUS --instances $INST --output $O --top-k 50 || return 1; }
  O=$OUT/$DS-two_stage.json
  [[ -s $O ]] || { echo "### $DS two_stage"; .venv/bin/sragents --plugin hyskill.plugin retrieve \
    --retriever two_stage --retriever-arg corpus_path=$CORPUS --retriever-arg encoder_name=$ENC \
    --retriever-arg recall_k=50 $GEN_ARGS --corpus $CORPUS --instances $INST --output $O --top-k 50 || return 1; }
}

trackb_stream() {  # one rule domain, arms serial (data dependencies)
  local DS=$1 INST=$SRA/data/bench/instances/$1.json
  echo "### $DS bare"
  .venv/bin/sragents infer --instances "$INST" --output "$OUT/$DS-bare.jsonl" \
    --model "$MODEL" --api-base "$API_BASE" --provider none --engine direct \
    --workers "$INFER_WORKERS" --max-tokens 2048 --label bare || return 1
  .venv/bin/sragents evaluate --input "$OUT/$DS-bare.jsonl" --instances "$INST" \
    --output "$OUT/$DS-bare.eval.json" || return 1
  echo "### $DS signals+calibrate+apply"
  .venv/bin/python scripts/gate.py signals --retrieval "$OUT/$DS-routed.json" \
    --corpus $CORPUS --instances "$INST" --cache-dir results/hyp_cache \
    --model "$MODEL" --k 4 --encoder "$ENC" --out "$OUT/$DS-signals.json" || return 1
  .venv/bin/python scripts/gate.py calibrate --signals "$OUT/$DS-signals.json" \
    --bare-eval "$OUT/$DS-bare.eval.json" --out "$OUT/$DS-taus.json" || return 1
  .venv/bin/python scripts/gate.py apply --signals "$OUT/$DS-signals.json" \
    --taus "$OUT/$DS-taus.json" --retrieval "$OUT/$DS-routed.json" \
    --out "$OUT/$DS-gated.json" || return 1
  local ARM SRC
  for ARM in always gated; do
    SRC=$OUT/$DS-routed.json; [[ $ARM == gated ]] && SRC=$OUT/$DS-gated.json
    echo "### $DS $ARM"
    .venv/bin/sragents infer --instances "$INST" --output "$OUT/$DS-$ARM.jsonl" \
      --model "$MODEL" --api-base "$API_BASE" \
      --provider topk --provider-arg source="$SRC" --provider-arg k=1 \
      --engine direct --workers "$INFER_WORKERS" --max-tokens 2048 --label $ARM || return 1
    .venv/bin/sragents evaluate --input "$OUT/$DS-$ARM.jsonl" --instances "$INST" \
      --output "$OUT/$DS-$ARM.eval.json" || return 1
  done
  if [[ "$SELECT" == "1" ]]; then
    echo "### $DS select"
    .venv/bin/sragents infer --instances "$INST" --output "$OUT/$DS-select.jsonl" \
      --model "$MODEL" --api-base "$API_BASE" \
      --provider llm_select --provider-arg source="$OUT/$DS-routed.json" \
      --provider-arg model="$MODEL" --provider-arg api_base="$API_BASE" \
      --provider-arg pool=50 \
      --engine direct --workers "$INFER_WORKERS" --max-tokens 2048 --label select || return 1
    .venv/bin/sragents evaluate --input "$OUT/$DS-select.jsonl" --instances "$INST" \
      --output "$OUT/$DS-select.eval.json" || return 1
  fi
}

# ------------------------------------------------------------ orchestration
echo "=== [1/5] rerank stream (background, overlaps warmup) + warmup ==="
rerank_stream > "$LOGS/rerank.log" 2>&1 &
RERANK_PID=$!

.venv/bin/python scripts/warm_cache.py \
  --instances $(for d in "${DATASETS[@]}"; do echo "$SRA/data/bench/instances/$d.json"; done) \
  --templates passage,skill,sentence --k 4 --model "$MODEL" --api-base "$API_BASE" \
  --workers "$WORKERS" $NOTHINK_FLAG || exit 1
echo WARMUP-DONE

echo "=== [2/5] variant retrieval: 5 parallel domain streams ==="
VPIDS=()
for DS in "${DATASETS[@]}"; do
  variant_stream $DS > "$LOGS/variants-$DS.log" 2>&1 &
  VPIDS+=($!)
done
VFAIL=0
for p in "${VPIDS[@]}"; do wait $p || VFAIL=1; done
[[ $VFAIL == 0 ]] || { echo "VARIANT-STREAM-FAILED (see $LOGS/variants-*.log)"; exit 1; }
echo TRACKA-VARIANTS-DONE

wait $RERANK_PID || { echo "RERANK-STREAM-FAILED (see $LOGS/rerank.log)"; exit 1; }
echo TRACKA-RERANK-DONE

echo "=== [3/5] granularity routing (20% val split) ==="
for DS in "${DATASETS[@]}"; do
  O=$OUT/$DS-routed.json
  [[ -s $O ]] || .venv/bin/python scripts/route_variant.py \
    --dir "$OUT" --dataset $DS --out $O || exit 1
done
echo ROUTE-DONE

if [[ "$TRACKB" == "1" ]]; then
  echo "=== [4/5] Track B: 4 parallel domain streams ==="
  BPIDS=()
  for DS in "${RULE_DATASETS[@]}"; do
    trackb_stream $DS > "$LOGS/trackb-$DS.log" 2>&1 &
    BPIDS+=($!)
  done
  BFAIL=0
  for p in "${BPIDS[@]}"; do wait $p || BFAIL=1; done
  [[ $BFAIL == 0 ]] || { echo "TRACKB-STREAM-FAILED (see $LOGS/trackb-*.log)"; exit 1; }
  echo TRACKB-DONE
fi

echo "=== [5/5] summary -> community-results/$TAG/summary.json ==="
.venv/bin/python scripts/summarize_multimodel.py "$TAG" "$MODEL"
echo ALL-DONE
