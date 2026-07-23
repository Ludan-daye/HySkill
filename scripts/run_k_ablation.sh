#!/usr/bin/env bash
# Run bounded cache-only retrieval and routing for explicit K/domain lists.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"

: "${TAG:?set TAG to the community-results model tag}"
: "${MODEL:?set MODEL to the cache model name}"
: "${API_BASE:?set API_BASE explicitly; a closed local endpoint is valid after cache audit}"
: "${K_VALUES:?set K_VALUES to a space-separated list such as '1 2 4 8 10'}"
: "${DATASETS:?set DATASETS to a space-separated domain list}"
: "${MAX_RETRIEVAL_JOBS:?set MAX_RETRIEVAL_JOBS to a positive integer}"
: "${ENC:?set ENC to the exact encoder path or model ID}"
: "${CACHE_DIR:?set CACHE_DIR to the complete imagination cache}"
: "${EMB_CACHE_DIR:?set EMB_CACHE_DIR to the corpus embedding cache}"
: "${INSTANCES_DIR:?set INSTANCES_DIR to the SR-Agents instance directory}"
: "${CORPUS:?set CORPUS to the SR-Agents corpus JSON}"
: "${SOURCE_REVISION:?set SOURCE_REVISION to the reproducible runner revision}"

if ! [[ "$MAX_RETRIEVAL_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_RETRIEVAL_JOBS must be a positive integer: value=$MAX_RETRIEVAL_JOBS" >&2
  exit 2
fi
if [[ ! -x .venv/bin/sragents ]]; then
  echo "SR-Agents executable is missing: path=$PWD/.venv/bin/sragents" >&2
  exit 2
fi
if [[ ! -x .venv/bin/python ]]; then
  echo "Project Python executable is missing: path=$PWD/.venv/bin/python" >&2
  exit 2
fi
for required_path in "$CACHE_DIR" "$EMB_CACHE_DIR" "$INSTANCES_DIR" "$CORPUS"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required experiment path does not exist: path=$required_path" >&2
    exit 2
  fi
done

read -r -a REQUESTED_K_VALUES <<< "$K_VALUES"
read -r -a REQUESTED_DATASETS <<< "$DATASETS"
if (( ${#REQUESTED_K_VALUES[@]} == 0 || ${#REQUESTED_DATASETS[@]} == 0 )); then
  echo "K_VALUES and DATASETS must each contain at least one value" >&2
  exit 2
fi

declare -A VALID_K_VALUES=([1]=1 [2]=1 [4]=1 [8]=1 [10]=1)
declare -A VALID_DATASETS=(
  [theoremqa]=1
  [logicbench]=1
  [medcalcbench]=1
  [champ]=1
  [bigcodebench]=1
)
MAX_K=0
for k_samples in "${REQUESTED_K_VALUES[@]}"; do
  if [[ -z "${VALID_K_VALUES[$k_samples]+x}" ]]; then
    echo "Unsupported K value: value=$k_samples, allowed=1,2,4,8,10" >&2
    exit 2
  fi
  if (( k_samples > MAX_K )); then
    MAX_K=$k_samples
  fi
  manifest="community-results/$TAG/imagination_full_k$k_samples.manifest.json"
  if [[ ! -f "$manifest" ]]; then
    echo "Required K cache manifest is missing: path=$manifest" >&2
    exit 2
  fi
done
for dataset in "${REQUESTED_DATASETS[@]}"; do
  if [[ -z "${VALID_DATASETS[$dataset]+x}" ]]; then
    echo "Unsupported dataset: value=$dataset" >&2
    exit 2
  fi
  if [[ ! -f "$INSTANCES_DIR/$dataset.json" ]]; then
    echo "Required instance file is missing: path=$INSTANCES_DIR/$dataset.json" >&2
    exit 2
  fi
done

VALIDATION_DOMAINS=$(IFS=,; echo "${REQUESTED_DATASETS[*]}")
for k_samples in "${REQUESTED_K_VALUES[@]}"; do
  .venv/bin/python scripts/validate_k_cache.py \
    --manifest "community-results/$TAG/imagination_full_k$k_samples.manifest.json" \
    --instances-dir "$INSTANCES_DIR" \
    --domains "$VALIDATION_DOMAINS" \
    --tag "$TAG" \
    --model "$MODEL" \
    --k "$k_samples"
done

OUT="results/k-ablation/$TAG"
LOGS="$OUT/logs"
mkdir -p "$OUT" "$LOGS"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="127.0.0.1,localhost"

INSTANCE_PATHS=()
for dataset in "${REQUESTED_DATASETS[@]}"; do
  INSTANCE_PATHS+=("$INSTANCES_DIR/$dataset.json")
done

AUDIT_PATH="$OUT/audits/cache-k$MAX_K.json"
mkdir -p "$OUT/audits"
.venv/bin/python scripts/audit_k_cache.py \
  --model "$MODEL" \
  --instances "${INSTANCE_PATHS[@]}" \
  --templates passage,skill,sentence \
  --temperature 0.7 \
  --cache-dir "$CACHE_DIR" \
  --k "$MAX_K" \
  --output "$AUDIT_PATH"

validation_args() {
  local result_path=$1
  local k_samples=$2
  local dataset=$3
  local variant=$4
  printf '%s\n' \
    --result "$result_path" \
    --instances "$INSTANCES_DIR/$dataset.json" \
    --cache-manifest "community-results/$TAG/imagination_full_k$k_samples.manifest.json" \
    --tag "$TAG" \
    --model "$MODEL" \
    --k "$k_samples" \
    --domain "$dataset" \
    --variant "$variant" \
    --encoder "$ENC" \
    --source-revision "$SOURCE_REVISION"
}

finalization_args() {
  local raw_path=$1
  local result_path=$2
  local k_samples=$3
  local dataset=$4
  local variant=$5
  printf '%s\n' \
    --input "$raw_path" \
    --output "$result_path" \
    --instances "$INSTANCES_DIR/$dataset.json" \
    --cache-manifest "community-results/$TAG/imagination_full_k$k_samples.manifest.json" \
    --tag "$TAG" \
    --model "$MODEL" \
    --k "$k_samples" \
    --domain "$dataset" \
    --variant "$variant" \
    --encoder "$ENC" \
    --source-revision "$SOURCE_REVISION"
}

validate_existing_result() {
  local result_path=$1
  local k_samples=$2
  local dataset=$3
  local variant=$4
  local args=()
  while IFS= read -r argument; do
    args+=("$argument")
  done < <(validation_args "$result_path" "$k_samples" "$dataset" "$variant")
  .venv/bin/python scripts/validate_k_retrieval.py "${args[@]}"
}

finalize_raw_result() {
  local raw_path=$1
  local result_path=$2
  local k_samples=$3
  local dataset=$4
  local variant=$5
  local args=()
  while IFS= read -r argument; do
    args+=("$argument")
  done < <(finalization_args "$raw_path" "$result_path" "$k_samples" "$dataset" "$variant")
  .venv/bin/python scripts/finalize_k_retrieval.py "${args[@]}"
}

run_one_variant() {
  local k_samples=$1
  local dataset=$2
  local variant=$3
  local output_dir="$OUT/k$k_samples"
  local result_path="$output_dir/$dataset-$variant.json"
  local raw_dir
  local raw_path
  mkdir -p "$output_dir"
  if [[ -e "$result_path" ]]; then
    validate_existing_result "$result_path" "$k_samples" "$dataset" "$variant"
    echo "K-RESULT-SKIP tag=$TAG k=$k_samples domain=$dataset variant=$variant"
    return 0
  fi
  raw_dir=$(mktemp -d "$output_dir/.$dataset-$variant.raw.XXXXXX")
  raw_path="$raw_dir/result.json"
  local common_args=(
    --plugin hyskill.plugin
    retrieve
    --retriever "$variant"
    --retriever-arg "model=$MODEL"
    --retriever-arg "api_base=$API_BASE"
    --retriever-arg "k_samples=$k_samples"
    --retriever-arg "cache_dir=$CACHE_DIR"
    --retriever-arg "emb_cache_dir=$EMB_CACHE_DIR"
    --retriever-arg "encoder_name=$ENC"
    --corpus "$CORPUS"
    --instances "$INSTANCES_DIR/$dataset.json"
    --output "$raw_path"
    --top-k 50
  )
  case "$variant" in
    naive_sentence)
      common_args[4]=naive_hyde
      .venv/bin/sragents "${common_args[@]}" \
        --retriever-arg template=sentence
      ;;
    naive_passage)
      common_args[4]=naive_hyde
      .venv/bin/sragents "${common_args[@]}" \
        --retriever-arg template=passage
      ;;
    naive_skill)
      common_args[4]=naive_hyde
      .venv/bin/sragents "${common_args[@]}" \
        --retriever-arg template=skill
      ;;
    hyskill)
      .venv/bin/sragents "${common_args[@]}" \
        --retriever-arg "corpus_path=$CORPUS"
      ;;
    two_stage)
      .venv/bin/sragents "${common_args[@]}" \
        --retriever-arg "corpus_path=$CORPUS" --retriever-arg recall_k=50
      ;;
    *)
      echo "Unsupported fixed variant: value=$variant" >&2
      return 2
      ;;
  esac
  finalize_raw_result "$raw_path" "$result_path" "$k_samples" "$dataset" "$variant"
  rm -f -- "$raw_path"
  rmdir -- "$raw_dir"
  validate_existing_result "$result_path" "$k_samples" "$dataset" "$variant"
  echo "K-RESULT-DONE tag=$TAG k=$k_samples domain=$dataset variant=$variant"
}

run_fixed_task() {
  local k_samples=$1
  local dataset=$2
  local variant
  for variant in naive_sentence naive_passage naive_skill hyskill two_stage; do
    run_one_variant "$k_samples" "$dataset" "$variant"
  done
}

PIDS=()
TASK_LABELS=()
FAILURES=0
wait_first_task() {
  local pid=${PIDS[0]}
  local label=${TASK_LABELS[0]}
  if ! wait "$pid"; then
    echo "K-TASK-FAILED tag=$TAG task=$label log=$LOGS/$label.log" >&2
    FAILURES=$((FAILURES + 1))
  fi
  PIDS=("${PIDS[@]:1}")
  TASK_LABELS=("${TASK_LABELS[@]:1}")
}

for k_samples in "${REQUESTED_K_VALUES[@]}"; do
  for dataset in "${REQUESTED_DATASETS[@]}"; do
    task_label="k$k_samples-$dataset"
    run_fixed_task "$k_samples" "$dataset" > "$LOGS/$task_label.log" 2>&1 &
    PIDS+=("$!")
    TASK_LABELS+=("$task_label")
    if (( ${#PIDS[@]} >= MAX_RETRIEVAL_JOBS )); then
      wait_first_task
    fi
  done
done
while (( ${#PIDS[@]} > 0 )); do
  wait_first_task
done
if (( FAILURES > 0 )); then
  echo "K-FIXED-MATRIX-FAILED tag=$TAG failures=$FAILURES" >&2
  exit 1
fi

for k_samples in "${REQUESTED_K_VALUES[@]}"; do
  for dataset in "${REQUESTED_DATASETS[@]}"; do
    routed_dir="$OUT/routed/k$k_samples"
    routed_path="$routed_dir/$dataset-routed.json"
    mkdir -p "$routed_dir"
    if [[ -e "$routed_path" ]]; then
      validate_existing_result "$routed_path" "$k_samples" "$dataset" routed
      echo "K-ROUTED-SKIP tag=$TAG k=$k_samples domain=$dataset"
      continue
    fi
    .venv/bin/python scripts/route_k_retrieval.py \
      --input-dir "$OUT/k$k_samples" \
      --output "$routed_path" \
      --instances "$INSTANCES_DIR/$dataset.json" \
      --cache-manifest "community-results/$TAG/imagination_full_k$k_samples.manifest.json" \
      --tag "$TAG" \
      --model "$MODEL" \
      --k "$k_samples" \
      --domain "$dataset" \
      --encoder "$ENC" \
      --source-revision "$SOURCE_REVISION"
    validate_existing_result "$routed_path" "$k_samples" "$dataset" routed
    echo "K-ROUTED-DONE tag=$TAG k=$k_samples domain=$dataset"
  done
done

echo "K-RETRIEVAL-DONE tag=$TAG k_values=$K_VALUES datasets=$DATASETS"
