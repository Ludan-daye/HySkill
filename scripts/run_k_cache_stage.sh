#!/usr/bin/env bash
# Warm one model to a target K, audit it, and export requested prefix packs.
set -euo pipefail

cd "$(dirname "$0")/.."

: "${TAG:?set TAG to the community-results model tag}"
: "${MODEL:?set MODEL to the served-model name and cache model tag}"
: "${API_BASE:?set API_BASE to an OpenAI-compatible /v1 endpoint}"
: "${MODEL_REVISION:?set MODEL_REVISION to the exact checkpoint revision}"
: "${GENERATION_COMMIT:?set GENERATION_COMMIT to the generation code commit}"
: "${TARGET_K:?set TARGET_K to the cache prefix length to generate}"
: "${PREFIX_K_VALUES:?set PREFIX_K_VALUES to space-separated prefixes to export}"
: "${WORKERS:?set WORKERS to the cache-generation concurrency}"
: "${CACHE_DIR:?set CACHE_DIR to the shared hypothetical cache directory}"
: "${INSTANCES_DIR:?set INSTANCES_DIR to the SR-Agents instances directory}"

if ! [[ "$TARGET_K" =~ ^[1-9][0-9]*$ ]]; then
  echo "TARGET_K must be a positive integer: received=$TARGET_K" >&2
  exit 2
fi
if ! [[ "$WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "WORKERS must be a positive integer: received=$WORKERS" >&2
  exit 2
fi

INSTANCE_PATHS=(
  "$INSTANCES_DIR/theoremqa.json"
  "$INSTANCES_DIR/logicbench.json"
  "$INSTANCES_DIR/medcalcbench.json"
  "$INSTANCES_DIR/champ.json"
  "$INSTANCES_DIR/bigcodebench.json"
)
for instance_path in "${INSTANCE_PATHS[@]}"; do
  if [[ ! -f "$instance_path" ]]; then
    echo "required instance file does not exist: path=$instance_path" >&2
    exit 2
  fi
done
if [[ ! -d "$CACHE_DIR" ]]; then
  echo "required cache directory does not exist: path=$CACHE_DIR" >&2
  exit 2
fi

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export PYTHONPATH="$PWD"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="127.0.0.1,localhost"

MODEL_RESPONSE=$(curl --fail --silent --show-error "$API_BASE/models")
printf '%s' "$MODEL_RESPONSE" | .venv/bin/python -c '
import json
import sys

expected = sys.argv[1]
payload = json.load(sys.stdin)
actual = [str(item["id"]) for item in payload.get("data", [])]
if expected not in actual:
    raise SystemExit(
        f"served-model mismatch: expected={expected!r}, actual={actual!r}")
print(f"served-model verified: {expected}")
' "$MODEL"

NO_THINK_ARGS=()
if [[ "$MODEL" == qwen* ]]; then
  NO_THINK_ARGS=(--no-think)
fi

mkdir -p "results/k-ablation/$TAG/logs" "results/k-ablation/$TAG/audits"
STARTED_AT=$(date --iso-8601=seconds)
.venv/bin/python scripts/warm_cache.py \
  --instances "${INSTANCE_PATHS[@]}" \
  --templates passage,skill,sentence \
  --k "$TARGET_K" \
  --model "$MODEL" \
  --api-base "$API_BASE" \
  --temperature 0.7 \
  --workers "$WORKERS" \
  --cache-dir "$CACHE_DIR" \
  "${NO_THINK_ARGS[@]}"

AUDIT_PATH="results/k-ablation/$TAG/audits/cache-k$TARGET_K.json"
.venv/bin/python scripts/audit_k_cache.py \
  --model "$MODEL" \
  --instances "${INSTANCE_PATHS[@]}" \
  --templates passage,skill,sentence \
  --temperature 0.7 \
  --cache-dir "$CACHE_DIR" \
  --k "$TARGET_K" \
  --output "$AUDIT_PATH"

for prefix_k in $PREFIX_K_VALUES; do
  if ! [[ "$prefix_k" =~ ^[1-9][0-9]*$ ]]; then
    echo "prefix K must be a positive integer: received=$prefix_k" >&2
    exit 2
  fi
  if (( prefix_k > TARGET_K )); then
    echo "prefix K exceeds target: prefix=$prefix_k target=$TARGET_K" >&2
    exit 2
  fi
  .venv/bin/python scripts/export_full_imagination_cache.py \
    --tag "$TAG" \
    --model "$MODEL" \
    --model-revision "$MODEL_REVISION" \
    --generation-commit "$GENERATION_COMMIT" \
    --k "$prefix_k" \
    --cache-dir "$CACHE_DIR" \
    --instances-dir "$INSTANCES_DIR"
done

FINISHED_AT=$(date --iso-8601=seconds)
.venv/bin/python -c '
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "tag": sys.argv[2],
    "model": sys.argv[3],
    "target_k": int(sys.argv[4]),
    "prefix_k_values": [int(value) for value in sys.argv[5].split()],
    "started_at": sys.argv[6],
    "finished_at": sys.argv[7],
    "audit": sys.argv[8],
}
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
' "results/k-ablation/$TAG/cache-stage-k$TARGET_K.complete.json" \
  "$TAG" "$MODEL" "$TARGET_K" "$PREFIX_K_VALUES" \
  "$STARTED_AT" "$FINISHED_AT" "$AUDIT_PATH"

echo "CACHE-STAGE-DONE tag=$TAG model=$MODEL target_k=$TARGET_K"
