#!/usr/bin/env bash
# Run all four fresh Always/Gated/Select K=2 jobs for one eligible model.
set -euo pipefail

: "${REPOSITORY_ROOT:?REPOSITORY_ROOT is required}"
: "${PYTHON_BIN:?PYTHON_BIN is required}"
: "${RESULT_TAG:?RESULT_TAG is required}"
: "${CACHE_MODEL_TAG:?CACHE_MODEL_TAG is required}"
: "${SERVED_MODEL:?SERVED_MODEL is required}"
: "${API_BASE:?API_BASE is required}"
: "${INSTANCES_ROOT:?INSTANCES_ROOT is required}"
: "${CORPUS:?CORPUS is required}"
: "${ROUTED_ROOT:?ROUTED_ROOT is required}"
: "${GATE_ROOT:?GATE_ROOT is required}"
: "${NEW_RUNTIME_ROOT:?NEW_RUNTIME_ROOT is required}"
: "${EMPTY_LEGACY_JSONL:?EMPTY_LEGACY_JSONL is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "${WORKERS:?WORKERS is required}"

cd "$REPOSITORY_ROOT"

DOMAINS=(theoremqa logicbench medcalcbench champ)
EXPECTED_COUNTS=(747 760 1100 223)
PIDS=()
LABELS=()
LOG_ROOT="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_ROOT"

launch_domain() {
  local domain="$1"
  local expected_count="$2"
  local instances="$INSTANCES_ROOT/$domain.json"
  local routed_source="$ROUTED_ROOT/$domain-routed.json"
  local gated_source="$GATE_ROOT/$domain-routed-gated.json"
  local taus="$GATE_ROOT/$domain-routed-taus.json"
  local runtime_manifest="$NEW_RUNTIME_ROOT/$domain.json"

  REPOSITORY_ROOT="$REPOSITORY_ROOT" \
  PYTHON_BIN="$PYTHON_BIN" \
  RESULT_TAG="$RESULT_TAG" \
  CACHE_MODEL_TAG="$CACHE_MODEL_TAG" \
  SERVED_MODEL="$SERVED_MODEL" \
  API_BASE="$API_BASE" \
  DOMAIN="$domain" \
  EXPECTED_COUNT="$expected_count" \
  INSTANCES="$instances" \
  CORPUS="$CORPUS" \
  ROUTED_SOURCE="$routed_source" \
  GATED_SOURCE="$gated_source" \
  TAUS="$taus" \
  NEW_RUNTIME_MANIFEST="$runtime_manifest" \
  OLD_ALWAYS_RUNTIME_MANIFEST="$runtime_manifest" \
  OLD_GATED_RUNTIME_MANIFEST="$runtime_manifest" \
  LEGACY_ALWAYS_JSONL="$EMPTY_LEGACY_JSONL" \
  LEGACY_GATED_JSONL="$EMPTY_LEGACY_JSONL" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  WORKERS="$WORKERS" \
    bash scripts/run_k2_main.sh \
    >"$LOG_ROOT/$RESULT_TAG-$domain-main.log" 2>&1 &
  PIDS+=("$!")
  LABELS+=("$domain-main")

  REPOSITORY_ROOT="$REPOSITORY_ROOT" \
  PYTHON_BIN="$PYTHON_BIN" \
  RESULT_TAG="$RESULT_TAG" \
  SERVED_MODEL="$SERVED_MODEL" \
  API_BASE="$API_BASE" \
  DOMAIN="$domain" \
  EXPECTED_COUNT="$expected_count" \
  INSTANCES="$instances" \
  CORPUS="$CORPUS" \
  ROUTED_SOURCE="$routed_source" \
  TAUS="$taus" \
  NEW_RUNTIME_MANIFEST="$runtime_manifest" \
  OLD_SELECT_RUNTIME_MANIFEST="$runtime_manifest" \
  LEGACY_SELECT_JSONL="$EMPTY_LEGACY_JSONL" \
  SELECTOR_CODE_BUNDLE_SHA256="$SELECTOR_CODE_BUNDLE_SHA256" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  WORKERS="$WORKERS" \
    bash scripts/run_k2_select_main.sh \
    >"$LOG_ROOT/$RESULT_TAG-$domain-select.log" 2>&1 &
  PIDS+=("$!")
  LABELS+=("$domain-select")
}

: "${SELECTOR_CODE_BUNDLE_SHA256:?SELECTOR_CODE_BUNDLE_SHA256 is required}"

for index in "${!DOMAINS[@]}"; do
  launch_domain "${DOMAINS[$index]}" "${EXPECTED_COUNTS[$index]}"
done

failed_labels=()
for index in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$index]}"; then
    failed_labels+=("${LABELS[$index]}")
  fi
done
if ((${#failed_labels[@]} > 0)); then
  printf 'K=2 eligible-model jobs failed: result_tag=%s jobs=%s\n' \
    "$RESULT_TAG" "${failed_labels[*]}" >&2
  exit 1
fi

completion_count=$(
  find "$OUTPUT_ROOT/$RESULT_TAG/audits" \
    -maxdepth 1 -type f -name "*completion.json" | wc -l
)
if [[ "$completion_count" -ne 16 ]]; then
  printf 'K=2 completion count mismatch: result_tag=%s expected=16 actual=%s\n' \
    "$RESULT_TAG" "$completion_count" >&2
  exit 1
fi

printf 'K2_ELIGIBLE_MODEL_FORMAL_COMPLETE result_tag=%s completions=%s\n' \
  "$RESULT_TAG" "$completion_count" |
  tee "$OUTPUT_ROOT/$RESULT_TAG/FORMAL_COMPLETE"
