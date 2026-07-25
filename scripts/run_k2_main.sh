#!/usr/bin/env bash
# Run one model-domain routed Always/Gated K=2 downstream job.
set -euo pipefail

: "${REPOSITORY_ROOT:?REPOSITORY_ROOT is required}"
: "${PYTHON_BIN:?PYTHON_BIN is required and must be Python 3.10-3.12}"
: "${RESULT_TAG:?RESULT_TAG is required}"
: "${CACHE_MODEL_TAG:?CACHE_MODEL_TAG is required}"
: "${SERVED_MODEL:?SERVED_MODEL is required}"
: "${API_BASE:?API_BASE is required}"
: "${DOMAIN:?DOMAIN is required}"
: "${EXPECTED_COUNT:?EXPECTED_COUNT is required}"
: "${INSTANCES:?INSTANCES is required}"
: "${CORPUS:?CORPUS is required}"
: "${ROUTED_SOURCE:?ROUTED_SOURCE is required}"
: "${GATED_SOURCE:?GATED_SOURCE is required}"
: "${TAUS:?TAUS is required}"
: "${NEW_RUNTIME_MANIFEST:?NEW_RUNTIME_MANIFEST is required}"
: "${OLD_ALWAYS_RUNTIME_MANIFEST:?OLD_ALWAYS_RUNTIME_MANIFEST is required}"
: "${OLD_GATED_RUNTIME_MANIFEST:?OLD_GATED_RUNTIME_MANIFEST is required}"
: "${LEGACY_ALWAYS_JSONL:?LEGACY_ALWAYS_JSONL is required}"
: "${LEGACY_GATED_JSONL:?LEGACY_GATED_JSONL is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "${WORKERS:?WORKERS is required}"

cd "$REPOSITORY_ROOT"
export PYTHONPATH="$REPOSITORY_ROOT:$REPOSITORY_ROOT/external/SR-Agents/src"

PYTHON_VERSION=$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
case "$PYTHON_VERSION" in
  3.10|3.11|3.12) ;;
  *)
    echo "unsupported Python for K=2 runner: version=$PYTHON_VERSION required=3.10-3.12" >&2
    exit 2
    ;;
esac

JOB_ROOT="$OUTPUT_ROOT/$RESULT_TAG"
AUDIT_ROOT="$JOB_ROOT/audits"
LOG_ROOT="$JOB_ROOT/logs"
mkdir -p "$JOB_ROOT" "$AUDIT_ROOT" "$LOG_ROOT"

"$PYTHON_BIN" scripts/export_k2_loading_decisions.py \
  --instances "$INSTANCES" \
  --always-source "$ROUTED_SOURCE" \
  --gated-source "$GATED_SOURCE" \
  --taus "$TAUS" \
  --model "$RESULT_TAG" \
  --domain "$DOMAIN" \
  --expected-count "$EXPECTED_COUNT" \
  --output "$JOB_ROOT/$DOMAIN-routed-always-gated.loading.jsonl"

run_answer_arm() {
  local arm="$1"
  local decision_source="$2"
  local legacy_jsonl="$3"
  local old_runtime_manifest="$4"
  local stem="$5"

  "$PYTHON_BIN" scripts/audit_k2_reuse.py \
    --instances "$INSTANCES" \
    --corpus "$CORPUS" \
    --decision-source "$decision_source" \
    --legacy-jsonl "$legacy_jsonl" \
    --old-runtime-manifest "$old_runtime_manifest" \
    --new-runtime-manifest "$NEW_RUNTIME_MANIFEST" \
    --result-tag "$RESULT_TAG" \
    --arm "$arm" \
    --domain "$DOMAIN" \
    --audit-output "$AUDIT_ROOT/$DOMAIN-$stem.reuse.jsonl" \
    --preseed-output "$AUDIT_ROOT/$DOMAIN-$stem.preseed.jsonl" \
    --pending-output "$AUDIT_ROOT/$DOMAIN-$stem.pending.json"

  "$PYTHON_BIN" scripts/run_k2_answers.py \
    --instances "$INSTANCES" \
    --corpus "$CORPUS" \
    --decision-source "$decision_source" \
    --audit "$AUDIT_ROOT/$DOMAIN-$stem.reuse.jsonl" \
    --preseed "$AUDIT_ROOT/$DOMAIN-$stem.preseed.jsonl" \
    --output "$JOB_ROOT/$DOMAIN-$stem.jsonl" \
    --attempt-log "$LOG_ROOT/$DOMAIN-$stem.attempts.jsonl" \
    --runtime-manifest "$NEW_RUNTIME_MANIFEST" \
    --model "$SERVED_MODEL" \
    --api-base "$API_BASE" \
    --arm "$arm" \
    --domain "$DOMAIN" \
    --workers "$WORKERS"

  "$PYTHON_BIN" scripts/validate_k2_downstream.py answer \
    --instances "$INSTANCES" \
    --corpus "$CORPUS" \
    --decision-source "$decision_source" \
    --answers "$JOB_ROOT/$DOMAIN-$stem.jsonl" \
    --audit "$AUDIT_ROOT/$DOMAIN-$stem.reuse.jsonl" \
    --legacy-jsonl "$legacy_jsonl" \
    --old-runtime-manifest "$old_runtime_manifest" \
    --runtime-manifest "$NEW_RUNTIME_MANIFEST" \
    --result-tag "$RESULT_TAG" \
    --model "$SERVED_MODEL" \
    --arm "$arm" \
    --domain "$DOMAIN" \
    --expected-count "$EXPECTED_COUNT" \
    --output "$AUDIT_ROOT/$DOMAIN-$stem.completion.json"
}

run_answer_arm \
  routed_always \
  "$ROUTED_SOURCE" \
  "$LEGACY_ALWAYS_JSONL" \
  "$OLD_ALWAYS_RUNTIME_MANIFEST" \
  routed-always

run_answer_arm \
  routed_gated \
  "$GATED_SOURCE" \
  "$LEGACY_GATED_JSONL" \
  "$OLD_GATED_RUNTIME_MANIFEST" \
  routed-gated

printf '{"event":"k2_always_gated_job_complete","result_tag":"%s","cache_model_tag":"%s","domain":"%s"}\n' \
  "$RESULT_TAG" "$CACHE_MODEL_TAG" "$DOMAIN"
