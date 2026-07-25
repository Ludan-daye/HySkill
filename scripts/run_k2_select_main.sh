#!/usr/bin/env bash
# Run one eligible model-domain routed Select K=2 downstream job.
set -euo pipefail

: "${REPOSITORY_ROOT:?REPOSITORY_ROOT is required}"
: "${PYTHON_BIN:?PYTHON_BIN is required and must be Python 3.10-3.12}"
: "${RESULT_TAG:?RESULT_TAG is required}"
: "${SERVED_MODEL:?SERVED_MODEL is required}"
: "${API_BASE:?API_BASE is required}"
: "${DOMAIN:?DOMAIN is required}"
: "${EXPECTED_COUNT:?EXPECTED_COUNT is required}"
: "${INSTANCES:?INSTANCES is required}"
: "${CORPUS:?CORPUS is required}"
: "${ROUTED_SOURCE:?ROUTED_SOURCE is required}"
: "${TAUS:?TAUS is required}"
: "${NEW_RUNTIME_MANIFEST:?NEW_RUNTIME_MANIFEST is required}"
: "${OLD_SELECT_RUNTIME_MANIFEST:?OLD_SELECT_RUNTIME_MANIFEST is required}"
: "${LEGACY_SELECT_JSONL:?LEGACY_SELECT_JSONL is required}"
: "${SELECTOR_CODE_BUNDLE_SHA256:?SELECTOR_CODE_BUNDLE_SHA256 is required}"
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

SELECTION="$JOB_ROOT/$DOMAIN-routed-select.selection.jsonl"
SELECTED_SOURCE="$JOB_ROOT/$DOMAIN-routed-select-source.json"

"$PYTHON_BIN" scripts/run_select_only.py \
  --instances "$INSTANCES" \
  --corpus "$CORPUS" \
  --source "$ROUTED_SOURCE" \
  --output "$SELECTION" \
  --selected-source "$SELECTED_SOURCE" \
  --attempt-log "$LOG_ROOT/$DOMAIN-routed-select.attempts.jsonl" \
  --runtime-manifest "$NEW_RUNTIME_MANIFEST" \
  --code-bundle-sha256 "$SELECTOR_CODE_BUNDLE_SHA256" \
  --model "$SERVED_MODEL" \
  --api-base "$API_BASE" \
  --domain "$DOMAIN" \
  --workers "$WORKERS"

"$PYTHON_BIN" scripts/validate_k2_downstream.py selection \
  --instances "$INSTANCES" \
  --corpus "$CORPUS" \
  --source "$ROUTED_SOURCE" \
  --selection "$SELECTION" \
  --selected-source "$SELECTED_SOURCE" \
  --attempt-log "$LOG_ROOT/$DOMAIN-routed-select.attempts.jsonl" \
  --runtime-manifest "$NEW_RUNTIME_MANIFEST" \
  --code-bundle-sha256 "$SELECTOR_CODE_BUNDLE_SHA256" \
  --model "$SERVED_MODEL" \
  --domain "$DOMAIN" \
  --expected-count "$EXPECTED_COUNT" \
  --output "$AUDIT_ROOT/$DOMAIN-routed-select.selection-completion.json"

"$PYTHON_BIN" scripts/export_k2_select_loading_decisions.py \
  --instances "$INSTANCES" \
  --selected-source "$SELECTED_SOURCE" \
  --selection "$SELECTION" \
  --taus "$TAUS" \
  --model "$RESULT_TAG" \
  --domain "$DOMAIN" \
  --expected-count "$EXPECTED_COUNT" \
  --output "$JOB_ROOT/$DOMAIN-routed-select.loading.jsonl"

"$PYTHON_BIN" scripts/audit_k2_reuse.py \
  --instances "$INSTANCES" \
  --corpus "$CORPUS" \
  --decision-source "$SELECTED_SOURCE" \
  --legacy-jsonl "$LEGACY_SELECT_JSONL" \
  --old-runtime-manifest "$OLD_SELECT_RUNTIME_MANIFEST" \
  --new-runtime-manifest "$NEW_RUNTIME_MANIFEST" \
  --result-tag "$RESULT_TAG" \
  --arm routed_select \
  --domain "$DOMAIN" \
  --audit-output "$AUDIT_ROOT/$DOMAIN-routed-select.reuse.jsonl" \
  --preseed-output "$AUDIT_ROOT/$DOMAIN-routed-select.preseed.jsonl" \
  --pending-output "$AUDIT_ROOT/$DOMAIN-routed-select.pending.json"

"$PYTHON_BIN" scripts/run_k2_answers.py \
  --instances "$INSTANCES" \
  --corpus "$CORPUS" \
  --decision-source "$SELECTED_SOURCE" \
  --audit "$AUDIT_ROOT/$DOMAIN-routed-select.reuse.jsonl" \
  --preseed "$AUDIT_ROOT/$DOMAIN-routed-select.preseed.jsonl" \
  --output "$JOB_ROOT/$DOMAIN-routed-select.jsonl" \
  --attempt-log "$LOG_ROOT/$DOMAIN-routed-select-answer.attempts.jsonl" \
  --runtime-manifest "$NEW_RUNTIME_MANIFEST" \
  --model "$SERVED_MODEL" \
  --api-base "$API_BASE" \
  --arm routed_select \
  --domain "$DOMAIN" \
  --workers "$WORKERS"

"$PYTHON_BIN" scripts/validate_k2_downstream.py answer \
  --instances "$INSTANCES" \
  --corpus "$CORPUS" \
  --decision-source "$SELECTED_SOURCE" \
  --answers "$JOB_ROOT/$DOMAIN-routed-select.jsonl" \
  --audit "$AUDIT_ROOT/$DOMAIN-routed-select.reuse.jsonl" \
  --legacy-jsonl "$LEGACY_SELECT_JSONL" \
  --old-runtime-manifest "$OLD_SELECT_RUNTIME_MANIFEST" \
  --runtime-manifest "$NEW_RUNTIME_MANIFEST" \
  --result-tag "$RESULT_TAG" \
  --model "$SERVED_MODEL" \
  --arm routed_select \
  --domain "$DOMAIN" \
  --expected-count "$EXPECTED_COUNT" \
  --output "$AUDIT_ROOT/$DOMAIN-routed-select.completion.json"

printf '{"event":"k2_select_job_complete","result_tag":"%s","domain":"%s"}\n' \
  "$RESULT_TAG" "$DOMAIN"
