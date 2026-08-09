#!/usr/bin/env bash
set -euo pipefail

if (( $# != 13 )); then
  printf '%s\n' \
    "usage: run_select_lane.sh REPO_ROOT SOURCE_ROOT RESULT_TAG SERVED_MODEL API_BASE PYTHON_PATH FIRST_DOMAIN FIRST_EXPECTED FIRST_PID FIRST_CANARY_SHA SECOND_DOMAIN SECOND_EXPECTED SECOND_CANARY_SHA" >&2
  exit 2
fi

repo_root=$1
source_root=$2
result_tag=$3
served_model=$4
api_base=$5
python_path=$6
first_domain=$7
first_expected=$8
first_pid=$9
first_canary_sha=${10}
second_domain=${11}
second_expected=${12}
second_canary_sha=${13}

result_root="$repo_root/results/baselines-runtime-matched-v1/$result_tag"
instance_root="$repo_root/external/SR-Agents/data/bench/instances"
corpus_path="$repo_root/external/SR-Agents/data/bench/corpus/corpus.json"
validation_root="$source_root/results/k2-main/$result_tag"
generation_path="$repo_root/runtime-staging/wave-c/direct-answer.generation.json"
source_pack_sha256=53abbb8c70e0468d853f81c858af895d51d7b5117efde7a37da19078877a4853

export PYTHONPATH="$repo_root:$repo_root/external/SR-Agents/src"
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

mkdir -p \
  "$result_root/answers" \
  "$result_root/eval" \
  "$result_root/logs" \
  "$result_root/runtime"

record_count() {
  local path=$1
  wc -l <"$path" | tr -d '[:space:]'
}

prefix_sha256() {
  local path=$1
  head -n 5 "$path" | sha256sum | awk '{print $1}'
}

verify_count() {
  local path=$1
  local expected=$2
  local context=$3
  local actual
  actual=$(record_count "$path")
  if [[ "$actual" != "$expected" ]]; then
    printf 'record count mismatch: context=%s expected=%s actual=%s path=%s\n' \
      "$context" "$expected" "$actual" "$path" >&2
    exit 1
  fi
}

verify_prefix() {
  local path=$1
  local expected_sha=$2
  local context=$3
  local actual_sha
  actual_sha=$(prefix_sha256 "$path")
  if [[ "$actual_sha" != "$expected_sha" ]]; then
    printf 'canary prefix SHA mismatch: context=%s expected=%s actual=%s path=%s\n' \
      "$context" "$expected_sha" "$actual_sha" "$path" >&2
    exit 1
  fi
}

wait_for_initial_decision() {
  local domain=$1
  local expected=$2
  local pid=$3
  local canary_sha=$4
  local output_path="$result_root/decisions/$domain-select-bm25.decisions.jsonl"

  while kill -0 "$pid" 2>/dev/null; do
    sleep 10
  done
  verify_count "$output_path" "$expected" "$domain decision full"
  verify_prefix "$output_path" "$canary_sha" "$domain decision"
  printf 'decision complete: domain=%s records=%s pid=%s\n' \
    "$domain" "$expected" "$pid"
}

run_decision_full() {
  local domain=$1
  local expected=$2
  local canary_sha=$3
  local output_path="$result_root/decisions/$domain-select-bm25.decisions.jsonl"
  local log_path="$result_root/logs/$domain-select-bm25.decision.full.log"
  local pid_path="$result_root/logs/$domain-select-bm25.decision.full.pid"
  local pid

  "$python_path" "$repo_root/scripts/run_runtime_matched_select_decisions.py" \
    --instances "$instance_root/$domain.json" \
    --corpus "$corpus_path" \
    --bm25-source "$repo_root/results/baselines-runtime-matched-v1/shared-bm25/$domain-bm25.json" \
    --output "$output_path" \
    --attempt-log "$result_root/logs/$domain-select-bm25.decision.attempts.jsonl" \
    --runtime-manifest "$result_root/runtime/$domain-select-bm25-decision.manifest.json" \
    --repository-root "$repo_root" \
    --result-tag "$result_tag" \
    --model "$served_model" \
    --api-base "$api_base" \
    --domain "$domain" \
    --expected-count "$expected" \
    --workers 3 \
    --max-new-records 0 \
    >"$log_path" 2>&1 &
  pid=$!
  printf '%s\n' "$pid" >"$pid_path"
  printf 'decision started: domain=%s pid=%s workers=3\n' "$domain" "$pid"
  if ! wait "$pid"; then
    printf 'decision failed: domain=%s pid=%s log=%s\n' \
      "$domain" "$pid" "$log_path" >&2
    exit 1
  fi
  verify_count "$output_path" "$expected" "$domain decision full"
  verify_prefix "$output_path" "$canary_sha" "$domain decision"
  printf 'decision complete: domain=%s records=%s pid=%s\n' \
    "$domain" "$expected" "$pid"
}

build_answer_manifest() {
  local domain=$1
  local facts_path="$result_root/runtime/$domain-select-bm25-answer.runtime-facts.json"
  local manifest_path="$result_root/runtime/$domain-select-bm25-answer.manifest.json"
  local decision_path="$result_root/decisions/$domain-select-bm25.decisions.jsonl"

  "$python_path" "$repo_root/runtime-staging/wave-c/clone_runtime_facts.py" \
    --base-manifest "$result_root/runtime/$domain-bare.manifest.json" \
    --job-id "$result_tag-$domain-select-bm25-answer-20260725-v1" \
    --result-tag "$result_tag" \
    --model "$served_model" \
    --domain "$domain" \
    --arm select_bm25 \
    --stage answer \
    --source-pack-sha256 "$source_pack_sha256" \
    --output "$facts_path"

  "$python_path" "$repo_root/scripts/build_runtime_matched_runtime_manifest.py" \
    --runtime-facts "$facts_path" \
    --generation "$generation_path" \
    --artifact "instances=$instance_root/$domain.json" \
    --artifact "corpus=$corpus_path" \
    --artifact "select_decisions=$decision_path" \
    --code-file hyskill/runtime_matched_execution.py \
    --code-file hyskill/runtime_matched_select.py \
    --code-file scripts/run_runtime_matched_select_answers.py \
    --code-file external/SR-Agents/src/sragents/llm.py \
    --code-file external/SR-Agents/src/sragents/prompts.py \
    --code-file external/SR-Agents/src/sragents/infer/base.py \
    --code-file external/SR-Agents/src/sragents/infer/engines/direct.py \
    --code-file external/SR-Agents/src/sragents/infer/engines/tool_loop.py \
    --repository-root "$repo_root" \
    --output "$manifest_path"
}

run_answer_canary() {
  local domain=$1
  local expected=$2
  local output_path="$result_root/answers/$domain-select-bm25.jsonl"
  local log_path="$result_root/logs/$domain-select-bm25.answer.canary.log"
  local pid_path="$result_root/logs/$domain-select-bm25.answer.canary.pid"
  local sha_path="$result_root/logs/$domain-select-bm25.answer.canary.sha256"
  local pid
  local canary_sha

  if [[ -e "$output_path" ]]; then
    printf 'answer canary output already exists: domain=%s path=%s\n' \
      "$domain" "$output_path" >&2
    exit 1
  fi

  "$python_path" "$repo_root/scripts/run_runtime_matched_select_answers.py" \
    --instances "$instance_root/$domain.json" \
    --corpus "$corpus_path" \
    --decisions "$result_root/decisions/$domain-select-bm25.decisions.jsonl" \
    --output "$output_path" \
    --attempt-log "$result_root/logs/$domain-select-bm25.answer.attempts.jsonl" \
    --runtime-manifest "$result_root/runtime/$domain-select-bm25-answer.manifest.json" \
    --repository-root "$repo_root" \
    --result-tag "$result_tag" \
    --model "$served_model" \
    --api-base "$api_base" \
    --domain "$domain" \
    --expected-count "$expected" \
    --workers 3 \
    --max-new-records 5 \
    >"$log_path" 2>&1 &
  pid=$!
  printf '%s\n' "$pid" >"$pid_path"
  printf 'answer canary started: domain=%s pid=%s workers=3\n' "$domain" "$pid"
  if ! wait "$pid"; then
    printf 'answer canary failed: domain=%s pid=%s log=%s\n' \
      "$domain" "$pid" "$log_path" >&2
    exit 1
  fi
  verify_count "$output_path" 5 "$domain answer canary"
  canary_sha=$(sha256sum "$output_path" | awk '{print $1}')
  printf '%s\n' "$canary_sha" >"$sha_path"
  printf 'answer canary complete: domain=%s records=5 sha256=%s\n' \
    "$domain" "$canary_sha"
}

run_answer_full() {
  local domain=$1
  local expected=$2
  local output_path="$result_root/answers/$domain-select-bm25.jsonl"
  local log_path="$result_root/logs/$domain-select-bm25.answer.full.log"
  local pid_path="$result_root/logs/$domain-select-bm25.answer.full.pid"
  local sha_path="$result_root/logs/$domain-select-bm25.answer.canary.sha256"
  local pid
  local canary_sha

  canary_sha=$(<"$sha_path")
  "$python_path" "$repo_root/scripts/run_runtime_matched_select_answers.py" \
    --instances "$instance_root/$domain.json" \
    --corpus "$corpus_path" \
    --decisions "$result_root/decisions/$domain-select-bm25.decisions.jsonl" \
    --output "$output_path" \
    --attempt-log "$result_root/logs/$domain-select-bm25.answer.attempts.jsonl" \
    --runtime-manifest "$result_root/runtime/$domain-select-bm25-answer.manifest.json" \
    --repository-root "$repo_root" \
    --result-tag "$result_tag" \
    --model "$served_model" \
    --api-base "$api_base" \
    --domain "$domain" \
    --expected-count "$expected" \
    --workers 3 \
    --max-new-records 0 \
    >"$log_path" 2>&1 &
  pid=$!
  printf '%s\n' "$pid" >"$pid_path"
  printf 'answer full started: domain=%s pid=%s workers=3\n' "$domain" "$pid"
  if ! wait "$pid"; then
    printf 'answer full failed: domain=%s pid=%s log=%s\n' \
      "$domain" "$pid" "$log_path" >&2
    exit 1
  fi
  verify_count "$output_path" "$expected" "$domain answer full"
  verify_prefix "$output_path" "$canary_sha" "$domain answer"
  printf 'answer full complete: domain=%s records=%s pid=%s\n' \
    "$domain" "$expected" "$pid"
}

evaluate_answer() {
  local domain=$1
  local expected=$2

  "$python_path" "$repo_root/scripts/evaluate_runtime_matched_baselines.py" \
    --answers "$result_root/answers/$domain-select-bm25.jsonl" \
    --instances "$instance_root/$domain.json" \
    --validation-source "$validation_root/$domain-routed-taus.json" \
    --result-tag "$result_tag" \
    --served-model "$served_model" \
    --domain "$domain" \
    --arm select_bm25 \
    --expected-count "$expected" \
    --output "$result_root/eval/$domain-select-bm25.eval.json"
  printf 'evaluation complete: domain=%s records=%s\n' "$domain" "$expected"
}

run_answer_pipeline() {
  local domain=$1
  local expected=$2

  build_answer_manifest "$domain"
  run_answer_canary "$domain" "$expected"
  run_answer_full "$domain" "$expected"
  evaluate_answer "$domain" "$expected"
}

wait_for_initial_decision \
  "$first_domain" \
  "$first_expected" \
  "$first_pid" \
  "$first_canary_sha"
run_answer_pipeline "$first_domain" "$first_expected"
run_decision_full "$second_domain" "$second_expected" "$second_canary_sha"
run_answer_pipeline "$second_domain" "$second_expected"
printf 'lane complete: first=%s second=%s result_tag=%s\n' \
  "$first_domain" "$second_domain" "$result_tag"
