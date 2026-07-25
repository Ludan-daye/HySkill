#!/usr/bin/env bash
set -euo pipefail

if (( $# != 10 )); then
  printf '%s\n' \
    "usage: run_rerank_lane.sh REPO_ROOT SOURCE_ROOT RESULT_TAG SERVED_MODEL API_BASE PYTHON_PATH FIRST_DOMAIN FIRST_EXPECTED SECOND_DOMAIN SECOND_EXPECTED" >&2
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
second_domain=$9
second_expected=${10}

result_root="$repo_root/results/baselines-runtime-matched-v1/$result_tag"
instance_root="$repo_root/external/SR-Agents/data/bench/instances"
corpus_path="$repo_root/external/SR-Agents/data/bench/corpus/corpus.json"
validation_root="$source_root/results/k2-main/$result_tag"
decision_generation="$repo_root/runtime-staging/wave-c/rerank-decision.generation.json"
answer_generation="$repo_root/runtime-staging/wave-c/direct-answer.generation.json"
facts_cloner="$repo_root/runtime-staging/wave-c/clone_runtime_facts.py"
sragents_revision=277fd8d2bbd7d3b81a5cf4ffa6e87e18c7906e4f
source_pack_sha256=53abbb8c70e0468d853f81c858af895d51d7b5117efde7a37da19078877a4853

export PYTHONPATH="$repo_root:$repo_root/external/SR-Agents/src"
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

mkdir -p \
  "$result_root/answers" \
  "$result_root/audits" \
  "$result_root/decisions" \
  "$result_root/eval" \
  "$result_root/logs" \
  "$result_root/runtime"

record_count() {
  local path=$1
  if [[ ! -f "$path" ]]; then
    printf '0\n'
    return
  fi
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

build_decision_manifest() {
  local domain=$1
  local facts_path="$result_root/runtime/$domain-always_rerank-decision.runtime-facts.json"
  local manifest_path="$result_root/runtime/$domain-always_rerank-decision.manifest.json"
  local bm25_path="$repo_root/results/baselines-runtime-matched-v1/shared-bm25/$domain-bm25.json"

  "$python_path" "$facts_cloner" \
    --base-manifest "$result_root/runtime/$domain-bare.manifest.json" \
    --job-id "$result_tag-$domain-always_rerank-decision-20260725-v1" \
    --result-tag "$result_tag" \
    --model "$served_model" \
    --domain "$domain" \
    --arm always_rerank \
    --stage decision \
    --source-pack-sha256 "$source_pack_sha256" \
    --output "$facts_path"

  "$python_path" "$repo_root/scripts/build_runtime_matched_runtime_manifest.py" \
    --runtime-facts "$facts_path" \
    --generation "$decision_generation" \
    --artifact "instances=$instance_root/$domain.json" \
    --artifact "corpus=$corpus_path" \
    --artifact "bm25_candidates=$bm25_path" \
    --code-file hyskill/runtime_matched_execution.py \
    --code-file hyskill/runtime_matched_rerank.py \
    --code-file scripts/run_runtime_matched_rerank_decisions.py \
    --code-file external/SR-Agents/src/sragents/corpus.py \
    --code-file external/SR-Agents/src/sragents/llm.py \
    --code-file external/SR-Agents/src/sragents/prompts.py \
    --code-file external/SR-Agents/src/sragents/retrieve/llm_rerank.py \
    --repository-root "$repo_root" \
    --output "$manifest_path"
}

run_decision() {
  local domain=$1
  local max_new_records=$2
  local run_label=$3
  local output_path="$result_root/decisions/$domain-always_rerank.decisions.jsonl"
  local bm25_path="$repo_root/results/baselines-runtime-matched-v1/shared-bm25/$domain-bm25.json"

  "$python_path" "$repo_root/scripts/run_runtime_matched_rerank_decisions.py" \
    --instances "$instance_root/$domain.json" \
    --corpus "$corpus_path" \
    --bm25-source "$bm25_path" \
    --output "$output_path" \
    --attempt-log "$result_root/logs/$domain-always_rerank.decision.attempts.jsonl" \
    --runtime-manifest "$result_root/runtime/$domain-always_rerank-decision.manifest.json" \
    --result-tag "$result_tag" \
    --model "$served_model" \
    --api-base "$api_base" \
    --domain "$domain" \
    --workers 3 \
    --max-new-records "$max_new_records" \
    --sragents-checkout "$repo_root/external/SR-Agents" \
    --repository-root "$repo_root" \
    --sragents-revision "$sragents_revision" \
    >"$result_root/logs/$domain-always_rerank.decision.$run_label.log" 2>&1
}

build_answer_manifest() {
  local domain=$1
  local facts_path="$result_root/runtime/$domain-always_rerank-answer.runtime-facts.json"
  local manifest_path="$result_root/runtime/$domain-always_rerank-answer.manifest.json"
  local decision_path="$result_root/decisions/$domain-always_rerank.decisions.jsonl"

  "$python_path" "$facts_cloner" \
    --base-manifest "$result_root/runtime/$domain-bare.manifest.json" \
    --job-id "$result_tag-$domain-always_rerank-answer-20260725-v1" \
    --result-tag "$result_tag" \
    --model "$served_model" \
    --domain "$domain" \
    --arm always_rerank \
    --stage answer \
    --source-pack-sha256 "$source_pack_sha256" \
    --output "$facts_path"

  "$python_path" "$repo_root/scripts/build_runtime_matched_runtime_manifest.py" \
    --runtime-facts "$facts_path" \
    --generation "$answer_generation" \
    --artifact "instances=$instance_root/$domain.json" \
    --artifact "corpus=$corpus_path" \
    --artifact "rerank_decisions=$decision_path" \
    --code-file hyskill/runtime_matched_execution.py \
    --code-file hyskill/runtime_matched_rerank.py \
    --code-file scripts/run_runtime_matched_rerank_answers.py \
    --code-file external/SR-Agents/src/sragents/llm.py \
    --code-file external/SR-Agents/src/sragents/prompts.py \
    --code-file external/SR-Agents/src/sragents/infer/base.py \
    --code-file external/SR-Agents/src/sragents/infer/engines/direct.py \
    --code-file external/SR-Agents/src/sragents/infer/engines/tool_loop.py \
    --repository-root "$repo_root" \
    --output "$manifest_path"
}

run_answer() {
  local domain=$1
  local max_new_records=$2
  local run_label=$3
  local output_path="$result_root/answers/$domain-always_rerank.jsonl"

  "$python_path" "$repo_root/scripts/run_runtime_matched_rerank_answers.py" \
    --instances "$instance_root/$domain.json" \
    --corpus "$corpus_path" \
    --decisions "$result_root/decisions/$domain-always_rerank.decisions.jsonl" \
    --output "$output_path" \
    --attempt-log "$result_root/logs/$domain-always_rerank.answer.attempts.jsonl" \
    --runtime-manifest "$result_root/runtime/$domain-always_rerank-answer.manifest.json" \
    --result-tag "$result_tag" \
    --model "$served_model" \
    --api-base "$api_base" \
    --domain "$domain" \
    --workers 3 \
    --max-new-records "$max_new_records" \
    --sragents-checkout "$repo_root/external/SR-Agents" \
    --repository-root "$repo_root" \
    --sragents-revision "$sragents_revision" \
    >"$result_root/logs/$domain-always_rerank.answer.$run_label.log" 2>&1
}

evaluate_answer() {
  local domain=$1
  local expected=$2

  "$python_path" "$repo_root/scripts/evaluate_runtime_matched_baselines.py" \
    --answers "$result_root/answers/$domain-always_rerank.jsonl" \
    --instances "$instance_root/$domain.json" \
    --validation-source "$validation_root/$domain-routed-taus.json" \
    --result-tag "$result_tag" \
    --served-model "$served_model" \
    --domain "$domain" \
    --arm always_rerank \
    --expected-count "$expected" \
    --output "$result_root/eval/$domain-always_rerank.eval.json"
}

audit_domain() {
  local domain=$1
  local expected=$2

  "$python_path" "$repo_root/scripts/audit_runtime_matched_native_domain.py" \
    --instances "$instance_root/$domain.json" \
    --decisions "$result_root/decisions/$domain-always_rerank.decisions.jsonl" \
    --answers "$result_root/answers/$domain-always_rerank.jsonl" \
    --decision-attempt-log "$result_root/logs/$domain-always_rerank.decision.attempts.jsonl" \
    --answer-attempt-log "$result_root/logs/$domain-always_rerank.answer.attempts.jsonl" \
    --decision-full-log "$result_root/logs/$domain-always_rerank.decision.full.log" \
    --answer-full-log "$result_root/logs/$domain-always_rerank.answer.full.log" \
    --decision-manifest "$result_root/runtime/$domain-always_rerank-decision.manifest.json" \
    --answer-manifest "$result_root/runtime/$domain-always_rerank-answer.manifest.json" \
    --evaluation "$result_root/eval/$domain-always_rerank.eval.json" \
    --repository-root "$repo_root" \
    --result-tag "$result_tag" \
    --served-model "$served_model" \
    --domain "$domain" \
    --arm always_rerank \
    --expected-count "$expected" \
    --output "$result_root/audits/$domain-always_rerank.audit.json"
}

run_decision_pipeline() {
  local domain=$1
  local expected=$2
  local output_path="$result_root/decisions/$domain-always_rerank.decisions.jsonl"
  local sha_path="$result_root/logs/$domain-always_rerank.decision.canary.sha256"
  local observed
  local canary_sha

  build_decision_manifest "$domain"
  observed=$(record_count "$output_path")
  if (( observed == 0 )); then
    run_decision "$domain" 5 canary
    verify_count "$output_path" 5 "$domain Rerank decision canary"
  elif (( observed < 5 || observed > expected )); then
    printf 'invalid resumable decision count: domain=%s observed=%s expected=%s\n' \
      "$domain" "$observed" "$expected" >&2
    exit 1
  fi
  if [[ ! -f "$sha_path" ]]; then
    canary_sha=$(prefix_sha256 "$output_path")
    printf '%s\n' "$canary_sha" >"$sha_path"
  fi
  canary_sha=$(<"$sha_path")
  verify_prefix "$output_path" "$canary_sha" "$domain Rerank decision"
  observed=$(record_count "$output_path")
  if (( observed < expected )); then
    run_decision "$domain" 0 full
  fi
  verify_count "$output_path" "$expected" "$domain Rerank decision full"
  verify_prefix "$output_path" "$canary_sha" "$domain Rerank decision"
}

run_answer_pipeline() {
  local domain=$1
  local expected=$2
  local output_path="$result_root/answers/$domain-always_rerank.jsonl"
  local sha_path="$result_root/logs/$domain-always_rerank.answer.canary.sha256"
  local observed
  local canary_sha

  build_answer_manifest "$domain"
  observed=$(record_count "$output_path")
  if (( observed == 0 )); then
    run_answer "$domain" 5 canary
    verify_count "$output_path" 5 "$domain Rerank answer canary"
  elif (( observed < 5 || observed > expected )); then
    printf 'invalid resumable answer count: domain=%s observed=%s expected=%s\n' \
      "$domain" "$observed" "$expected" >&2
    exit 1
  fi
  if [[ ! -f "$sha_path" ]]; then
    canary_sha=$(prefix_sha256 "$output_path")
    printf '%s\n' "$canary_sha" >"$sha_path"
  fi
  canary_sha=$(<"$sha_path")
  verify_prefix "$output_path" "$canary_sha" "$domain Rerank answer"
  observed=$(record_count "$output_path")
  if (( observed < expected )); then
    run_answer "$domain" 0 full
  fi
  verify_count "$output_path" "$expected" "$domain Rerank answer full"
  verify_prefix "$output_path" "$canary_sha" "$domain Rerank answer"
}

run_domain_pipeline() {
  local domain=$1
  local expected=$2

  run_decision_pipeline "$domain" "$expected"
  run_answer_pipeline "$domain" "$expected"
  evaluate_answer "$domain" "$expected"
  audit_domain "$domain" "$expected"
  printf 'Rerank domain complete: domain=%s records=%s result_tag=%s\n' \
    "$domain" "$expected" "$result_tag"
}

run_domain_pipeline "$first_domain" "$first_expected"
run_domain_pipeline "$second_domain" "$second_expected"
printf 'Rerank lane complete: first=%s second=%s result_tag=%s\n' \
  "$first_domain" "$second_domain" "$result_tag"
