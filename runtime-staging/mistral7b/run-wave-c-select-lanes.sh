#!/usr/bin/env bash
set -euo pipefail

repo_root=/root/HySkill-baseline-runtime-matched-20260724
result_root=/root/HySkill-baseline-runtime-matched-20260724-results/mistral7b
instance_root="$repo_root/external/SR-Agents/data/bench/instances"
corpus_path="$repo_root/external/SR-Agents/data/bench/corpus/corpus.json"
validation_root="$repo_root/results/k2-main/mistral7b"
python_path="$repo_root/runtime/runner/bin/python"
api_base=http://127.0.0.1:8000/v1
facts_cloner="$repo_root/staging/wave-c/clone_runtime_facts.py"
decision_generation="$repo_root/staging/mistral7b/select-decision.generation.json"
answer_generation="$repo_root/staging/mistral7b/generation.json"
source_pack_sha256=53abbb8c70e0468d853f81c858af895d51d7b5117efde7a37da19078877a4853

export PYTHONPATH="$repo_root:$repo_root/external/SR-Agents/src"
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

mkdir -p \
  "$result_root/answers" \
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
    printf 'Record count mismatch: context=%s expected=%s actual=%s path=%s\n' \
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
    printf 'Canary prefix SHA mismatch: context=%s expected=%s actual=%s path=%s\n' \
      "$context" "$expected_sha" "$actual_sha" "$path" >&2
    exit 1
  fi
}

build_decision_manifest() {
  local domain=$1
  local facts_path="$result_root/runtime/$domain-select-bm25-decision.runtime-facts.json"
  local manifest_path="$result_root/runtime/$domain-select-bm25-decision.manifest.json"

  "$python_path" "$facts_cloner" \
    --base-manifest "$result_root/runtime/$domain-bare.manifest.json" \
    --job-id "mistral7b-$domain-select-bm25-decision-20260725-v1" \
    --result-tag mistral7b \
    --model mistral7b \
    --domain "$domain" \
    --arm select_bm25 \
    --stage decision \
    --source-pack-sha256 "$source_pack_sha256" \
    --output "$facts_path"

  "$python_path" "$repo_root/scripts/build_runtime_matched_runtime_manifest.py" \
    --runtime-facts "$facts_path" \
    --generation "$decision_generation" \
    --artifact "instances=$instance_root/$domain.json" \
    --artifact "corpus=$corpus_path" \
    --artifact "bm25_candidates=$result_root/bm25/$domain-bm25.json" \
    --code-file hyskill/runtime_matched_execution.py \
    --code-file hyskill/runtime_matched_select.py \
    --code-file scripts/run_runtime_matched_select_decisions.py \
    --code-file external/SR-Agents/src/sragents/corpus.py \
    --code-file external/SR-Agents/src/sragents/llm.py \
    --code-file external/SR-Agents/src/sragents/prompts.py \
    --code-file external/SR-Agents/src/sragents/infer/providers/llm_select.py \
    --repository-root "$repo_root" \
    --output "$manifest_path"
}

run_decision() {
  local domain=$1
  local expected=$2
  local max_new_records=$3
  local run_label=$4
  local output_path="$result_root/decisions/$domain-select-bm25.decisions.jsonl"

  "$python_path" "$repo_root/scripts/run_runtime_matched_select_decisions.py" \
    --instances "$instance_root/$domain.json" \
    --corpus "$corpus_path" \
    --bm25-source "$result_root/bm25/$domain-bm25.json" \
    --output "$output_path" \
    --attempt-log "$result_root/logs/$domain-select-bm25.decision.attempts.jsonl" \
    --runtime-manifest "$result_root/runtime/$domain-select-bm25-decision.manifest.json" \
    --repository-root "$repo_root" \
    --result-tag mistral7b \
    --model mistral7b \
    --api-base "$api_base" \
    --domain "$domain" \
    --expected-count "$expected" \
    --workers 1 \
    --max-new-records "$max_new_records" \
    >"$result_root/logs/$domain-select-bm25.decision.$run_label.log" 2>&1
}

build_answer_manifest() {
  local domain=$1
  local facts_path="$result_root/runtime/$domain-select-bm25-answer.runtime-facts.json"
  local manifest_path="$result_root/runtime/$domain-select-bm25-answer.manifest.json"

  "$python_path" "$facts_cloner" \
    --base-manifest "$result_root/runtime/$domain-bare.manifest.json" \
    --job-id "mistral7b-$domain-select-bm25-answer-20260725-v1" \
    --result-tag mistral7b \
    --model mistral7b \
    --domain "$domain" \
    --arm select_bm25 \
    --stage answer \
    --source-pack-sha256 "$source_pack_sha256" \
    --output "$facts_path"

  "$python_path" "$repo_root/scripts/build_runtime_matched_runtime_manifest.py" \
    --runtime-facts "$facts_path" \
    --generation "$answer_generation" \
    --artifact "instances=$instance_root/$domain.json" \
    --artifact "corpus=$corpus_path" \
    --artifact "select_decisions=$result_root/decisions/$domain-select-bm25.decisions.jsonl" \
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

run_answer() {
  local domain=$1
  local expected=$2
  local max_new_records=$3
  local run_label=$4
  local output_path="$result_root/answers/$domain-select-bm25.jsonl"

  "$python_path" "$repo_root/scripts/run_runtime_matched_select_answers.py" \
    --instances "$instance_root/$domain.json" \
    --corpus "$corpus_path" \
    --decisions "$result_root/decisions/$domain-select-bm25.decisions.jsonl" \
    --output "$output_path" \
    --attempt-log "$result_root/logs/$domain-select-bm25.answer.attempts.jsonl" \
    --runtime-manifest "$result_root/runtime/$domain-select-bm25-answer.manifest.json" \
    --repository-root "$repo_root" \
    --result-tag mistral7b \
    --model mistral7b \
    --api-base "$api_base" \
    --domain "$domain" \
    --expected-count "$expected" \
    --workers 1 \
    --max-new-records "$max_new_records" \
    >"$result_root/logs/$domain-select-bm25.answer.$run_label.log" 2>&1
}

evaluate_answer() {
  local domain=$1
  local expected=$2

  "$python_path" "$repo_root/scripts/evaluate_runtime_matched_baselines.py" \
    --answers "$result_root/answers/$domain-select-bm25.jsonl" \
    --instances "$instance_root/$domain.json" \
    --validation-source "$validation_root/$domain-routed-taus.json" \
    --result-tag mistral7b \
    --served-model mistral7b \
    --domain "$domain" \
    --arm select_bm25 \
    --expected-count "$expected" \
    --output "$result_root/eval/$domain-select-bm25.eval.json"
}

run_domain_lane() {
  local domain=$1
  local expected=$2
  local decision_path="$result_root/decisions/$domain-select-bm25.decisions.jsonl"
  local answer_path="$result_root/answers/$domain-select-bm25.jsonl"
  local decision_canary_sha
  local answer_canary_sha

  build_decision_manifest "$domain"
  run_decision "$domain" "$expected" 5 canary
  verify_count "$decision_path" 5 "$domain Select decision canary"
  decision_canary_sha=$(prefix_sha256 "$decision_path")
  printf '%s\n' "$decision_canary_sha" \
    >"$result_root/logs/$domain-select-bm25.decision.canary.sha256"
  run_decision "$domain" "$expected" 0 full
  verify_count "$decision_path" "$expected" "$domain Select decision full"
  verify_prefix "$decision_path" "$decision_canary_sha" "$domain Select decision"

  build_answer_manifest "$domain"
  run_answer "$domain" "$expected" 5 canary
  verify_count "$answer_path" 5 "$domain Select answer canary"
  answer_canary_sha=$(prefix_sha256 "$answer_path")
  printf '%s\n' "$answer_canary_sha" \
    >"$result_root/logs/$domain-select-bm25.answer.canary.sha256"
  run_answer "$domain" "$expected" 0 full
  verify_count "$answer_path" "$expected" "$domain Select answer full"
  verify_prefix "$answer_path" "$answer_canary_sha" "$domain Select answer"
  evaluate_answer "$domain" "$expected"
  printf 'Select lane complete: domain=%s records=%s\n' "$domain" "$expected"
}

domains=(theoremqa logicbench medcalcbench champ)
counts=(747 760 1100 223)
pids=()
labels=()
for index in "${!domains[@]}"; do
  domain=${domains[$index]}
  expected=${counts[$index]}
  run_domain_lane "$domain" "$expected" \
    >"$result_root/logs/$domain-select-bm25.lane.log" 2>&1 &
  pid=$!
  printf '%s\n' "$pid" >"$result_root/logs/$domain-select-bm25.lane.pid"
  pids+=("$pid")
  labels+=("$domain")
done

status=0
for index in "${!pids[@]}"; do
  pid=${pids[$index]}
  domain=${labels[$index]}
  if ! wait "$pid"; then
    printf 'Select lane failed: domain=%s pid=%s log=%s\n' \
      "$domain" "$pid" "$result_root/logs/$domain-select-bm25.lane.log" >&2
    status=1
  fi
done
exit "$status"
