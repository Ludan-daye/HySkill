#!/usr/bin/env bash
set -euo pipefail

if (( $# != 14 )); then
  printf '%s\n' \
    "usage: run_rerank_after_select_lane.sh REPO_ROOT SOURCE_ROOT RESULT_TAG SERVED_MODEL API_BASE PYTHON_PATH SELECT_LANE_PID SELECT_LANE_LOG SELECT_FIRST_DOMAIN SELECT_SECOND_DOMAIN RERANK_FIRST_DOMAIN RERANK_FIRST_EXPECTED RERANK_SECOND_DOMAIN RERANK_SECOND_EXPECTED" >&2
  exit 2
fi

repo_root=$1
source_root=$2
result_tag=$3
served_model=$4
api_base=$5
python_path=$6
select_lane_pid=$7
select_lane_log=$8
select_first_domain=$9
select_second_domain=${10}
rerank_first_domain=${11}
rerank_first_expected=${12}
rerank_second_domain=${13}
rerank_second_expected=${14}

result_root="$repo_root/results/baselines-runtime-matched-v1/$result_tag"
claim_path="$result_root/runtime/rerank-handoff-$rerank_first_domain-$rerank_second_domain.claim"
expected_select_marker="lane complete: first=$select_first_domain second=$select_second_domain result_tag=$result_tag"
rerank_lane_script="$repo_root/runtime-staging/wave-c/run_rerank_lane.sh"

select_command() {
  if [[ ! -r "/proc/$select_lane_pid/cmdline" ]]; then
    return
  fi
  tr '\0' ' ' <"/proc/$select_lane_pid/cmdline" 2>/dev/null || true
}

initial_command=$(select_command)
if [[ "$initial_command" != *"run_select_lane.sh"* ]] \
  || [[ "$initial_command" != *"$result_tag"* ]]; then
  printf 'select lane PID identity mismatch: pid=%s command=%s\n' \
    "$select_lane_pid" "$initial_command" >&2
  exit 1
fi

printf 'waiting for Select lane: pid=%s result_tag=%s first=%s second=%s\n' \
  "$select_lane_pid" "$result_tag" "$select_first_domain" "$select_second_domain"
while [[ -r "/proc/$select_lane_pid/cmdline" ]]; do
  current_command=$(select_command)
  if [[ "$current_command" != *"run_select_lane.sh"* ]] \
    || [[ "$current_command" != *"$result_tag"* ]]; then
    break
  fi
  sleep 10
done

if [[ ! -f "$select_lane_log" ]] \
  || ! grep -Fqx "$expected_select_marker" "$select_lane_log"; then
  printf 'Select lane did not finish cleanly: pid=%s marker=%s log=%s\n' \
    "$select_lane_pid" "$expected_select_marker" "$select_lane_log" >&2
  exit 1
fi

endpoint_ready=false
for endpoint_attempt in 1 2 3 4 5 6; do
  if curl -fsS "$api_base/models" | grep -Fq "\"id\":\"$served_model\""; then
    endpoint_ready=true
    break
  fi
  printf 'endpoint health retry: attempt=%s api_base=%s model=%s\n' \
    "$endpoint_attempt" "$api_base" "$served_model" >&2
  sleep 10
done
if [[ "$endpoint_ready" != true ]]; then
  printf 'endpoint health gate failed: api_base=%s model=%s\n' \
    "$api_base" "$served_model" >&2
  exit 1
fi

active_jobs=$(
  pgrep -af \
    'run_runtime_matched_(select|rerank)_(decisions|answers)\.py' \
    | grep -F -- "--api-base $api_base" \
    || true
)
active_count=$(
  printf '%s\n' "$active_jobs" \
    | sed '/^[[:space:]]*$/d' \
    | wc -l \
    | tr -d '[:space:]'
)
if (( active_count > 1 )); then
  printf 'native-client concurrency gate failed: api_base=%s active_jobs=%s\n%s\n' \
    "$api_base" "$active_count" "$active_jobs" >&2
  exit 1
fi
if [[ -n "$active_jobs" && "$active_jobs" != *"--workers 3"* ]]; then
  printf 'native-client worker identity mismatch: api_base=%s jobs=%s\n' \
    "$api_base" "$active_jobs" >&2
  exit 1
fi

if ! mkdir "$claim_path"; then
  printf 'Rerank handoff is already claimed: path=%s\n' "$claim_path" >&2
  exit 1
fi

printf 'Rerank handoff gate passed: result_tag=%s active_workers_before_start=%s first=%s second=%s\n' \
  "$result_tag" "$((active_count * 3))" "$rerank_first_domain" "$rerank_second_domain"
exec bash "$rerank_lane_script" \
  "$repo_root" \
  "$source_root" \
  "$result_tag" \
  "$served_model" \
  "$api_base" \
  "$python_path" \
  "$rerank_first_domain" \
  "$rerank_first_expected" \
  "$rerank_second_domain" \
  "$rerank_second_expected"
