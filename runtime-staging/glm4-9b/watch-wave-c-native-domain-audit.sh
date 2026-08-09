#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  printf 'Usage: %s DOMAIN ARM\n' "$0" >&2
  exit 2
fi
domain=$1
arm=$2
case "$domain" in
  theoremqa) expected=747 ;;
  logicbench) expected=760 ;;
  medcalcbench) expected=1100 ;;
  champ) expected=223 ;;
  *)
    printf 'Unsupported domain: %s\n' "$domain" >&2
    exit 2
    ;;
esac
case "$arm" in
  always_rerank) terminal_event=rerank_answer_complete ;;
  select_bm25) terminal_event=runtime_matched_select_answers_complete ;;
  *)
    printf 'Unsupported arm: %s\n' "$arm" >&2
    exit 2
    ;;
esac

repo_root=/root/HySkill-baseline-runtime-matched-20260724/repo
source_root=/root/HySkill-k-run-20260723
result_root=/root/HySkill-baseline-runtime-matched-20260724-results/glm4-9b
instance_root="$source_root/external/SR-Agents/data/bench/instances"
validation_root="$source_root/results/k2-main/glm4-9b"
python_path="$source_root/.venv/bin/python"
answers_path="$result_root/answers/$domain-$arm.jsonl"
answer_full_log="$result_root/logs/$domain-$arm.answer.full.log"
evaluation_path="$result_root/eval/$domain-$arm.eval.json"
audit_path="$result_root/audits/$domain-$arm.audit.json"
export PYTHONPATH="$repo_root:$repo_root/external/SR-Agents/src"

mkdir -p "$result_root/eval" "$result_root/audits"
while true; do
  observed=0
  if [[ -f "$answers_path" ]]; then
    observed=$(wc -l <"$answers_path")
  fi
  if [[ "$observed" -gt "$expected" ]]; then
    printf \
      'Native answer count exceeds protocol: domain=%s arm=%s expected=%s observed=%s\n' \
      "$domain" \
      "$arm" \
      "$expected" \
      "$observed" >&2
    exit 1
  fi
  if [[ "$observed" -eq "$expected" ]] \
    && grep -q "\"event\":\"$terminal_event\"" "$answer_full_log"; then
    break
  fi
  sleep 15
done

"$python_path" "$repo_root/scripts/evaluate_runtime_matched_baselines.py" \
  --answers "$answers_path" \
  --instances "$instance_root/$domain.json" \
  --validation-source "$validation_root/$domain-routed-taus.json" \
  --result-tag glm4-9b \
  --served-model glm4-9b \
  --domain "$domain" \
  --arm "$arm" \
  --expected-count "$expected" \
  --output "$evaluation_path"

"$python_path" "$repo_root/scripts/audit_runtime_matched_native_domain.py" \
  --instances "$instance_root/$domain.json" \
  --decisions "$result_root/decisions/$domain-$arm.jsonl" \
  --answers "$answers_path" \
  --decision-attempt-log "$result_root/logs/$domain-$arm.decision.attempts.jsonl" \
  --answer-attempt-log "$result_root/logs/$domain-$arm.answer.attempts.jsonl" \
  --decision-full-log "$result_root/logs/$domain-$arm.decision.full.log" \
  --answer-full-log "$answer_full_log" \
  --decision-manifest "$result_root/runtime/$domain-$arm-decision.manifest.json" \
  --answer-manifest "$result_root/runtime/$domain-$arm-answer.manifest.json" \
  --evaluation "$evaluation_path" \
  --repository-root "$repo_root" \
  --result-tag glm4-9b \
  --served-model glm4-9b \
  --domain "$domain" \
  --arm "$arm" \
  --expected-count "$expected" \
  --output "$audit_path"
