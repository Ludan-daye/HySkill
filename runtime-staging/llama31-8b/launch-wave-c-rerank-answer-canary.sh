#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'Usage: %s DOMAIN\n' "$0" >&2
  exit 2
fi
domain=$1
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

repo_root=/root/HySkill-baseline-runtime-matched-20260724/repo
source_root=/root/HySkill-k-run-20260723
result_root=/root/HySkill-baseline-runtime-matched-20260724-results/llama31-8b
instance_root="$source_root/external/SR-Agents/data/bench/instances"
corpus_path="$source_root/external/SR-Agents/data/bench/corpus/corpus.json"
python_path="$source_root/.venv/bin/python"
api_base=http://127.0.0.1:8000/v1
export PYTHONPATH="$repo_root:$repo_root/external/SR-Agents/src"
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

"$python_path" "$repo_root/scripts/run_runtime_matched_rerank_answers.py" \
  --instances "$instance_root/$domain.json" \
  --corpus "$corpus_path" \
  --decisions "$result_root/decisions/$domain-always_rerank.jsonl" \
  --output "$result_root/answers/$domain-always_rerank.jsonl" \
  --attempt-log "$result_root/logs/$domain-always_rerank.answer.attempts.jsonl" \
  --runtime-manifest "$result_root/runtime/$domain-always_rerank-answer.manifest.json" \
  --result-tag llama31-8b \
  --model llama31-8b \
  --api-base "$api_base" \
  --domain "$domain" \
  --workers 1 \
  --max-new-records 5 \
  --sragents-checkout "$repo_root/external/SR-Agents" \
  --repository-root "$repo_root" \
  --sragents-revision 277fd8d2bbd7d3b81a5cf4ffa6e87e18c7906e4f \
  >"$result_root/logs/$domain-always_rerank.answer.canary.log" 2>&1

observed=$(wc -l <"$result_root/answers/$domain-always_rerank.jsonl")
if [[ "$observed" -ne 5 ]]; then
  printf \
    'Rerank answer canary count mismatch: domain=%s expected=5 observed=%s\n' \
    "$domain" \
    "$observed" >&2
  exit 1
fi
