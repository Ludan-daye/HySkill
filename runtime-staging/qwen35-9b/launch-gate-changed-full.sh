#!/usr/bin/env bash
set -euo pipefail

repo_root=/root/HySkill-baseline-run-20260724
result_root="$repo_root/results/baselines-runtime-matched-v1/qwen35-9b"
instance_root="$repo_root/external/SR-Agents/data/bench/instances"
corpus_path="$repo_root/external/SR-Agents/data/bench/corpus/corpus.json"
python_path=/root/HySkill-k2-main-20260723/runtime/vllm-env/bin/python
api_base=http://127.0.0.1:8000/v1
audit_root="$result_root/gate-audit"
export PYTHONPATH="$repo_root:$repo_root/external/SR-Agents/src"
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

domains=(theoremqa medcalcbench)
arms=(routed_gated routed_gated)
audit_dirs=(theoremqa-routed medcalcbench-routed)
pids=()
for index in "${!domains[@]}"; do
  domain=${domains[$index]}
  arm=${arms[$index]}
  audit_dir=${audit_dirs[$index]}
  "$python_path" "$repo_root/scripts/run_runtime_matched_gate_answers.py" \
    --instances "$instance_root/$domain.json" \
    --corpus "$corpus_path" \
    --gate-audit "$audit_root/$audit_dir/audit.json" \
    --gate-decisions "$audit_root/$audit_dir/new-decisions.jsonl" \
    --gate-rerun "$audit_root/$audit_dir/rerun-ids.json" \
    --runtime-manifest "$result_root/runtime/$domain-$arm.manifest.json" \
    --repository-root "$repo_root" \
    --output "$result_root/gate-rerun/$domain-$arm.answers.jsonl" \
    --usage-log "$result_root/logs/$domain-$arm.gate-rerun.usage.jsonl" \
    --attempt-log "$result_root/logs/$domain-$arm.gate-rerun.attempts.jsonl" \
    --api-base "$api_base" \
    --workers 3 \
    --max-new-records 0 \
    >"$result_root/logs/$domain-$arm.gate-rerun.full.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done
